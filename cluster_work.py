#############################################################################
#
# ClusterWork
#
# A framework to run experiments on an computing cluster.
#
# Based on the Python Experiment Suite by Thomas Rückstiess. (see expsuite_LICENSE)
# Licensed under the modified BSD License.
#
# Copyright 2017 - Gregor Gebhardt
#
#############################################################################

import argparse
import collections
import itertools
import os
import re
import socket
import abc
from copy import deepcopy

import job_stream.inline
import pandas as pd
import yaml


# from multiprocessing import Pool, cpu_count


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = deep_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def progress(config, rep):
    """ Helper function to calculate the progress made on one experiment. """
    fullpath = os.path.join(config['path'], config['name'])
    logname = os.path.join(fullpath, '%i.log' % rep)
    if os.path.exists(logname):
        logfile = open(logname, 'r')
        lines = logfile.readlines()
        logfile.close()
        return int(100 * len(lines) / config['iterations'])
    else:
        return 0


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, collections.MutableSequence):
            keys = map(lambda i: new_key + "_" + str(i), range(len(v)))
            items.extend(zip(keys, v))
        else:
            items.append((new_key, v))
    return dict(items)


def get_experiments(path='.'):
    """ go through all subdirectories starting at path and return the experiment
        identifiers (= directory names) of all existing experiments. A directory
        is considered an experiment if it contains a experiment.cfg file.
    """
    exps = []
    for dp, dn, fn in os.walk(path):
        if 'experiment.yml' in fn:
            subdirs = [os.path.join(dp, d) for d in os.listdir(dp) if os.path.isdir(os.path.join(dp, d))]
            if all(map(lambda s: get_experiments(s) == [], subdirs)):
                exps.append(dp)
    return exps


def get_experiment_config(path, cfgname='experiment.yml'):
    """ reads the parameters of the experiment (= path) given.
    """
    with open(os.path.join(path, cfgname), 'r') as f:
        config = yaml.load(f)
        return config


def get_experiment_directories(name, path='.'):
    """ given an experiment name (used in section titles), this function
        returns all subdirectories that contain an experiment with that name.
    """
    experiments = []
    for dir_path, dir_names, filenames in os.walk(path):
        if 'experiment.yml' in filenames:
            with open(os.path.join(dir_path, 'experiment.yml')) as f:
                for d in yaml.load_all(f):
                    if 'name' in d and d['name'] == name:
                        experiments.append(dir_path)
    return experiments


class ClusterWork:
    # change this in subclass, if you support restoring state on iteration level
    _restore_supported = False
    _default_params = {}
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _VERBOSE = False

    _parser = argparse.ArgumentParser()
    _parser.add_argument('config', metavar='CONFIG.yml', type=argparse.FileType('r'))
    _parser.add_argument('-c', '--cluster', action='store_true')
    _parser.add_argument('-d', '--delete', action='store_true')
    _parser.add_argument('-e', '--experiments', nargs='+')
    _parser.add_argument('-v', '--verbose', action='store_true')
    _parser.add_argument('--no_compare_configs', action='store_true')
    _parser.add_argument('--restart_full_repetitions', action='store_true')

    @classmethod
    def _init_experiments(cls, config_default, config_experiments, options):
        """allows subclasses to modify the default configuration or the configuration of the experiments.

        :param config_default: the default configuration document
        :param config_experiments: the documents of the experiment configurations
        :param options: the options read by the argument parser
        :return: has to return the default configuration and the configuration of the experiments.
        """
        return config_default, config_experiments

    @classmethod
    def __init_experiments(cls):
        """initializes the experiment by loading the configuration file and creating the directory structure.
        :return:
        """
        options = cls._parser.parse_args()

        try:
            _config_documents = [*yaml.load_all(options.config)]
        except IOError:
            raise SystemExit('config file %s not found.' % options.config)

        if _config_documents[0]['name'].lower() == 'DEFAULT'.lower():
            config_default = _config_documents[0]
            config_experiments = _config_documents[1:]
        else:
            config_default = dict()
            config_experiments = _config_documents

        # allow subclasses to modify the configurations
        config_default, config_experiments = cls._init_experiments(config_default, config_experiments, options)

        # iterate over experiments and merge with default configuration
        _config_experiments_tmp = []
        for _config_e in config_experiments:
            if not options.experiments or _config_e['name'] in options.experiments:
                _config = deepcopy(config_default)
                deep_update(_config, _config_e)
                _config_experiments_tmp.append(_config)

                # check for all required param keys
                missing_keys = [key for key in ['name', 'path', 'repetitions', 'iterations'] if key not in _config]
                if missing_keys:
                    raise IncompleteConfigurationError(
                        'config does not contian all required keys: {}'.format(missing_keys))

        config_experiments = _config_experiments_tmp
        del _config_experiments_tmp

        config_experiments_w_expanded_params = cls.__expand_param_list(config_experiments)

        # create directories, write config files
        for _config in config_experiments_w_expanded_params:
            effective_params = dict()
            deep_update(effective_params, cls._default_params)
            deep_update(effective_params, _config['params'])
            _config['params'] = effective_params
            cls.__create_experiment_directory(_config, options.delete)

        # check for existing results and skip experiment
        if not options.delete:
            for _config in config_experiments_w_expanded_params:
                # check if experiment exists and has finished
                if cls.__experiment_has_finished(_config):
                    if options.no_compare_config:
                        # remove experiment from list
                        config_experiments_w_expanded_params.remove(_config)
                    else:
                        # check if experiment configs are identical
                        if cls.__experiments_exists_identically(_config):
                            # remove experiment from list
                            config_experiments_w_expanded_params.remove(_config)

        return config_experiments_w_expanded_params

    @classmethod
    def __setup_work_flow(cls, w: job_stream.inline.Work):

        @w.init
        def _work_init():
            """initializes the work, i.e., loads the configuration file, compiles the configuration documents from
            the file and the default configurations, expands grid and list experiments, and creates the directory
            structure.

            :return: a job_stream.inline.Multiple object with one configuration document for each experiment.
            """
            if cls._VERBOSE:
                print("[rank {}] In function '_work_init'".format(job_stream.inline.getRank()))

            work_list = cls.__init_experiments()
            return job_stream.inline.Multiple(work_list)

        @w.frame
        def _start_experiment(store, config):
            """starts an experiment frame for each experiment. Inside the frame one job is started for each repetition.

            :param store: object to store results
            :param config: the configuration document for the experiment
            """
            if cls._VERBOSE:
                print("[rank {}] In function '_start_experiment'".format(job_stream.inline.getRank()))

            if not hasattr(store, 'index'):
                store.index = pd.MultiIndex.from_product([range(config['repetitions']), range(config['iterations'])])
                store.index.set_names(['r', 'i'], inplace=True)
                store.config = config
                # create work list
                work_list = [job_stream.inline.Args(cls(), config, i) for i in range(config['repetitions'])]

                return job_stream.inline.Multiple(work_list)

        @w.job
        def _run_repetition(suite, exp_config, r):
            """runs a single repetition of the experiment by calling run_rep(exp_config, r) on the instance of
            PyExperimentSuite.

            :param suite: the instance of PyExperimentSuite
            :param exp_config: the configuration document for the experiment
            :param r: the repetition number
            """
            if cls._VERBOSE:
                print("[rank {}] In function '_run_repetition' on host {}".format(job_stream.inline.getRank(),
                                                                                  socket.gethostname()))

            repetition_results = suite.__run_rep(exp_config, r)

            return repetition_results

        @w.frameEnd
        def _end_experiment(store, repetition_results):
            """collects the results from the individual repetitions in a pandas.DataFrame.

            :param store: the store object from the frame.
            :param repetition_results: the pandas.DataFrame with the results of the repetition.
            """
            if cls._VERBOSE:
                print("[rank {}] In function '_end_experiment'".format(job_stream.inline.getRank()))

            if not hasattr(store, 'results'):
                store.results = pd.DataFrame(index=store.index, columns=repetition_results.columns)

            store.results.update(repetition_results)

        @w.result()
        def _work_results(store):
            """takes the resulting store object and writes the pandas.DataFrame to a file results.csv in the
            experiment folder.

            :param store: the store object emitted from the frame.
            """
            if cls._VERBOSE:
                print("[rank {}] In function '_work_results'".format(job_stream.inline.getRank()))

            with open(os.path.join(store.config['path'], 'results.csv'), 'w') as results_file:
                store.results.to_csv(results_file, **cls._pandas_to_csv_options)

    @classmethod
    def run(cls):
        """ starts the experiments as given in the config file. """
        options = cls._parser.parse_args()
        cls._VERBOSE = options.verbose
        if cls._VERBOSE:
            print("starting {} with the following options:".format(cls.__name__))
            print(options)

        if options.cluster:
            # without setting the useMultiprocessing flag to False, we get errors on the cluster
            with job_stream.inline.Work(useMultiprocessing=False) as w:
                cls.__setup_work_flow(w)
                print('[rank {}] Work has been setup...'.format(job_stream.getRank()))

                # w.run()
        else:
            # TODO needs to be implemented
            # if -b, -B or -p option is set, only show information, don't
            # start the experiments
            # if self.options.browse or self.options.browse_big or self.options.progress:
            #     self.browse()
            #     raise SystemExit

            # TODO add multiprocessing here
            config_experiments_w_expanded_params = cls.__init_experiments()
            for experiment in config_experiments_w_expanded_params:
                instance = cls()

                # expand config_list_w_expanded_params for all repetitions and add self and rep number
                repetitions_list = []
                num_repetitions = experiment['repetitions']
                repetitions_list.extend(zip([instance] * num_repetitions,
                                            [experiment] * num_repetitions,
                                            range(num_repetitions)))

                results = dict()
                for repetition in repetitions_list:
                    result = ClusterWork.__run_rep(*repetition)
                    results[repetition[2]] = result

                _index = pd.MultiIndex.from_product([range(experiment['repetitions']),
                                                     range(experiment['iterations'])],
                                                    names=['r', 'i'])
                result_frame = pd.DataFrame(index=_index, columns=results[0].columns, dtype=float)
                for i in results:
                    result_frame.update(results[i])
                with open(os.path.join(experiment['path'], 'results.csv'), 'w') as results_file:
                    result_frame.to_csv(results_file, **cls._pandas_to_csv_options)

    def __run_rep(self, config, rep) -> pd.DataFrame:
        """ run a single repetition including directory creation, log files, etc. """
        log_filename = os.path.join(config['log_path'], 'rep_{}.csv'.format(rep))

        # check if log-file for repetition exists
        repetition_has_finished, n_finished_reps, results = self.__repetition_has_completed(config, rep)

        # skip repetition if it has finished
        if repetition_has_finished:
            if self._VERBOSE:
                print('Repetition {} of experiment {} has finished before. Skipping...'.format(rep, config['name']))
            return results

        self.reset(config, rep)

        # if not completed but some iterations have finished, check for restart capabilities
        if self._restore_supported and n_finished_reps > 0:
            if self._VERBOSE:
                print('Repetition {} of experiment {} has started before. Restarting at {}.'.format(rep,
                                                                                                    config['name'],
                                                                                                    n_finished_reps))
            start_iteration = n_finished_reps
            self.restore_state(config, rep, start_iteration)
        else:
            start_iteration = 0
            results = None

        for it in range(start_iteration, config['iterations']):
            it_result = self.iterate(config, rep, it)
            flat_it_result = flatten_dict(it_result)

            if results is None:
                results = pd.DataFrame(index=pd.MultiIndex.from_product([[rep], range(config['iterations'])],
                                                                        names=['r', 'i']),
                                       columns=flat_it_result.keys(), dtype=float)

            results.loc[(rep, it)] = flat_it_result

            # write first line with header
            if it == 0:
                results.iloc[[it]].to_csv(log_filename, mode='w', header=True, **self._pandas_to_csv_options)
            else:
                results.iloc[[it]].to_csv(log_filename, mode='a', header=False, **self._pandas_to_csv_options)

        return results

    @staticmethod
    def __expand_param_list(config_list):
        """ expands the parameters list according to one of these schemes:
            grid: every list item is combined with every other list item
            list: every n-th list item of parameter lists are combined
        """
        # for one single experiment, still wrap it in list
        if type(config_list) == dict:
            config_list = [config_list]

        # get all options that are iteratable and build all combinations (grid) or tuples (list)
        expanded_config_list = []
        for config in config_list:
            if 'grid' in config or 'list' in config:
                if 'grid' in config:
                    # if we want a grid then we choose the product of all parameters
                    iter_fun = itertools.product
                else:
                    # if we want a list then we zip the parameters together
                    iter_fun = zip

                # create a new config for each parameter setting
                for values in iter_fun(*[v for v in config['grid'].values()]):
                    # create config file for
                    _config = deepcopy(config)
                    del _config['grid']
                    _converted_name = str(zip(config['grid'].keys(), map(str, values)))
                    _converted_name = re.sub("[' \[\],()]", '', _converted_name)
                    _config['path'] = os.path.join(config['path'], config['name'], _converted_name)
                    _config['name'] += '_' + _converted_name
                    _config['log_path'] = os.path.join(_config['path'], 'log')
                    for i, ip in enumerate(config['grid'].keys()):
                        _config['params'][ip] = values[i]
                    expanded_config_list.append(_config)
            else:
                _config = deepcopy(config)
                _config['path'] = os.path.join(config['path'], config['name'])
                _config['log_path'] = os.path.join(_config['path'], 'log')
                expanded_config_list.append(_config)

        return expanded_config_list

    @staticmethod
    def __convert_param_to_dirname(param):
        """ Helper function to convert a parameter value to a valid directory name. """
        if type(param) == str:
            return param
        else:
            return re.sub("0+$", '0', '%f' % param)

    @staticmethod
    def __create_experiment_directory(config, delete_old_files=False):
        """ creates a subdirectory for the experiment, and deletes existing
            files, if the delete flag is true. then writes the current
            experiment.cfg file in the folder.
        """
        # create experiment path and subdir
        os.makedirs(config['path'], exist_ok=True)

        # delete old histories if --del flag is active
        if delete_old_files:
            os.system('rm -rf {}/*'.format(config['path']))

        # create a directory for the log path
        os.makedirs(config['log_path'], exist_ok=True)

        # write a config file for this single exp. in the folder
        ClusterWork.__write_config_file(config)

    @staticmethod
    def __write_config_file(config):
        """ write a config file for this single exp in the folder path.
        """
        with open(os.path.join(config['path'], 'experiment.yml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def __experiment_exists(config):
        return os.path.exists(os.path.join(config['path'], 'experiment.yml'))

    @staticmethod
    def __experiments_exists_identically(config):
        if ClusterWork.__experiment_exists(config):
            with open(os.path.join(config['path'], 'experiment.yml'), 'r') as f:
                dumped_config = yaml.load(f)
                return dumped_config == config

        return False

    @staticmethod
    def __experiment_has_finished(config):
        return os.path.exists(os.path.join(config['path'], 'results.csv'))

    @staticmethod
    def __load_repetition_results(config, rep):
        log_filename = os.path.join(config['log_path'], 'rep_{}.csv'.format(rep))

        if os.path.exists(log_filename):
            log_df = pd.read_csv(log_filename, sep='\t')
            log_df.set_index(keys=['r', 'i'], inplace=True)

            return log_df

        return None

    @staticmethod
    def __repetition_has_completed(config, rep) -> (bool, int, pd.DataFrame):
        log_df = ClusterWork.__load_repetition_results(config, rep)

        if log_df is not None:
            # if repetition has completed
            return log_df.shape[0] == config['iterations'], log_df.shape[0], log_df

        return False, 0, None


    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        """ needs to be implemented by subclass. """
        pass

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        """ needs to be implemented by subclass. """
        pass

    def save_state(self, config: dict, rep: int, n: int) -> None:
        """ optionally can be implemented by subclass. """
        pass

    def restore_state(self, config: dict, rep: int, n: int) -> None:
        """ if the experiment supports restarting within a repetition
            (on iteration level), load necessary stored state in this 
            function. Otherwise, restarting will be done on repetition 
            level, deleting all unfinished repetitions and restarting 
            the experiments.
        """
        pass

        # def get_history(self, exp, rep, tags):
        #     """ returns the whole history for one experiment and one repetition.
        #         tags can be a string or a list of strings. if tags is a string,
        #         the history is returned as list of values, if tags is a list of
        #         strings or 'all', history is returned as a dictionary of lists
        #         of values.
        #     """
        #     params = self.get_config(exp)
        #
        #     if params == None:
        #         raise SystemExit('experiment %s not found.' % exp)
        #
        #         # make list of tags, even if it is only one
        #     if tags != 'all' and not hasattr(tags, '__iter__'):
        #         tags = [tags]
        #
        #     results = {}
        #     logfile = os.path.join(exp, '%i.log' % rep)
        #     try:
        #         f = open(logfile)
        #     except IOError:
        #         if len(tags) == 1:
        #             return []
        #         else:
        #             return {}
        #
        #     for line in f:
        #         pairs = line.split()
        #         for pair in pairs:
        #             tag, val = pair.split(':')
        #             if tags == 'all' or tag in tags:
        #                 if not tag in results:
        #                     try:
        #                         results[tag] = [eval(val)]
        #                     except (NameError, SyntaxError):
        #                         results[tag] = [val]
        #                 else:
        #                     try:
        #                         results[tag].append(eval(val))
        #                     except (NameError, SyntaxError):
        #                         results[tag].append(val)
        #
        #     f.close()
        #     if len(results) == 0:
        #         if len(tags) == 1:
        #             return []
        #         else:
        #             return {}
        #             # raise ValueError('tag(s) not found: %s'%str(tags))
        #     if len(tags) == 1:
        #         return results[results.keys()[0]]
        #     else:
        #         return results
        #
        # def get_history_tags(self, exp, rep=0):
        #     """ returns all available tags (logging keys) of the given experiment
        #         repetition.
        #
        #         Note: Technically, each repetition could have different
        #         tags, therefore the rep number can be passed in as parameter,
        #         even though usually all repetitions have the same tags. The default
        #         repetition is 0 and in most cases, can be omitted.
        #     """
        #     history = self.get_history(exp, rep, 'all')
        #     return history.keys()
        #
        # def get_value(self, exp, rep, tags, which='last'):
        #     """ Like get_history(..) but returns only one single value rather
        #         than the whole list.
        #         tags can be a string or a list of strings. if tags is a string,
        #         the history is returned as a single value, if tags is a list of
        #         strings, history is returned as a dictionary of values.
        #         'which' can be one of the following:
        #             last: returns the last value of the history
        #              min: returns the minimum value of the history
        #              max: returns the maximum value of the history
        #                #: (int) returns the value at that index
        #     """
        #     history = self.get_history(exp, rep, tags)
        #
        #     # empty histories always return None
        #     if len(history) == 0:
        #         return None
        #
        #     # distinguish dictionary (several tags) from list
        #     if type(history) == dict:
        #         for h in history:
        #             if which == 'last':
        #                 history[h] = history[h][-1]
        #             if which == 'min':
        #                 history[h] = min(history[h])
        #             if which == 'max':
        #                 history[h] = max(history[h])
        #             if type(which) == int:
        #                 history[h] = history[h][which]
        #         return history
        #
        #     else:
        #         if which == 'last':
        #             return history[-1]
        #         if which == 'min':
        #             return min(history)
        #         if which == 'max':
        #             return max(history)
        #         if type(which) == int:
        #             return history[which]
        #         else:
        #             return None
        #
        # def get_values_fix_params(self, exp, rep, tag, which='last', **kwargs):
        #     """ this function uses get_value(..) but returns all values where the
        #         subexperiments match the additional kwargs arguments. if alpha=1.0,
        #         beta=0.01 is given, then only those experiment values are returned,
        #         as a list.
        #     """
        #     subexps = self.get_exps(exp)
        #     tagvalues = ['%s%s' % (k, convert_param_to_dirname(kwargs[k])) for k in kwargs]
        #
        #     values = [self.get_value(se, rep, tag, which) for se in subexps if all(map(lambda tv: tv in se, tagvalues))]
        #     params = [self.get_config(se) for se in subexps if all(map(lambda tv: tv in se, tagvalues))]
        #
        #     return values, params
        #
        # def get_histories_fix_params(self, exp, rep, tag, **kwargs):
        #     """ this function uses get_history(..) but returns all histories where the
        #         subexperiments match the additional kwargs arguments. if alpha=1.0,
        #         beta = 0.01 is given, then only those experiment histories are returned,
        #         as a list.
        #     """
        #     subexps = self.get_exps(exp)
        #     tagvalues = [re.sub("0+$", '0', '%s%f' % (k, kwargs[k])) for k in kwargs]
        #
        #     histories = [self.get_history(se, rep, tag) for se in subexps if all(map(lambda tv: tv in se, tagvalues))]
        #     params = [self.get_config(se) for se in subexps if all(map(lambda tv: tv in se, tagvalues))]
        #
        #     return histories, params
        #
        # def get_histories_over_repetitions(self, exp, tags, aggregate):
        #     """ this function gets all histories of all repetitions using get_history() on the given
        #         tag(s), and then applies the function given by 'aggregate' to all corresponding values
        #         in each history over all iterations. Typical aggregate functions could be 'mean' or
        #         'max'.
        #     """
        #     params = self.get_config(exp)
        #
        #     # explicitly make tags list in case of 'all'
        #     if tags == 'all':
        #         tags = self.get_history(exp, 0, 'all').keys()
        #
        #     # make list of tags if it is just a string
        #     if not hasattr(tags, '__iter__'):
        #         tags = [tags]
        #
        #     results = {}
        #     for tag in tags:
        #         # get all histories
        #         histories = np.zeros((params['repetitions'], params['iterations']))
        #         skipped = []
        #         for i in range(params['repetitions']):
        #             try:
        #                 histories[i, :] = self.get_history(exp, i, tag)
        #             except ValueError:
        #                 h = self.get_history(exp, i, tag)
        #                 if len(h) == 0:
        #                     # history not existent, skip it
        #                     print('warning: history %i has length 0 (expected: %i). it will be skipped.' % (
        #                         i, params['iterations']))
        #                     skipped.append(i)
        #                 elif len(h) > params['iterations']:
        #                     # if history too long, crop it
        #                     print('warning: history %i has length %i (expected: %i). it will be truncated.' % (
        #                         i, len(h), params['iterations']))
        #                     h = h[:params['iterations']]
        #                     histories[i, :] = h
        #                 elif len(h) < params['iterations']:
        #                     # if history too short, crop everything else
        #                     print(
        #                         'warning: history %i has length %i (expected: %i). all other histories will be truncated.' %
        #                         (i, len(h), params['iterations']))
        #                     params['iterations'] = len(h)
        #                     histories = histories[:, :params['iterations']]
        #                     histories[i, :] = h
        #
        #         # remove all rows that have been skipped
        #         histories = np.delete(histories, skipped, axis=0)
        #         params['repetitions'] -= len(skipped)
        #
        #         # calculate result from each column with aggregation function
        #         aggregated = np.zeros(params['iterations'])
        #         for i in range(params['iterations']):
        #             aggregated[i] = aggregate(histories[:, i])
        #
        #         # if only one tag is requested, return list immediately, otherwise append to dictionary
        #         if len(tags) == 1:
        #             return aggregated
        #         else:
        #             results[tag] = aggregated
        #
        #     return results
        #
        # def browse(self):
        #     """ go through all subfolders (starting at '.') and return information
        #         about the existing experiments. if the -B option is given, all
        #         parameters are shown, -b only displays the most important ones.
        #         this function does *not* execute any experiments.
        #     """
        #     for d in self.get_exps('.'):
        #         config = self.get_config(d)
        #         basename = config['name'].split('/')[0]
        #         # if -e option is used, only show requested experiments
        #         if self.options.experiments and basename not in self.options.experiments:
        #             continue
        #
        #         fullpath = os.path.join(config['path'], config['name'])
        #
        #         # calculate progress
        #         prog = 0
        #         for i in range(config['repetitions']):
        #             prog += progress(config, i)
        #         prog /= config['repetitions']
        #
        #         # if progress flag is set, only show the progress bars
        #         if self.options.progress:
        #             bar = "["
        #             bar += "=" * int(prog / 4)
        #             bar += " " * int(25 - prog / 4)
        #             bar += "]"
        #             print('%3i%% %27s %s' % (prog, bar, d))
        #             continue
        #
        #         print('%16s %s' % ('experiment', d))
        #
        #         try:
        #             minfile = min(
        #                 (os.path.join(dirname, filename)
        #                  for dirname, dirnames, filenames in os.walk(fullpath)
        #                  for filename in filenames
        #                  if filename.endswith(('.log', '.cfg'))),
        #                 key=lambda fn: os.stat(fn).st_mtime)
        #
        #             maxfile = max(
        #                 (os.path.join(dirname, filename)
        #                  for dirname, dirnames, filenames in os.walk(fullpath)
        #                  for filename in filenames
        #                  if filename.endswith(('.log', '.cfg'))),
        #                 key=lambda fn: os.stat(fn).st_mtime)
        #         except ValueError:
        #             print('         started %s' % 'not yet')
        #
        #         else:
        #             print('         started %s' % time.strftime('%Y-%m-%d %H:%M:%S',
        #                                                         time.localtime(os.stat(minfile).st_mtime)))
        #             print('           ended %s' % time.strftime('%Y-%m-%d %H:%M:%S',
        #                                                         time.localtime(os.stat(maxfile).st_mtime)))
        #
        #         for k in ['repetitions', 'iterations']:
        #             print('%16s %s' % (k, config[k]))
        #
        #         print('%16s %i%%' % ('progress', prog))
        #
        #         if self.options.browse_big:
        #             # more verbose output
        #             for p in [p for p in config if p not in ('repetitions', 'iterations', 'path', 'name')]:
        #                 print('%16s %s' % (p, config[p]))
        #
        #         print()


class IncompleteConfigurationError(Exception):
    pass
