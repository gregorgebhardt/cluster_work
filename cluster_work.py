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

import abc
import argparse
import collections
import itertools
import os
import sys
import re
import socket
import time
from copy import deepcopy
import fnmatch
import traceback

import pandas as pd
import yaml
import logging

_logging_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
_logging_std_handler = logging.StreamHandler(sys.stdout)
_logging_std_handler.setFormatter(_logging_formatter)
_logging_std_handler.setLevel(logging.DEBUG)
_logging_std_handler.addFilter(lambda lr: lr.levelno < logging.WARNING)
_logging_err_handler = logging.StreamHandler(sys.stderr)
_logging_err_handler.setFormatter(_logging_formatter)
_logging_err_handler.setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, handlers=[_logging_std_handler, _logging_err_handler])
# logging.root.addHandler(_logging_std_handler)
# logging.root.addHandler(_logging_err_handler)

_logger = logging.getLogger('cluster_work')


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = deep_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


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


def flatten_dict_to_tuple_keys(d: collections.MutableMapping):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            sub_dict = flatten_dict_to_tuple_keys(v)
            flat_dict.update({(k, *sk): sv for sk, sv in sub_dict.items()})

        elif isinstance(v, collections.MutableSequence):
            flat_dict[(k,)] = v

    return flat_dict


def insert_deep_dictionary(d: collections.MutableMapping, t: tuple, value):
    if type(t) is tuple:
        if len(t) == 1:  # tuple contains only one key
            d[t[0]] = value
        else:  # tuple contains more than one key
            if t[0] not in d:
                d[t[0]] = dict()
            insert_deep_dictionary(d[t[0]], t[1:], value)
    else:
        d[t] = value


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


class ClusterWork(object):
    # change this in subclass, if you support restoring state on iteration level
    _restore_supported = False
    _default_params = {}
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _VERBOSE = False

    _parser = argparse.ArgumentParser()
    _parser.add_argument('config', metavar='CONFIG.yml', type=argparse.FileType('r'))
    _parser.add_argument('-c', '--cluster', action='store_true',
                         help='runs the experiment in cluster mode, i.e., uses the openmpi features')
    _parser.add_argument('-d', '--delete', action='store_true',
                         help='CAUTION deletes results of previous runs')
    _parser.add_argument('-o', '--overwrite', action='store_true',
                         help='CAUTION overwrites results of previous runs if config has changed')
    _parser.add_argument('-e', '--experiments', nargs='+',
                         help='allows to specify which experiments should be run')
    _parser.add_argument('-v', '--verbose', action='store_true',
                         help='DEPRECATED, use log-level instead')
    _parser.add_argument('-p', '--progress', action='store_true',
                         help='outputs the progress of the experiment and exits')
    _parser.add_argument('-P', '--full_progress', action='store_true',
                         help='outputs a more detailed progress of the experiment and exits')
    _parser.add_argument('-g', '--no_gui', action='store_true',
                         help='tells the experiment to not use any feature that requires a GUI')
    _parser.add_argument('--skip_ignore_config', action='store_true')
    _parser.add_argument('--restart_full_repetitions', action='store_true')
    _parser.add_argument('--log_level', nargs='?', default='INFO',
                         help='sets the log-level for the output of ClusterWork')

    __runs_on_cluster = False

    def __init__(self):
        self.__log_path_rep = None
        self.__log_path_rep_exists = False
        self.__log_path_it = None
        self.__log_path_it_exists = False

    @property
    def _log_path_rep(self):
        if not self.__log_path_rep_exists:
            os.makedirs(self.__log_path_rep, exist_ok=True)
            self.__log_path_rep_exists = True
        return self.__log_path_rep

    @_log_path_rep.setter
    def _log_path_rep(self, log_path_rep: str):
        if os.path.exists(log_path_rep):
            if not os.path.isdir(log_path_rep):
                raise NotADirectoryError("The log path {} exists but is not a directory".format(log_path_rep))
            self.__log_path_rep_exists = True
        else:
            self.__log_path_rep_exists = False
        self.__log_path_rep = log_path_rep

    @property
    def _log_path_it(self):
        if not self.__log_path_it_exists:
            os.makedirs(self.__log_path_it, exist_ok=True)
            self.__log_path_it_exists = True
        return self.__log_path_it

    @_log_path_it.setter
    def _log_path_it(self, log_path_it: str):
        if os.path.exists(log_path_it):
            if not os.path.isdir(log_path_it):
                raise NotADirectoryError("The log path {} exists but is not a directory".format(log_path_it))
            self.__log_path_it_exists = True
        else:
            self.__log_path_it_exists = False
        self.__log_path_it = log_path_it

    @classmethod
    def _init_experiments(cls, config_default, config_experiments, options):
        """allows subclasses to modify the default configuration or the configuration of the experiments.

        :param config_default: the default configuration document
        :param config_experiments: the documents of the experiment configurations
        :param options: the options read by the argument parser
        """
        pass

    @classmethod
    def __load_experiments(cls, config_file, experiment_selectors=None):
        """loads all experiment configurations from the given file, merges them with the default configuration and
        expands list or grid parameters

        :config_file: path to the configuration yaml for the experiments that should be loaded.
        :experiment_selectors: list of experiment names. Only the experiments with a name in this list will be loaded.
        :return: returns the experiment configurations
        """
        try:
            _config_documents = [*yaml.load_all(config_file)]
        except IOError:
            raise SystemExit('config file %s not found.' % config_file)

        if _config_documents[0]['name'].lower() == 'DEFAULT'.lower():
            default_config = _config_documents[0]
            experiments_config = _config_documents[1:]
        else:
            default_config = dict()
            experiments_config = _config_documents

        # iterate over experiments and compute effective configuration and parameters
        effective_experiments = []
        for _config_e in experiments_config:
            if not experiment_selectors or _config_e['name'] in experiment_selectors:
                # merge config with default config from yaml file
                _effective_config = deepcopy(default_config)
                deep_update(_effective_config, _config_e)

                # merge params with default params from subclass
                _effective_params = dict()
                deep_update(_effective_params, cls._default_params)
                deep_update(_effective_params, _effective_config['params'])
                _effective_config['params'] = _effective_params

                effective_experiments.append(_effective_config)

                # check for all required param keys
                required_keys = ['name', 'path', 'repetitions', 'iterations']
                missing_keys = [key for key in required_keys if key not in _effective_config]
                if missing_keys:
                    raise IncompleteConfigurationError(
                        'config does not contain all required keys: {}'.format(missing_keys))

        expanded_experiments = cls.__expand_param_list(effective_experiments)

        return expanded_experiments

    @classmethod
    def __init_experiments(cls, config_file, experiments=None, delete_old=False, ignore_config_for_skip=False,
                           overwrite_old=False):
        """initializes the experiment by loading the configuration file and creating the directory structure.
        :return:
        """
        expanded_experiments = cls.__load_experiments(config_file, experiments)

        # check for finished experiments
        skip_experiments = []
        clear_experiments = []
        if not delete_old:
            for _config in expanded_experiments:
                # check if experiment exists and has finished
                if cls.__experiment_has_finished(_config):
                    if ignore_config_for_skip:
                        # remove experiment from list
                        skip_experiments.append(_config)
                        _logger.info('Experiment {} has finished before. Skipping...'.format(_config['name']))
                    else:
                        # check if experiment configs are identical
                        if cls.__experiment_exists_identically(_config):
                            # remove experiment from list
                            skip_experiments.append(_config)
                            _logger.info('Experiment {} has finished identically before. '
                                         'Skipping...'.format(_config['name']))
                        elif overwrite_old:
                            _logger.warning('Experiment {} has finished before, but configuration has '
                                            'changed! Overwriting...'.format(_config['name']))
                            clear_experiments.append(_config)
                        else:
                            # remove experiment from list
                            skip_experiments.append(_config)
                            _logger.warning('Experiment {} has finished before, but configuration has '
                                            'changed! Skipping...'.format(_config['name']))
                            _logger.warning('--> To overwrite existing results, use the option -o/--overwrite')
                elif cls.__experiment_exists(_config) and not cls.__experiment_exists_identically(_config):
                    if cls.__experiment_has_finished_repetitions(_config):
                        if overwrite_old:
                            _logger.warning('Experiment {} has started before, but configuration has '
                                            'changed! Overwriting...'.format(_config['name']))
                            clear_experiments.append(_config)
                        else:
                            # remove experiment from list
                            skip_experiments.append(_config)
                            _logger.warning('Experiment {} has started before, but configuration has '
                                            'changed! Skipping...'.format(_config['name']))
                            _logger.warning('--> To overwrite existing results, use the option -o/--overwrite')
                    else:
                        _logger.info('Experiment {} has started before, but configuration has '
                                     'changed! Restarting since no results were found.'.format(_config['name']))

        run_experiments = [_config for _config in expanded_experiments if _config not in skip_experiments]

        if not run_experiments:
            SystemExit('No work to do...')

        for _config in run_experiments:
            cls.__create_experiment_directory(_config, delete_old or _config in clear_experiments)

        return run_experiments

    @classmethod
    def __setup_work_flow(cls, work):
        import job_stream.common
        import job_stream.inline

        @work.init
        def _work_init():
            """initializes the work, i.e., loads the configuration file, compiles the configuration documents from
            the file and the default configurations, expands grid and list experiments, and creates the directory
            structure.

            :return: a job_stream.inline.Multiple object with one configuration document for each experiment.
            """
            _logger.debug("[rank {}] In function '_work_init'".format(job_stream.inline.getRank()))

            options = cls._parser.parse_args()
            work_list = cls.__init_experiments(config_file=options.config, experiments=options.experiments,
                                               delete_old=options.delete,
                                               ignore_config_for_skip=options.skip_ignore_config,
                                               overwrite_old=options.overwrite)
            return job_stream.inline.Multiple(work_list)

        @work.frame
        def _start_experiment(store, config):
            """starts an experiment frame for each experiment. Inside the frame one job is started for each repetition.

            :param store: object to store results
            :param config: the configuration document for the experiment
            """
            _logger.debug("[rank {}] In function '_start_experiment' of {}".format(job_stream.inline.getRank(),
                                                                                   config['name']))
            _logger.debug("[rank {}] cpuCount: {}, hostCpuCount: {}".format(job_stream.common.getRank(),
                                                                            job_stream.common.getCpuCount(),
                                                                            job_stream.common.getHostCpuCount()))

            if not hasattr(store, 'index'):
                store.index = pd.MultiIndex.from_product([range(config['repetitions']), range(config['iterations'])])
                store.index.set_names(['r', 'i'], inplace=True)
                store.config = config
                # create work list
                work_list = [job_stream.inline.Args(cls(), config, i) for i in range(config['repetitions'])]

                return job_stream.inline.Multiple(work_list)

        @work.job
        def _run_repetition(suite, exp_config, r):
            """runs a single repetition of the experiment by calling run_rep(exp_config, r) on the instance of
            PyExperimentSuite.

            :param suite: the instance of PyExperimentSuite
            :param exp_config: the configuration document for the experiment
            :param r: the repetition number
            """
            _logger.debug("[rank {}] In function '_run_repetition' on host {}".format(job_stream.common.getRank(),
                                                                                      socket.gethostname()))

            repetition_results = suite.__run_rep(exp_config, r)

            return repetition_results

        @work.frameEnd
        def _end_experiment(store, repetition_results):
            """collects the results from the individual repetitions in a pandas.DataFrame.

            :param store: the store object from the frame.
            :param repetition_results: the pandas.DataFrame with the results of the repetition.
            """
            _logger.debug("[rank {}] In function '_end_experiment'".format(job_stream.inline.getRank()))

            if not hasattr(store, 'results'):
                store.results = pd.DataFrame(index=store.index, columns=repetition_results.columns)

            store.results.update(repetition_results)

        @work.result()
        def _work_results(store):
            """takes the resulting store object and writes the pandas.DataFrame to a file results.csv in the
            experiment folder.

            :param store: the store object emitted from the frame.
            """
            _logger.debug("[rank {}] In function '_work_results'".format(job_stream.inline.getRank()))

            with open(os.path.join(store.config['path'], 'results.csv'), 'w') as results_file:
                store.results.to_csv(results_file, **cls._pandas_to_csv_options)

    @classmethod
    def run(cls):
        """ starts the experiments as given in the config file. """
        options = cls._parser.parse_args()

        cls._NO_GUI = options.no_gui
        cls._VERBOSE = options.verbose
        cls._LOG_LEVEL = options.log_level.upper()
        logging.root.setLevel(level=cls._LOG_LEVEL)
        if cls._VERBOSE:
            logging.root.setLevel(level=logging.DEBUG)

        if options.progress:
            cls.show_progress(options.config, options.experiments)
            return

        if options.full_progress:
            cls.show_progress(options.config, options.experiments, full_progress=True)

        if options.cluster:
            import job_stream.common
            import job_stream.inline

            if job_stream.common.getRank() == 0:
                _logger.info("starting {} with the following options:".format(cls.__name__))
                for option, value in vars(options).items():
                    _logger.info("  - {}: {}".format(option, value))

            # without setting the useMultiprocessing flag to False, we get errors on the cluster
            with job_stream.inline.Work(useMultiprocessing=False) as w:
                cls.__setup_work_flow(w)
                cls.__runs_on_cluster = True
                _logger.debug('[rank {}] Work has been setup...'.format(job_stream.getRank()))

                # w.run()
        else:
            _logger.info("starting {} with the following options:".format(cls.__name__))
            for option, value in vars(options).items():
                _logger.info("  - {}: {}".format(option, value))

            config_experiments_w_expanded_params = cls.__init_experiments(config_file=options.config,
                                                                          experiments=options.experiments,
                                                                          delete_old=options.delete,
                                                                          ignore_config_for_skip=options.skip_ignore_config,
                                                                          overwrite_old=options.overwrite)
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

    @classmethod
    def show_progress(cls, config_file, experiment_selectors=None, full_progress=False):
        """ shows the progress of all experiments defined in the config_file.
        """
        experiments_config = cls.__load_experiments(config_file, experiment_selectors)

        for config in experiments_config:
            exp_progress, rep_progress = cls.__experiment_progress(config)

            # if progress flag is set, only show the progress bars
            bar = "["
            bar += "=" * round(25 * exp_progress)
            bar += " " * round(25 - 25 * exp_progress)
            bar += "]"
            print('{:5.1f}% {:27} {}'.format(exp_progress * 100, bar, config['name']))
            # print('%3.1f%% %27s %s' % (exp_progress * 100, bar, config['name']))

            if full_progress:
                for i, p in enumerate(rep_progress):
                    bar = "["
                    bar += "=" * round(25 * p)
                    bar += " " * round(25 - 25 * p)
                    bar += "]"
                    print('    |- {:2d} {:5.1f}% {:27}'.format(i, p * 100, bar))

                try:
                    minfile = min(
                        [os.path.join(dirname, filename) for dirname, dirnames, filenames in os.walk(config['path'])
                         for filename in filenames if filename.endswith(('.csv', '.yml'))],
                        key=lambda fn: os.stat(fn).st_mtime)

                    maxfile = max(
                        [os.path.join(dirname, filename) for dirname, dirnames, filenames in os.walk(config['path'])
                         for filename in filenames if filename.endswith(('.csv', '.yml'))],
                        key=lambda fn: os.stat(fn).st_mtime)
                except ValueError:
                    print('         started %s' % 'not yet')

                else:
                    print('         started %s' % time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(os.stat(minfile).st_mtime)))
                    print('           ended %s' % time.strftime('%Y-%m-%d %H:%M:%S',
                                                                time.localtime(os.stat(maxfile).st_mtime)))

                for k in ['repetitions', 'iterations']:
                    print('%16s %s' % (k, config[k]))

                print()

    def __run_rep(self, config, rep) -> pd.DataFrame:
        """ run a single repetition including directory creation, log files, etc. """
        # set configuration of this repetition
        self._name = config['name']
        self._repetitions = config['repetitions']
        self._iterations = config['iterations']
        self._path = config['path']
        self._log_path = config['log_path']
        self._log_path_rep = os.path.join(config['log_path'], '{:02d}'.format(rep), '')
        self._plotting = config['plotting'] if 'plotting' in config else False
        self._no_gui = not config['gui'] if 'gui' in config else self.__runs_on_cluster or self._NO_GUI
        self._seed = int(hash(self._name)) % int(1e6)

        # set params of this repetition
        self._params = config['params']
        self._rep = rep

        log_filename = os.path.join(self._log_path, 'rep_{}.csv'.format(rep))

        # check if log-file for repetition exists
        repetition_has_finished, n_finished_its, results = self.__repetition_has_completed(config, rep)

        # skip repetition if it has finished
        if repetition_has_finished:
            _logger.info('Repetition {} of experiment {} has finished before. '
                         'Skipping...'.format(rep, config['name']))
            return results

        self.reset(config, rep)

        # if not completed but some iterations have finished, check for restart capabilities
        if self._restore_supported and n_finished_its > 0:
            _logger.info('Repetition {} of experiment {} has started before. '
                         'Trying to restore at iteration {}.'.format(rep, config['name'], n_finished_its))
            # set start for iterations and restore state in subclass
            start_iteration = n_finished_its
            if self.restore_state(config, rep, start_iteration):
                _logger.info('Restoring iteration succeeded. Restarting from iteration {}.'.format(n_finished_its))
                results = pd.DataFrame(data=results,
                                       index=pd.MultiIndex.from_product([[rep], range(config['iterations'])],
                                                                        names=['r', 'i']),
                                       columns=results.columns, dtype=float)
            else:
                _logger.info('Restoring iteration NOT successful. Restarting from iteration 0.')
                start_iteration = 0
                results = None
        else:
            start_iteration = 0
            results = None

        # switch logging output to file
        file_handler_mode = 'a' if start_iteration else 'w'
        file_handler = logging.FileHandler(os.path.join(self._log_path_rep, 'log.txt'), file_handler_mode)
        file_handler.setLevel(self._LOG_LEVEL)
        file_handler.setFormatter(_logging_formatter)
        logging.root.addHandler(file_handler)
        if self.__runs_on_cluster:
            logging.root.removeHandler(_logging_std_handler)

        for it in range(start_iteration, config['iterations']):
            self._it = it

            # update iteration log directory
            self._log_path_it = os.path.join(config['log_path'], '{:02d}'.format(rep), '{:02d}'.format(it), '')

            # run iteration and get results
            try:
                it_result = self.iterate(config, rep, it)
            except ValueError or OverflowError or ZeroDivisionError or ArithmeticError or FloatingPointError:
                _logger.error('Experiment {} - Repetition {} - Iteration {}'.format(config['name'], rep, it))
                _logger.error(traceback.format_exception(*sys.exc_info()))
                self.finalize()
                return results

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

            if self._restore_supported:
                self.save_state(config, rep, it)

        self.finalize()

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
                    key = 'grid'
                else:
                    # if we want a list then we zip the parameters together
                    iter_fun = zip
                    key = 'list'

                # TODO add support for both list and grid

                # convert list/grid dictionary into flat dictionary, where the key is a tuple of the keys and the
                # value is the list of values
                tuple_dict = flatten_dict_to_tuple_keys(config[key])
                _param_names = ['.'.join(t) for t in tuple_dict]

                # create a new config for each parameter setting
                for values in iter_fun(*tuple_dict.values()):
                    # create config file for
                    _config = deepcopy(config)
                    del _config[key]

                    _converted_name = '_'.join("{}{}".format(k, v) for k, v in zip(_param_names, values))
                    _converted_name = re.sub("[' \[\],()]", '', _converted_name)
                    _config['path'] = os.path.join(config['path'], config['name'], _converted_name)
                    _config['name'] += '_' + _converted_name
                    if 'log_path' in config:
                        _config['log_path'] = os.path.join(config['log_path'], config['name'], _converted_name, 'log')
                    else:
                        _config['log_path'] = os.path.join(_config['path'], 'log')
                    for i, t in enumerate(tuple_dict.keys()):
                        insert_deep_dictionary(_config['params'], t, values[i])
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
    def __experiment_exists_identically(config):
        if ClusterWork.__experiment_exists(config):
            with open(os.path.join(config['path'], 'experiment.yml'), 'r') as f:
                dumped_config = yaml.load(f)
                return dumped_config == config

        return False

    @staticmethod
    def __experiment_has_finished(config):
        return os.path.exists(os.path.join(config['path'], 'results.csv'))

    @staticmethod
    def __experiment_has_finished_repetitions(config):
        for file in os.listdir(config['log_path']):
            if fnmatch.fnmatch(file, 'rep_*.csv'):
                return True
        return False

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

        if log_df is None:
            return False, 0, None

        # if repetition has completed
        return log_df.shape[0] == config['iterations'], log_df.shape[0], log_df

    @staticmethod
    def __experiment_progress(config) -> (float, [float]):
        rep_progress = [ClusterWork.__repetition_progress(config, rep) for rep in range(config['repetitions'])]
        exp_progress = sum(rep_progress) / (config['repetitions'])
        return exp_progress, rep_progress

    @staticmethod
    def __repetition_progress(config, rep) -> float:
        log_df = ClusterWork.__load_repetition_results(config, rep)
        if log_df is None:
            return .0
        completed_iterations = log_df.shape[0]
        return float(completed_iterations) / config['iterations']

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        """ needs to be implemented by subclass. """
        pass

    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        """ needs to be implemented by subclass. """
        pass

    def finalize(self):
        pass

    def save_state(self, config: dict, rep: int, n: int) -> None:
        """ optionally can be implemented by subclass. """
        pass

    def restore_state(self, config: dict, rep: int, n: int) -> bool:
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


class IncompleteConfigurationError(Exception):
    pass
