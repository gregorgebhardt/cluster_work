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
import gc
import socket
import time
import zlib
from copy import deepcopy
import fnmatch
from typing import Generator, Tuple, List

import pandas as pd
import yaml
import logging

_logging_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
_direct_output_formatter = logging.Formatter('[%(asctime)s] %(message)s')
_direct_output_handler = logging.StreamHandler(sys.stdout)
_direct_output_handler.setFormatter(_direct_output_formatter)
DIR_OUT = 200
_direct_output_handler.addFilter(lambda lr: lr.levelno == DIR_OUT)
_logging_std_handler = logging.StreamHandler(sys.stdout)
_logging_std_handler.setFormatter(_logging_formatter)
_logging_std_handler.setLevel(logging.DEBUG)
_logging_std_handler.addFilter(lambda lr: lr.levelno != DIR_OUT)
_logging_filtered_std_handler = logging.StreamHandler(sys.stdout)
_logging_filtered_std_handler.setFormatter(_logging_formatter)
_logging_filtered_std_handler.setLevel(logging.DEBUG)
_logging_filtered_std_handler.addFilter(lambda lr: lr.levelno < logging.WARNING)
_logging_err_handler = logging.StreamHandler(sys.stderr)
_logging_err_handler.setFormatter(_logging_formatter)
_logging_err_handler.setLevel(logging.WARNING)
_logging_err_handler.addFilter(lambda lr: lr.levelno != DIR_OUT)

# default logging configuration: log everything up to WARNING to stdout and from WARNING upwards to stderr
# set log-level to INFO
logging.basicConfig(level=logging.INFO, handlers=[_logging_std_handler, _logging_err_handler])

# get logger for cluster_work package
_logger = logging.getLogger('cluster_work')
_logger.addHandler(_logging_filtered_std_handler)
_logger.addHandler(_logging_err_handler)
_logger.addHandler(_direct_output_handler)
_logger.propagate = False

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


class ClusterWork(object):
    # change this in subclass, if you support restoring state on iteration level
    _restore_supported = False
    _default_params = {}
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _VERBOSE = False
    _NO_GUI = False
    _LOG_LEVEL = 'DEBUG'
    _RESTART_FULL_REPETITIONS = False
    _MP_CONTEXT = 'fork'

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
    _parser.add_argument('-l', '--log_level', nargs='?', default='INFO',
                         choices=['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR'],
                         help='sets the log-level for the output of ClusterWork')
    _parser.add_argument('--plot', nargs='?', const=True, default=False,
                         help='calls the plotting function of the experiment and exits')
    _parser.add_argument('--filter', default=argparse.SUPPRESS,
                         help='allows to filter the plotted experiments')

    __runs_on_cluster = False

    def __init__(self):
        self.__log_path_rep = None
        self.__log_path_rep_exists = False
        self.__log_path_it = None
        self.__log_path_it_exists = False

        self.__results = None
        self.__completed = False

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

    @staticmethod
    def get_experiments(path='.'):
        """ go through all subdirectories starting at path and return the experiment
            identifiers (= directory names) of all existing experiments. A directory
            is considered an experiment if it contains a experiment.yml file.
        """
        exps = []
        for dp, dn, fn in os.walk(path):
            if 'experiment.yml' in fn:
                subdirs = [os.path.join(dp, d) for d in os.listdir(dp) if os.path.isdir(os.path.join(dp, d))]
                if all(map(lambda s: ClusterWork.get_experiments(s) == [], subdirs)):
                    exps.append(dp)
        return exps

    @staticmethod
    def get_experiment_config(path, config_filename='experiment.yml'):
        """ reads the parameters of the experiment (= path) given.
        """
        with open(os.path.join(path, config_filename), 'r') as f:
            config = yaml.load(f)
            return config

    @staticmethod
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

    @classmethod
    def load_experiments(cls, config_file, experiment_selectors=None):
        """loads all experiment configurations from the given stream, merges them with the default configuration and
        expands list or grid parameters

        :config_file: file stream of the configuration yaml for the experiments that should be loaded.
        :experiment_selectors: list of experiment names. Only the experiments in this list will be loaded.
        :return: returns the experiment configurations
        """
        try:
            _config_documents = [*yaml.load_all(config_file)]
        except IOError:
            raise SystemExit('config file %s not found.' % config_file)

        if _config_documents[0]['name'].lower() == 'default':
            default_config = _config_documents[0]
            experiments_config = _config_documents[1:]
        else:
            default_config = dict()
            experiments_config = _config_documents

        # iterate over experiments and compute effective configuration and parameters
        # TODO add warning if yaml has parameters that do not appear in cls._default_parameters
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

        _experiments = cls.__adapt_experiment_path(effective_experiments)
        _experiments = cls.__expand_experiments(_experiments)
        _experiments = cls.__adapt_experiment_log_path(_experiments)

        return _experiments

    @staticmethod
    def __adapt_experiment_path(config_list):
        """ adapts the path of the experiment and sets the log-path
        """
        # for one single experiment, still wrap it in list
        if type(config_list) == dict:
            config_list = [config_list]

        expanded_config_list = []
        for config in config_list:
            config['path'] = os.path.join(config['path'], config['name'])
            expanded_config_list.append(config)

        return expanded_config_list

    @staticmethod
    def __adapt_experiment_log_path(config_list):
        """ adapts the log path of the experiment and sets the log-path
        """
        # for one single experiment, still wrap it in list
        if type(config_list) == dict:
            config_list = [config_list]

        expanded_config_list = []
        for config in config_list:
            if 'log_path' in config:
                config['log_path'] = os.path.join(config['log_path'], config['name'])
            else:
                config['log_path'] = os.path.join(config['path'], 'log')
            expanded_config_list.append(config)

        return expanded_config_list

    @staticmethod
    def __expand_experiments(config_list):
        """ expands the parameters list according to one of these schemes:
            grid: every list item is combined with every other list item
            list: every n-th list item of parameter lists are combined
        """
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
                    _config['path'] = os.path.join(config['path'], _converted_name)
                    _config['name'] += '_' + _converted_name
                    # if 'log_path' in config:
                    #     _config['log_path'] = os.path.join(config['log_path'], config['name'], _converted_name, 'log')
                    # else:
                    #     _config['log_path'] = os.path.join(_config['path'], 'log')
                    for i, t in enumerate(tuple_dict.keys()):
                        insert_deep_dictionary(_config['params'], t, values[i])
                    expanded_config_list.append(_config)
            else:
                expanded_config_list.append(config)

        return expanded_config_list

    @classmethod
    def __init_experiments(cls, config_file, experiments=None, delete_old=False, ignore_config_for_skip=False,
                           overwrite_old=False):
        """initializes the experiment by loading the configuration file and creating the directory structure.
        :return:
        """
        expanded_experiments = cls.load_experiments(config_file, experiments)

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
        try:
            import job_stream.common
            import job_stream.inline
        except ModuleNotFoundError:
            print('ClusterWork requires the job_stream package for distribution jobs via MPI.')
            raise

        @work.init
        def _work_init():
            """initializes the work, i.e., loads the configuration file, compiles the configuration documents from
            the file and the default configurations, expands grid and list experiments, and creates the directory
            structure.

            :return: a job_stream.inline.Multiple object with one configuration document for each experiment.
            """
            _logger.debug("[rank {}] [{}] parsing arguments and initializing experiments".format(
                job_stream.inline.getRank(), socket.gethostname()))

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
            _logger.info("[rank {}] [{}] creating work for <{}>".format(job_stream.inline.getRank(),
                                                                        socket.gethostname(),
                                                                        config['name']))
            _logger.debug("[rank {}] [{}] cpuCount: {}, hostCpuCount: {}".format(job_stream.common.getRank(),
                                                                                socket.gethostname(),
                                                                                job_stream.common.getCpuCount(),
                                                                                job_stream.common.getHostCpuCount()))

            if not hasattr(store, 'index'):
                store.index = pd.MultiIndex.from_product([range(config['repetitions']), range(config['iterations'])])
                store.index.set_names(['r', 'i'], inplace=True)
                store.config = config
                # create work list
                work_list = [job_stream.inline.Args(config, i) for i in range(config['repetitions'])]

                return job_stream.inline.Multiple(work_list)

        @work.job
        def _run_repetition(exp_config, r):
            """runs a single repetition of the experiment by calling run_rep(exp_config, r) on the instance of
            ClusterWork.

            :param exp_config: the configuration document for the experiment
            :param r: the repetition number
            """
            _logger.info('[rank {}] [{}] starting <{}> - Rep {}'.format(job_stream.getRank(),
                                                                        socket.gethostname(),
                                                                        exp_config['name'], r))

            repetition_results = cls().__init_rep(exp_config, r).__run_rep(exp_config, r)
            gc.collect()

            return repetition_results

        @work.frameEnd
        def _end_experiment(store, repetition_results):
            """collects the results from the individual repetitions in a pandas.DataFrame.

            :param store: the store object from the frame.
            :param repetition_results: the pandas.DataFrame with the results of the repetition.
            """
            _logger.info("[rank {}] [{}] storing results of <{}> - Rep {}".format(job_stream.inline.getRank(),
                                                                                  socket.gethostname(),
                                                                                  store.config['name'],
                                                                                  repetition_results.index[0][0]))

            if repetition_results is None:
                return

            if not hasattr(store, 'results'):
                store.results = pd.DataFrame(index=store.index, columns=repetition_results.columns)

            store.results.update(repetition_results)

        @work.result()
        def _work_results(store):
            """takes the resulting store object and writes the pandas.DataFrame to a file results.csv in the
            experiment folder.

            :param store: the store object emitted from the frame.
            """
            _logger.info("[rank {}] [{}] saving results of <{}> to disk".format(job_stream.inline.getRank(),
                                                                                socket.gethostname(),
                                                                                store.config['name']))

            if hasattr(store, 'results'):
                with open(os.path.join(store.config['path'], 'results.csv'), 'w') as results_file:
                    store.results.to_csv(results_file, **cls._pandas_to_csv_options)
            else:
                _logger.warning("[rank {}] [{}] no results available for <{}>".format(job_stream.inline.getRank(),
                                                                                      socket.gethostname(),
                                                                                      store.config['name']))

    @classmethod
    def init_from_config(cls, config, rep=0, it=0):
        instance = cls().__init_rep(config, rep)
        instance.restore_state(config, rep, it)
        instance._it = it

        def exception_stub(c, r, i):
            raise Exception('Experiment not properly initialized. Cannot run.')

        instance.iterate = exception_stub
        instance.reset = exception_stub

        return instance

    @classmethod
    def run(cls):
        """ starts the experiments as given in the config file. """
        options = cls._parser.parse_args()

        cls._NO_GUI = options.no_gui
        cls._VERBOSE = options.verbose
        cls._LOG_LEVEL = options.log_level.upper()
        cls._RESTART_FULL_REPETITIONS = options.restart_full_repetitions

        logging.root.setLevel(cls._LOG_LEVEL)
        _logging_filtered_std_handler.setLevel(level=cls._LOG_LEVEL)
        _logging_std_handler.setLevel(level=cls._LOG_LEVEL)
        if cls._VERBOSE:
            _logging_filtered_std_handler.setLevel(level=logging.DEBUG)
            _logging_std_handler.setLevel(level=cls._LOG_LEVEL)

        if options.progress:
            cls.__show_progress(options.config, options.experiments)
            return

        if options.full_progress:
            cls.__show_progress(options.config, options.experiments, full_progress=True)
            return

        if options.plot:
            if hasattr(options, 'filter'):
                cls.__plot_experiment_results(options.config, options.experiments, options.filter)
            else:
                cls.__plot_experiment_results(options.config, options.experiments)
            return

        if options.cluster:
            try:
                import job_stream.common
                import job_stream.inline
            except ModuleNotFoundError:
                print('ClusterWork requires the job_stream package for distribution jobs via MPI.')
                raise
            cls.__runs_on_cluster = True
            _logger.debug('Setting multiprocessing context to \'forkserver\'.')
            cls._MP_CONTEXT = 'forkserver'

            if job_stream.common.getRank() == 0:
                _logger.info("starting {} with the following options:".format(cls.__name__))
                for option, value in vars(options).items():
                    _logger.info("  - {}: {}".format(option, value))

            # without setting the useMultiprocessing flag to False, we get errors on the cluster
            with job_stream.inline.Work(useMultiprocessing=False) as w:
                cls.__setup_work_flow(w)
                _logger.removeHandler(_logging_filtered_std_handler)
                _logger.addHandler(_logging_std_handler)
                _logger.debug('[rank {}] Work has been setup...'.format(job_stream.getRank()))

                # w.run()
        else:
            _logger.info("starting {} with the following options:".format(cls.__name__))
            for option, value in vars(options).items():
                _logger.info("  - {}: {}".format(option, value))

            config_exps_w_expanded_params = cls.__init_experiments(config_file=options.config,
                                                                   experiments=options.experiments,
                                                                   delete_old=options.delete,
                                                                   ignore_config_for_skip=options.skip_ignore_config,
                                                                   overwrite_old=options.overwrite)
            for experiment in config_exps_w_expanded_params:
                # expand config_list_w_expanded_params for all repetitions and add self and rep number
                repetitions_list = []
                num_repetitions = experiment['repetitions']
                repetitions_list.extend(zip([experiment] * num_repetitions,
                                            range(num_repetitions)))
                _logger.info("starting experiment {}".format(experiment['name']))

                results = dict()
                for repetition in repetitions_list:
                    time_start = time.perf_counter()
                    _logger.log(DIR_OUT, '====================================================')
                    _logger.log(DIR_OUT, '>  Running Repetition {} '.format(repetition[1] + 1))
                    _logger.log(DIR_OUT, '----------------------------------------------------')
                    result = cls().__init_rep(*repetition).__run_rep(*repetition)
                    _elapsed_time = time.perf_counter() - time_start
                    _elapsed_hours = int(_elapsed_time) // 60 ** 2
                    _elapsed_minutes = (int(_elapsed_time) // 60) % 60
                    _elapsed_seconds = _elapsed_time % 60
                    _logger.log(DIR_OUT, '----------------------------------------------------')
                    _logger.log(DIR_OUT, '>  Finished Repetition {}'.format(repetition[1] + 1))
                    _logger.log(DIR_OUT, '>  Elapsed time: {:d}h:{:d}m:{:.2f}s'.format(_elapsed_hours,
                                                                                        _elapsed_minutes,
                                                                                        _elapsed_seconds))
                    _logger.log(DIR_OUT, '////////////////////////////////////////////////////')
                    results[repetition[1]] = result
                    gc.collect()

                _index = pd.MultiIndex.from_product([range(experiment['repetitions']),
                                                     range(experiment['iterations'])],
                                                    names=['r', 'i'])
                result_frame = None
                for i in results:
                    if results[i] is None:
                        continue
                    if result_frame is None:
                        result_frame = pd.DataFrame(index=_index, columns=results[i].columns, dtype=float)
                    result_frame.update(results[i])

                if result_frame is not None:
                    with open(os.path.join(experiment['path'], 'results.csv'), 'w') as results_file:
                        result_frame.to_csv(results_file, **cls._pandas_to_csv_options)

    def __init_rep(self, config, rep):
        """ run a single repetition including directory creation, log files, etc. """
        # set configuration of this repetition
        self._name = config['name']
        self._repetitions = config['repetitions']
        self._iterations = config['iterations']
        self._path = config['path']
        self._log_path = config['log_path']
        self._log_path_rep = os.path.join(config['log_path'], '{:02d}'.format(rep), '')
        self._plotting = config['plotting'] if 'plotting' in config else True
        self._no_gui = (not config['gui'] if 'gui' in config else False) or self.__runs_on_cluster or self._NO_GUI
        self._seed_base = zlib.adler32(self._name.encode()) % int(1e6)
        self._seed = self._seed_base + 1000 * rep

        # set params of this repetition
        self._params = config['params']
        self._rep = rep

        # check if log-file for repetition exists
        repetition_has_finished, n_finished_its, results = self.__repetition_has_completed(config, rep)

        # skip repetition if it has finished
        if repetition_has_finished or n_finished_its == config['iterations']:
            _logger.info('Repetition {} of experiment {} has finished before. '
                         'Skipping...'.format(rep + 1, config['name']))
            self.__results = results
            self.__completed = True
            return self

        self.reset(config, rep)

        # if not completed but some iterations have finished, check for restart capabilities
        if self._restore_supported and n_finished_its > 0 and not self._RESTART_FULL_REPETITIONS:
            _logger.info('Repetition {} of experiment {} has started before. '
                         'Trying to restore state after iteration {}.'.format(rep + 1, config['name'], n_finished_its))
            # set start for iterations and restore state in subclass
            self.start_iteration = n_finished_its
            try:
                self._log_path_it = os.path.join(config['log_path'], '{:02d}'.format(rep), '{:02d}'.format(n_finished_its - 1), '')
                if self.restore_state(config, rep, self.start_iteration - 1):
                    _logger.info('Restoring iteration succeeded. Restarting after iteration {}.'.format(n_finished_its))
                    self.__results = pd.DataFrame(data=results,
                                                  index=pd.MultiIndex.from_product([[rep], range(config['iterations'])],
                                                                                 names=['r', 'i']),
                                                  columns=results.columns, dtype=float)
                else:
                    _logger.info('Restoring iteration NOT successful. Restarting from iteration 1.')
                    self.start_iteration = 0
                    self.__results = None
            except:
                _logger.error('Exception during restore_state of experiment {} in repetition {}.'
                              'Restarting from iteration 1.'.format(config['name'], rep + 1), exc_info=True)
                self.start_iteration = 0
                self.__results = None
        else:
            self.start_iteration = 0
            self.__results = None

        # set logging handlers for current repetition
        file_handler_mode = 'a' if n_finished_its else 'w'
        file_handler = logging.FileHandler(os.path.join(self._log_path_rep, 'log.txt'), file_handler_mode)
        file_handler.setLevel(self._LOG_LEVEL)
        file_handler.setFormatter(_logging_formatter)
        file_handler.addFilter(lambda lr: lr.levelno != DIR_OUT)
        if self.__runs_on_cluster:
            logging.root.handlers.clear()
            logging.root.handlers = [file_handler, _logging_err_handler]
        else:
            logging.root.handlers.clear()
            logging.root.handlers = [file_handler, _logging_filtered_std_handler, _logging_err_handler,
                                     _direct_output_handler]

        return self

    def __run_rep(self, config, rep) -> pd.DataFrame:
        log_filename = os.path.join(self._log_path, 'rep_{}.csv'.format(rep))
        if self.__completed:
            return self.__results

        for it in range(self.start_iteration, config['iterations']):
            self._it = it
            self._seed = self._seed_base + 1000 * rep + it

            # update iteration log directory
            self._log_path_it = os.path.join(config['log_path'], '{:02d}'.format(rep), '{:02d}'.format(it), '')

            # run iteration and get results
            try:
                time_start = time.perf_counter()
                _logger.log(DIR_OUT, '----------------------------------------------------')
                _logger.log(DIR_OUT, '>  Starting Iteration {} of Repetition {}'.format(it + 1, rep + 1))
                _logger.log(DIR_OUT, '----------------------------------------------------')
                it_result = self.iterate(config, rep, it)

                flat_it_result = flatten_dict(it_result)

                if self.__results is None:
                    self.__results = pd.DataFrame(index=pd.MultiIndex.from_product([[rep], range(config['iterations'])],
                                                                                   names=['r', 'i']),
                                                  columns=flat_it_result.keys(), dtype=float)

                self.__results.loc[(rep, it)] = flat_it_result

                # write first line with header
                if it == 0:
                    self.__results.iloc[[it]].to_csv(log_filename, mode='w', header=True, **self._pandas_to_csv_options)
                else:
                    self.__results.iloc[[it]].to_csv(log_filename, mode='a', header=False,
                                                     **self._pandas_to_csv_options)

                self.save_state(config, rep, it)
            except ValueError or OverflowError or ZeroDivisionError or ArithmeticError or FloatingPointError:
                _logger.error('Experiment {} - Repetition {} - Iteration {}'.format(config['name'], rep + 1, it + 1),
                              exc_info=True)
                self.finalize()
                return self.__results
            except:
                self.finalize()
                raise
            finally:
                _elapsed_time = time.perf_counter() - time_start
                _logger.log(DIR_OUT, '----------------------------------------------------')
                _logger.log(DIR_OUT, '>  Finished Iteration {} of Repetition {}'.format(it + 1, rep + 1))
                _elapsed_minutes, _elapsed_seconds = int(_elapsed_time) // 60, _elapsed_time % 60
                _logger.log(DIR_OUT, '>  Elapsed time: {:d}m:{:.2f}s'.format(_elapsed_minutes, _elapsed_seconds))
                _logger.log(DIR_OUT, '----------------------------------------------------')

        self.finalize()
        self.__completed = True
        return self.__results

    @classmethod
    def get_progress(cls, config_file, experiment_selectors=None) -> Tuple[float, List[Tuple[str, float, List[float]]]]:
        experiments_config = cls.load_experiments(config_file, experiment_selectors)
        total_progress = .0
        experiment_progress = []

        for config in experiments_config:
            exp_progress, rep_progress = cls.__experiment_progress(config)
            total_progress += exp_progress / len(experiments_config)
            experiment_progress.append((config['name'], exp_progress, rep_progress))

        return total_progress, experiment_progress

    @classmethod
    def __show_progress(cls, config_file, experiment_selectors=None, full_progress=False):
        """ shows the progress of all experiments defined in the config_file.
        """
        total_progress, experiments_progress = cls.get_progress(config_file, experiment_selectors)

        for name, exp_progress, rep_progress in experiments_progress:
            # progress bar
            bar = "["
            bar += "#" * round(25 * exp_progress)
            bar += " " * (25 - round(25 * exp_progress))
            bar += "]"
            print('{:5.1f}% {:27} {}'.format(exp_progress * 100, bar, name))
            # print('%3.1f%% %27s %s' % (exp_progress * 100, bar, config['name']))

            if full_progress:
                for i, p in enumerate(rep_progress):
                    bar = "["
                    bar += "#" * round(25 * p)
                    bar += " " * (25 - round(25 * p))
                    bar += "]"
                    print('    |- {:2d} {:5.1f}% {:27}'.format(i+1, p * 100, bar))
                print()
        print()

        # print total progress
        bar = "["
        bar += "%" * round(50 * total_progress)
        bar += " " * (50 - round(50 * total_progress))
        bar += "]"
        print('  Total: {:5.1f}% {:52}\n'.format(total_progress * 100, bar))

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
    def load_repetition_results(config, rep):
        rep_results_filename = os.path.join(config['log_path'], 'rep_{}.csv'.format(rep))

        if os.path.exists(rep_results_filename):
            rep_results_df = pd.read_csv(rep_results_filename, sep='\t')
            rep_results_df.set_index(keys=['r', 'i'], inplace=True)

            return rep_results_df
        else:
            return None

    @staticmethod
    def load_experiment_results(config):
        results_filename = os.path.join(config['path'], 'results.csv')

        if os.path.exists(results_filename):
            results_df = pd.read_csv(results_filename, sep='\t')
            results_df.set_index(keys=['r', 'i'], inplace=True)
            return results_df
        else:
            return None

    @classmethod
    def __plot_experiment_results(cls, config_file, experiment_selectors=None, experiment_filter=''):
        experiment_configs = cls.load_experiments(config_file, experiment_selectors)

        def create_config_and_results_generator():
            for config in experiment_configs:
                if experiment_filter not in config['name']:
                    continue
                results = ClusterWork.load_experiment_results(config)
                yield config, results

        cls.plot_results(create_config_and_results_generator())

    @classmethod
    def iterate_config_and_results(cls, config_filename, experiment_selectors=None):
        with open(config_filename, 'r') as config_file:
            experiment_configs = cls.load_experiments(config_file, experiment_selectors)

        def config_and_results_generator():
            for config in experiment_configs:
                results = ClusterWork.load_experiment_results(config)
                yield config, results

        return config_and_results_generator

    @staticmethod
    def __repetition_has_completed(config, rep) -> (bool, int, pd.DataFrame):
        log_df = ClusterWork.load_repetition_results(config, rep)

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
        log_df = ClusterWork.load_repetition_results(config, rep)
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

    @classmethod
    def plot_results(cls, configs_results: Generator):
        raise NotImplementedError('plot_results needs to be implemented by subclass.')


class IncompleteConfigurationError(Exception):
    pass


class InvalidParameterArgument(Exception):
    pass


class StreamLogger:
    class LoggerWriter:
        def __init__(self, logger: logging.Logger, level: logging.DEBUG):
            # self.level is really like using log.debug(message)
            # at least in my case
            self.logger = logger
            self.level = level

        def write(self, message):
            # if statement reduces the amount of newlines that are
            # printed to the logger
            if message.strip() is not '':
                self.logger.log(self.level, message)

        def flush(self):
            # create a flush method so things can be flushed when
            # the system wants to. Not sure if simply 'printing'
            # sys.stderr is the correct way to do it, but it seemed
            # to work properly for me.
            # self.level(sys.stderr)
            pass

    def __init__(self, logger, stdout_level=logging.INFO, stderr_level=logging.WARNING):
        self.logger = logger
        self.stdout_level = stdout_level
        self.stderr_level = stderr_level

        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.LoggerWriter(self.logger, self.stdout_level)
        sys.stderr = self.LoggerWriter(self.logger, self.stderr_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
