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
import itertools
import collections
import os
import signal
import sys
import re
import gc
import socket
import time
import zlib
from copy import deepcopy
import fnmatch
from typing import Generator, Tuple, List

import numpy as np
import pandas as pd
import yaml
import logging


class _CWFormatter(logging.Formatter):
    def __init__(self):
        self.std_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)


_logging_formatter = _CWFormatter()

# _info_output_formatter = logging.Formatter('[%(asctime)s] %(message)s')
_info_content_output_handler = logging.StreamHandler(sys.stdout)
_info_content_output_handler.setFormatter(_logging_formatter)
_info_border_output_handler = logging.StreamHandler(sys.stdout)
_info_border_output_handler.setFormatter(_logging_formatter)
INFO_CONTNT = 200
INFO_BORDER = 150
_info_content_output_handler.setLevel(INFO_CONTNT)
_info_border_output_handler.setLevel(INFO_BORDER)

# _logging_std_handler = logging.StreamHandler(sys.stdout)
# _logging_std_handler.setFormatter(_logging_formatter)
# _logging_std_handler.setLevel(logging.DEBUG)
# _logging_std_handler.addFilter(lambda lr: lr.levelno <= logging.ERROR)

_logging_filtered_std_handler = logging.StreamHandler(sys.stdout)
_logging_filtered_std_handler.setFormatter(_logging_formatter)
_logging_filtered_std_handler.setLevel(logging.DEBUG)
_logging_filtered_std_handler.addFilter(lambda lr: lr.levelno < logging.WARNING)

_logging_err_handler = logging.StreamHandler(sys.stderr)
_logging_err_handler.setFormatter(_logging_formatter)
_logging_err_handler.setLevel(logging.WARNING)
_logging_err_handler.addFilter(lambda lr: lr.levelno <= logging.ERROR)

# default logging configuration: log everything up to WARNING to stdout and from WARNING upwards to stderr
# set log-level to INFO
logging.basicConfig(level=logging.INFO, handlers=[_logging_filtered_std_handler,
                                                  _logging_err_handler])

# get logger for cluster_work package
_logger = logging.getLogger('cluster_work')
_logger.addHandler(_logging_filtered_std_handler)
_logger.addHandler(_logging_err_handler)
# _logger.addHandler(_info_content_output_handler)
_logger.addHandler(_info_border_output_handler)
_logger.propagate = False


def _sigint_handler(sgn, ft):
    _logger.warning('Received signal {}, will exit program.'.format(sgn))
    if MPI:
        MPI.COMM_WORLD.Abort(0)
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)

# global MPI handle, will be set to mpi4py.MPI if mpi support is used
MPI = None


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


def format_time(time_in_secs: float):
    _hours = int(time_in_secs) // 60 ** 2
    _minutes = (int(time_in_secs) // 60) % 60
    _seconds = time_in_secs % 60

    time_str = ""
    if _hours:
        time_str += "{:d}h:".format(_hours)
    if _minutes or _hours:
        time_str += "{:d}m:".format(_minutes)
    time_str += "{:05.2f}s".format(_seconds)

    return time_str


def shorten_param(_param_name):
    name_parts = _param_name.split('.')
    shortened_parts = '.'.join(map(lambda s: s[:3], name_parts[:-1]))
    shortened_leaf = ''.join(map(lambda s: s[0], name_parts[-1].split('_')))
    if shortened_parts:
        return shortened_parts + '.' + shortened_leaf
    else:
        return shortened_leaf


class ClusterWork(object):
    # change this in subclass, if you support restoring state on iteration level
    _restore_supported = False
    _default_params = {}
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _NO_GUI = False
    _LOG_LEVEL = 'DEBUG'
    _RESTART_FULL_REPETITIONS = False
    _MP_CONTEXT = 'fork'

    _parser = argparse.ArgumentParser()
    _parser.add_argument('config', metavar='CONFIG.yml', type=argparse.FileType('r'))
    _parser.add_argument('-m', '--mpi', action='store_true',
                         help='Runs the experiments with mpi support.')
    _parser.add_argument('-g', '--mpi_groups', nargs='?', type=int,
                         help='The number of MPI groups to create.')
    _parser.add_argument('-d', '--delete', action='store_true',
                         help='CAUTION deletes results of previous runs.')
    _parser.add_argument('-o', '--overwrite', action='store_true',
                         help='CAUTION overwrites results of previous runs if config has changed.')
    _parser.add_argument('-e', '--experiments', nargs='+',
                         help='Allows to specify which experiments should be run.')
    _parser.add_argument('-p', '--progress', action='store_true',
                         help='Outputs the progress of the experiment and exits.')
    _parser.add_argument('-P', '--full_progress', action='store_true',
                         help='Outputs a more detailed progress of the experiment and exits.')
    _parser.add_argument('--no_gui', action='store_true',
                         help='Tells the experiment to not use any feature that requires a GUI.')
    _parser.add_argument('-I', '--ignore_config', action='store_true',
                         help='Ignores changes in the configuration file for skipping or overwriting experiments..')
    _parser.add_argument('--restart_full_repetitions', action='store_true')
    _parser.add_argument('-l', '--log_level', nargs='?', default='INFO',
                         choices=['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR'],
                         help='Sets the log-level for the output of ClusterWork.')
    _parser.add_argument('-r', '--repetition', type=int,
                         help='Start only given repetition, assumes that only one experiment will be started.')
    _parser.add_argument('-i', '--iteration', type=int,
                         help='Restart repetition from iteration i, works only together with -r/--repetition.')
    _parser.add_argument('--plot', nargs='?', const=True, default=False,
                         help='Calls the plotting function of the experiment and exits.')
    _parser.add_argument('--filter', default=argparse.SUPPRESS,
                         help='Allows to filter the plotted experiments.')
    _parser.add_argument('-c', '--cluster', action='store_true',
                         help='DEPRECATED use the -m/--mpi argument.',)
    _parser.add_argument('-v', '--verbose', action='store_true',
                         help='DEPRECATED, use log-level instead.')

    __run_with_mpi = False

    def __init__(self):
        self.__log_path_rep = None
        self.__log_path_rep_exists = False
        self.__log_path_it = None
        self.__log_path_it_exists = False

        self.__results = None
        self.__completed = False
        self._COMM = None

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

        # TODO use namedtuple or own ParameterStore??

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
        """ adapts the path of the experiment
        """
        # for one single experiment, still wrap it in list
        if type(config_list) == dict:
            config_list = [config_list]

        expanded_config_list = []
        for config in config_list:
            config['_config_path'] = config['path']
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

                    _converted_name = '_'.join("{}{}".format(shorten_param(k), v) for k, v in zip(_param_names, values))
                    # _converted_name = re.sub("[' \[\],()]", '', _converted_name)
                    _converted_name = re.sub("[' ]", '', _converted_name)
                    _converted_name = re.sub('["]', '', _converted_name)
                    _converted_name = re.sub("[(\[]", '_', _converted_name)
                    _converted_name = re.sub("[)\]]", '', _converted_name)
                    _converted_name = re.sub("[,]", '_', _converted_name)
                    _config['_experiment_path'] = config['path']
                    _config['path'] = os.path.join(config['path'], _converted_name)
                    _config['experiment_name'] = _config['name']
                    _config['name'] += '__' + _converted_name
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
    def __init_experiments(cls, config_file, experiments=None, delete_old=False, ignore_config=False,
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
                    if ignore_config:
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
                            # add experiment to clear list
                            _logger.warning('Experiment {} has finished before, but configuration has '
                                            'changed! Overwriting...'.format(_config['name']))
                            clear_experiments.append(_config)
                        else:
                            # add experiment to skip list
                            skip_experiments.append(_config)
                            _logger.warning('Experiment {} has finished before, but configuration has '
                                            'changed! Skipping...'.format(_config['name']))
                            _logger.warning('--> To overwrite existing results, use the option -o/--overwrite')
                elif cls.__experiment_exists(_config) and not cls.__experiment_exists_identically(_config):
                    if ignore_config:
                        _logger.warning('Experiment {} has started before, but configuration has '
                                        'changed! '.format(_config['name'])) + \
                                        'Starting Experiment anyways due to option -I/--ignore-config'
                    elif overwrite_old:
                        # add experiment to clear list
                        _logger.warning('Experiment {} has started before, but configuration has '
                                        'changed! Overwriting...'.format(_config['name']))
                        clear_experiments.append(_config)
                    else:
                        # add experiment to skip list
                        skip_experiments.append(_config)
                        _logger.warning('Experiment {} has started before, but configuration has '
                                        'changed! Skipping...'.format(_config['name']))
                        _logger.warning('--> To overwrite existing results, use the option -o/--overwrite')
                    if cls.__experiment_has_finished_repetitions(_config):
                        pass
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
    def __run_mpi(cls, mpi_groups=0):
        from mpi4py.futures import MPICommExecutor

        cls.__num_mpi_groups = mpi_groups
        hostname = socket.gethostname()

        if mpi_groups:
            assert mpi_groups <= MPI.COMM_WORLD.size, 'mpi_groups must be <= {}'.format(MPI.COMM_WORLD.size)
            if mpi_groups == 1:
                MPI.COMM_GROUP = MPI.COMM_WORLD
                MPI.COMM_JOB = MPI.COMM_SELF if MPI.COMM_WORLD.rank == 0 else MPI.COMM_NULL
            else:
                if 0 < MPI.COMM_WORLD.rank < (MPI.COMM_WORLD.size - ((MPI.COMM_WORLD.size - 1) % mpi_groups)):
                    MPI.COMM_GROUP = MPI.COMM_WORLD.Split((MPI.COMM_WORLD.rank - 1) % mpi_groups,
                                                      (MPI.COMM_WORLD.rank - 1) // mpi_groups)
                    MPI.COMM_GROUP.name = "comm_group_{}".format((MPI.COMM_WORLD.rank - 1) % mpi_groups)
                else:
                    MPI.COMM_GROUP = MPI.COMM_WORLD.Split(MPI.UNDEFINED, MPI.COMM_WORLD.rank)
                if MPI.COMM_WORLD.rank == 0 or (MPI.COMM_GROUP and MPI.COMM_GROUP.rank == 0):
                    MPI.COMM_JOB = MPI.COMM_WORLD.Split(0, MPI.COMM_WORLD.rank)
                else:
                    MPI.COMM_JOB = MPI.COMM_WORLD.Split(MPI.UNDEFINED, MPI.COMM_WORLD.rank)
        else:
            MPI.COMM_JOB = MPI.COMM_WORLD
            MPI.COMM_GROUP = MPI.COMM_NULL

        _logger.debug('[rank {}] [{}] {}{}'.format(MPI.COMM_WORLD.rank, hostname,
                                                   'in {} '.format(MPI.COMM_GROUP.name) if MPI.COMM_GROUP else '',
                                                   'in comm_job' if MPI.COMM_JOB else ''))

        def _run_repetition(_config, _r):
            """Runs a single repetition of the experiment by calling run_rep(exp_config, r) on a new instance.

            :param _config: the configuration document for the experiment
            :param _r: the repetition number
            """
            import socket
            hostname = socket.gethostname()
            # if 'MPI' in globals():
            #     MPI = globals()['MPI']
            # else:
            #     _logger.warning('no MPI in global variables')
            #     from mpi4py import MPI
            #     MPI.COMM_GROUP = MPI.COMM_SELF

            _logger.debug('[rank {}] [{}] starting <{}> - Rep {}'.format(MPI.COMM_WORLD.rank, hostname,
                                                                         _config['name'], _r + 1))
            if mpi_groups:
                _control = 'run_rep'
                MPI.COMM_GROUP.bcast(_control, root=0)

                _instance = cls()
                _instance._COMM = MPI.COMM_GROUP
                MPI.COMM_GROUP.bcast(_config, root=0)
                MPI.COMM_GROUP.bcast(_r, root=0)
            else:
                _instance = cls()
                _instance._COMM = MPI.COMM_SELF

            repetition_results = _instance.__init_rep(_config, _r).__run_rep(_config, _r)
            gc.collect()

            _logger.debug('[rank {}] [{}] finished <{}> - Rep {}'.format(MPI.COMM_WORLD.rank, hostname,
                                                                         _config['name'], _r + 1))

            # return repetition_results
            return _config, _r, repetition_results

        exp_results = dict()
        exp_configs = dict()

        if MPI.COMM_JOB != MPI.COMM_NULL:
            with MPICommExecutor(MPI.COMM_JOB, root=0) as executor:
                if executor is not None:
                    _logger.debug("[rank {}] [{}] parsing arguments and initializing experiments".format(
                        MPI.COMM_WORLD.rank, hostname))

                    options = cls._parser.parse_args()
                    exp_list = cls.__init_experiments(config_file=options.config, experiments=options.experiments,
                                                      delete_old=options.delete,
                                                      ignore_config=options.ignore_config,
                                                      overwrite_old=options.overwrite)

                    _logger.debug("[rank {}] [{}] emitting the following initial work:".format(MPI.COMM_WORLD.rank,
                                                                                                   hostname))

                    work_list = []
                    if isinstance(options.repetition, int):
                        exp_config = exp_list[0]
                        if not 0 <= options.repetition < exp_config['repetitions']:
                            # _logger.error('Repetition has to be in range [0, {}]'.format(num_repetitions))
                            raise InvalidParameterArgument(
                                'Repetition has to be in range [0, {}]'.format(exp_config['repetitions']))
                        _logger.debug("     - <{}>".format(exp_config['name']))
                        _logger.debug("       repetition {}".format(options.repetition + 1))
                        work_list = [(exp_config, options.repetition)]
                    else:
                        for exp_config in exp_list:
                            _logger.debug("     - <{}>".format(exp_config['name']))
                            _logger.debug("       creating {} repetitions...".format(exp_config['repetitions']))
                            exp_work = [(exp_config, i) for i in range(exp_config['repetitions'])]
                            work_list.extend(exp_work)

                    executor_results = executor.starmap(_run_repetition, work_list)

                    for _config, _r, _rep_results in executor_results:
                        _logger.debug("[rank {}] [{}] adding results from <{}> - Rep {}".format(
                            MPI.COMM_WORLD.rank, hostname, _config['name'], _r + 1))

                        if _config['name'] not in exp_results:
                            index = pd.MultiIndex.from_product([range(_config['repetitions']),
                                                                range(_config['iterations'])])
                            index.set_names(['r', 'i'], inplace=True)
                            exp_results[_config['name']] = pd.DataFrame(index=index, columns=_rep_results.columns)
                            exp_configs[_config['name']] = _config
                        exp_results[_config['name']].update(_rep_results)

                    _logger.info("[rank {}] [{}] saving results to disk".format(MPI.COMM_WORLD.rank, hostname))

                    for exp_name, results in exp_results.items():
                        exp_config = exp_configs[exp_name]
                        with open(os.path.join(exp_config['path'], 'results.csv'), 'w') as results_file:
                            results.to_csv(results_file, **cls._pandas_to_csv_options)

            if MPI.COMM_GROUP != MPI.COMM_NULL:
                _logger.debug('[rank {}] [{}] sending break...'.format(MPI.COMM_WORLD.rank, hostname))
                _logger.warning('Will send Abort to {}'.format(MPI.COMM_GROUP.name))
                MPI.COMM_GROUP.Abort(0)

            if MPI.COMM_WORLD.rank == 0:
                # the last one turns off the light
                MPI.COMM_WORLD.Abort(0)
                sys.exit()

        elif MPI.COMM_GROUP != MPI.COMM_NULL:
            while True:
                _logger.debug('[rank {}] [{}] waiting for control...'.format(MPI.COMM_WORLD.rank, hostname))
                control = MPI.COMM_GROUP.bcast(None, root=0)
                _logger.debug('[rank {}] [{}] received {}.'.format(MPI.COMM_WORLD.rank, hostname, control))

                if control == 'run_rep':
                    instance = cls()
                    instance._COMM = MPI.COMM_GROUP
                    _config = MPI.COMM_GROUP.bcast(None, root=0)
                    _r = MPI.COMM_GROUP.bcast(None, root=0)

                    instance.__init_rep(_config, _r).__run_rep(_config, _r)
                    gc.collect()
                else:
                    break

    @classmethod
    def init_from_config(cls, config, rep=0, it=0):
        instance = cls().__init_rep_without_checks(config, rep)

        instance._log_path_it = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), 'it_{:04d}'.format(it), '')

        try:
            instance.restore_state(config, rep, it)
        except IOError:
            _logger.warning('Could not restore experiment {}, rep {} at iteration {}.'.format(config['name'], rep, it))
            return None

        instance._it = it

        def exception_stub(_c, _r, _i):
            raise Exception('Experiment not properly initialized. Cannot run.')

        instance.iterate = exception_stub
        instance.reset = exception_stub

        return instance

    @classmethod
    def run(cls):
        """ starts the experiments as given in the config file. """
        options = cls._parser.parse_args()

        cls._NO_GUI = options.no_gui
        cls._LOG_LEVEL = options.log_level.upper()
        cls._RESTART_FULL_REPETITIONS = options.restart_full_repetitions

        logging.root.setLevel(cls._LOG_LEVEL)
        _logging_filtered_std_handler.setLevel(level=cls._LOG_LEVEL)
        if options.verbose:
            _logger.warning('DEPRECATED option -v/--verbose is deprecated, use -l/-log_level instead.')

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

        if options.mpi or options.cluster:
            if options.cluster:
                _logger.warning('DEPRECATED option -c/--cluster is deprecated, use -m/-mpi instead.')

            try:
                import mpi4py
                import cloudpickle
                global MPI
                MPI = mpi4py.MPI
                MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)
            except ModuleNotFoundError:
                _logger.error('ClusterWork requires the mpi4py and cloudpickle packages for distributing jobs via MPI.')
                raise
            cls.__run_with_mpi = True
            cls._MP_CONTEXT = 'forkserver'
            import multiprocessing as mp
            if not mp.get_start_method(allow_none=True):
                mp.set_start_method(cls._MP_CONTEXT)

            if MPI.COMM_WORLD.rank == 0:
                _logger.info("starting {} with the following options:".format(cls.__name__))
                for option, value in vars(options).items():
                    _logger.info("  - {}: {}".format(option, value))

            cls.__run_mpi(options.mpi_groups)
        else:
            _logger.info("starting {} with the following options:".format(cls.__name__))
            for option, value in vars(options).items():
                _logger.info("  - {}: {}".format(option, value))

            config_exps_w_expanded_params = cls.__init_experiments(config_file=options.config,
                                                                   experiments=options.experiments,
                                                                   delete_old=options.delete,
                                                                   ignore_config=options.ignore_config,
                                                                   overwrite_old=options.overwrite)
            for experiment in config_exps_w_expanded_params:
                num_repetitions = experiment['repetitions']

                if isinstance(options.repetition, int):
                    if not 0 <= options.repetition < num_repetitions:
                        # _logger.error('Repetition has to be in range [0, {}]'.format(num_repetitions))
                        raise InvalidParameterArgument('Repetition has to be in range [0, {}]'.format(num_repetitions))
                    repetitions_list = [(experiment, options.repetition)]

                else:
                    # expand config_list_w_expanded_params for all repetitions and add self and rep number
                    repetitions_list = list(zip([experiment] * num_repetitions, range(num_repetitions)))

                _logger.info("starting experiment {}".format(experiment['name']))

                results = dict()
                for repetition in repetitions_list:
                    time_start = time.perf_counter()
                    _logger.log(INFO_BORDER, '====================================================')
                    _logger.log(INFO_CONTNT, '>  Running Repetition {} '.format(repetition[1] + 1))
                    result = cls().__init_rep(*repetition).__run_rep(*repetition)
                    _elapsed_time = time.perf_counter() - time_start
                    _logger.log(INFO_BORDER, '////////////////////////////////////////////////////')
                    _logger.log(INFO_CONTNT, '>  Finished Repetition {}'.format(repetition[1] + 1))
                    _logger.log(INFO_CONTNT, '>  Elapsed time: {}'.format(format_time(_elapsed_time)))
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

        sys.exit(0)

    def __init_rep(self, config, rep):
        """ run a single repetition including directory creation, log files, etc. """
        # set configuration of this repetition
        self._name = config['name']
        self._repetitions = config['repetitions']
        self._iterations = config['iterations']
        self._path = config['path']
        self._log_path = config['log_path']
        self._log_path_rep = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), '')
        self._plotting = config['plotting'] if 'plotting' in config else True
        self._no_gui = (not config['gui'] if 'gui' in config else False) or self.__run_with_mpi or self._NO_GUI
        self._seed_base = zlib.adler32(self._name.encode()) % int(1e6)
        self._seed = self._seed_base + 1000 * rep + 5
        if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
            self._seed += int(1e5 * self._COMM.rank)

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

        # set logging handlers for current repetition
        file_handler_mode = 'a' if n_finished_its else 'w'
        file_handler = logging.FileHandler(os.path.join(self._log_path_rep, 'log.txt'), file_handler_mode)
        file_handler.setLevel(self._LOG_LEVEL)
        file_handler.setFormatter(_logging_formatter)
        # file_handler.addFilter(lambda lr: lr.levelno <= logging.ERROR)
        if self.__run_with_mpi and MPI.COMM_WORLD.size > 1:
            logging.root.handlers.clear()
            _logger.handlers.clear()
            if MPI.COMM_WORLD.rank == 0 and self.__num_mpi_groups == 1:
                # if we run just one group, rank 0 can output to stdout
                logging.root.handlers = [file_handler,
                                         _logging_filtered_std_handler,
                                         _logging_err_handler]
                _logger.handlers = [file_handler,
                                    _logging_filtered_std_handler,
                                    _logging_err_handler,
                                    _info_border_output_handler]
            elif self._COMM.rank == 0:
                logging.root.handlers = [file_handler, _logging_err_handler]
                _logger.handlers = [file_handler,
                                    _logging_filtered_std_handler,
                                    _logging_err_handler,
                                    _info_content_output_handler]
            else:
                logging.root.addHandler(_logging_err_handler)
                _logger.addHandler(_logging_err_handler)

        else:
            logging.root.handlers.clear()
            logging.root.handlers = [file_handler, _logging_filtered_std_handler, _logging_err_handler]
            _logger.addHandler(file_handler)

        self.reset(config, rep)

        # if not completed but some iterations have finished, check for restart capabilities
        if self._restore_supported and n_finished_its > 0 and not self._RESTART_FULL_REPETITIONS:
            _logger.info('Repetition {} of experiment {} has started before. '
                         'Trying to restore state after iteration {}.'.format(rep + 1, config['name'], n_finished_its))
            # set start for iterations and restore state in subclass
            self.start_iteration = n_finished_its
            for self.start_iteration in n_finished_its, n_finished_its - 1:
                try:
                    self._log_path_it = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep),
                                                     'it_{:04d}'.format(n_finished_its - 1), '')
                    if self.start_iteration and self.restore_state(config, rep, self.start_iteration - 1):
                        _logger.info('Restoring iteration succeeded. Restarting after iteration {}.'.format(
                            self.start_iteration))
                        self.__results = pd.DataFrame(data=results,
                                                      index=pd.MultiIndex.from_product(
                                                          [[rep], range(config['iterations'])],
                                                          names=['r', 'i']),
                                                      columns=results.columns, dtype=float)
                        break
                except IOError:
                    _logger.error('Exception during restore_state of experiment {} in repetition {}.'.format(
                        config['name'], rep + 1), exc_info=True)
            else:
                _logger.warning('Restoring iteration NOT successful. Restarting from iteration 1.')
                self.start_iteration = 0
                self.__results = None
        else:
            self.start_iteration = 0
            self.__results = None

        return self

    def __run_rep(self, config, rep) -> pd.DataFrame:
        log_filename = os.path.join(self._log_path, 'rep_{}.csv'.format(rep))
        if self.__completed:
            return self.__results

        repetition_time = .0
        if self.start_iteration > 0:
            repetition_time = self.__results.repetition_time.loc[(rep, self.start_iteration - 1)]

        for it in range(self.start_iteration, config['iterations']):
            self._it = it
            self._seed = self._seed_base + 1000 * rep + it
            if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
                self._seed += int(1e5 * self._COMM.rank)

            # update iteration log directory
            self._log_path_it = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), 'it_{:04d}'.format(it), '')

            # run iteration and get results
            iteration_time = None
            time_start = time.perf_counter()
            try:
                _logger.log(INFO_BORDER, '----------------------------------------------------')
                _logger.log(INFO_CONTNT, '>  Starting Iteration {}/{} of Repetition {}/{}'.format(
                    it + 1, self._iterations, rep + 1, self._repetitions))
                _logger.log(INFO_BORDER, '----------------------------------------------------')
                it_result = self.iterate(config, rep, it)
                iteration_time = time.perf_counter() - time_start
                repetition_time += iteration_time

                if it_result is None:
                    continue

                flat_it_result = flatten_dict(it_result)
                if 'iteration_time' not in flat_it_result:
                    flat_it_result['iteration_time'] = iteration_time
                if 'repetition_time' not in flat_it_result:
                    flat_it_result['repetition_time'] = repetition_time

                if self.__results is None:
                    self.__results = pd.DataFrame(index=pd.MultiIndex.from_product([[rep], range(config['iterations'])],
                                                                                   names=['r', 'i']),
                                                  columns=flat_it_result.keys(), dtype=float)

                self.__results.loc[(rep, it)] = flat_it_result

                # save state before results, so that we know the saved state can be restored if we find the results.
                self.save_state(config, rep, it)

                # write first line with header
                if it == 0:
                    self.__results.iloc[[it]].to_csv(log_filename, mode='w', header=True, **self._pandas_to_csv_options)
                else:
                    self.__results.iloc[[it]].to_csv(log_filename, mode='a', header=False,
                                                     **self._pandas_to_csv_options)
            except ValueError or ArithmeticError or np.linalg.linalg.LinAlgError:
                _logger.error('Experiment {} - Repetition {} - Iteration {}'.format(config['name'], rep + 1, it + 1),
                              exc_info=True)
                self.finalize()
                return self.__results
            except Exception:
                self.finalize()
                raise
            finally:
                if iteration_time is None:
                    iteration_time = time.perf_counter() - time_start
                _logger.log(INFO_BORDER, '----------------------------------------------------')
                _logger.log(INFO_CONTNT, '>  Finished Iteration {}/{} of Repetition {}/{}'.format(
                    it + 1, self._iterations, rep + 1, self._repetitions))
                _logger.log(INFO_CONTNT, '>  Iteration time: {}'.format(format_time(iteration_time)))
                _logger.log(INFO_CONTNT, '>  Repetition time: {}'.format(format_time(repetition_time)))

        self.finalize()
        self.__completed = True
        return self.__results

    def __init_rep_without_checks(self, config, rep):
        # set configuration of this repetition
        self._name = config['name']
        self._repetitions = config['repetitions']
        self._iterations = config['iterations']
        self._path = config['path']
        self._log_path = config['log_path']
        self._log_path_rep = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), '')
        self._plotting = config['plotting'] if 'plotting' in config else True
        self._no_gui = (not config['gui'] if 'gui' in config else False) or self.__run_with_mpi or self._NO_GUI
        self._seed_base = zlib.adler32(self._name.encode()) % int(1e6)
        self._seed = self._seed_base + 1000 * rep
        if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
            self._seed += int(1e5 * self._COMM.rank)

        # set params of this repetition
        self._params = config['params']
        self._rep = rep

        self.reset(config, rep)

        return self

    ExperimentProgress = collections.namedtuple('ExperimentProgress', ['exp_name', 'name', 'num_iterations',
                                                                       'num_repetitions',
                                                                       'exp_progress', 'rep_progress',
                                                                       'finished_repetitions', 'finished_iterations'])

    @classmethod
    def get_progress(cls, config_file, experiment_selectors=None) -> Tuple[float, List[ExperimentProgress]]:
        experiments_config = cls.load_experiments(config_file, experiment_selectors)
        total_progress = .0
        experiment_progress = []

        for config in experiments_config:
            exp_progress, rep_progress, finished_repetitions, finished_iterations = cls.__experiment_progress(config)
            total_progress += exp_progress / len(experiments_config)
            experiment_progress.append(ClusterWork.ExperimentProgress(config['experiment_name'], config['name'],
                                                                      config['iterations'],
                                                                      config['repetitions'], exp_progress,
                                                                      rep_progress, finished_repetitions,
                                                                      finished_iterations))

        return total_progress, experiment_progress

    @classmethod
    def __show_progress(cls, config_file, experiment_selectors=None, full_progress=False):
        """ shows the progress of all experiments defined in the config_file.
        """
        total_progress, experiments_progress = cls.get_progress(config_file, experiment_selectors)

        for exp_progress in experiments_progress:
            # progress bar
            num_indicators = 25
            num_marked_indicators = round(num_indicators * exp_progress.exp_progress)
            bar = "["
            bar += "." * max(num_marked_indicators - 2, 0) + '‍🌪 ' * (num_marked_indicators > 0)
            bar += " " * (num_indicators - num_marked_indicators - (num_marked_indicators == 1))
            bar += "]"

            finished_reps = '{}/{}'.format(exp_progress.finished_repetitions, exp_progress.num_repetitions)
            print('{:5.1f}% {} {}   {}'.format(exp_progress.exp_progress * 100, bar, finished_reps,
                                               exp_progress.name))

            if full_progress:
                for i, (p, f_i) in enumerate(zip(exp_progress.rep_progress, exp_progress.finished_iterations)):
                    num_indicators = 25
                    num_marked_indicators = round(num_indicators * p)
                    bar = "["
                    bar += "." * max(num_marked_indicators - 2, 0) + '💃🏽' * (num_marked_indicators > 0)
                    bar += " " * (num_indicators - num_marked_indicators)
                    bar += "]"
                    print('    • {:2d}: {:5.1f}% {} {:3}/{}'.format(i + 1, p * 100, bar, f_i,
                                                                    exp_progress.num_iterations))
                print()
        print()

        # print total progress
        num_indicators = 50
        num_marked_indicators = round(num_indicators * total_progress)
        bar = "["
        bar += "." * max(num_marked_indicators - 2, 0) + '🐌' * (num_marked_indicators > 0)
        bar += " " * (num_indicators - num_marked_indicators)
        bar += "]"
        print('  Total: {:5.1f}% {}\n'.format(total_progress * 100, bar))

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
            results_dfs = [ClusterWork.load_repetition_results(config, r) for r in range(config['repetitions'])]
            results_df = None

            for df in filter(lambda d: isinstance(d, pd.DataFrame), results_dfs):
                if results_df is None:
                    results_df = df
                else:
                    results_df = pd.concat([results_df, df])

            return results_df

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
    def __experiment_progress(config) -> (float, [float], int, [int]):
        rep_progress = [ClusterWork.__repetition_progress(config, rep) for rep in range(config['repetitions'])]
        rep_progress_f, rep_progress_i = map(list, zip(*rep_progress))
        exp_progress_f = sum(rep_progress_f) / (config['repetitions'])
        exp_progress_i = sum(map(lambda i: i == config['iterations'], rep_progress_i))
        return exp_progress_f, rep_progress_f, exp_progress_i, rep_progress_i

    @staticmethod
    def __repetition_progress(config, rep) -> (float, int):
        log_df = ClusterWork.load_repetition_results(config, rep)
        if log_df is None:
            return .0, 0
        completed_iterations = log_df.shape[0]
        return float(completed_iterations) / config['iterations'], completed_iterations

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
