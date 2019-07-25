import abc
import argparse
import fnmatch
import os
import re
import signal
import sys
import time
import zlib

import gc
import gin
import numpy as np
import pandas as pd
import yaml

from ._experiment import ClusterWorkExperiment
from ._tools import format_time, flatten_dict
from ._logging import _logger, log_info_message

# global MPI handle, will be set to mpi4py.MPI if mpi support is used
MPI = None


def _sigint_handler(sgn, _):
    _logger.warning('Received signal {}, will exit program.'.format(sgn))
    if MPI:
        MPI.COMM_WORLD.Abort(0)
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


class ClusterWork(object):
    # change this in subclass, if you support restoring state on iteration level
    _restore_supported = False
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _RESTART_FULL_REPETITIONS = False
    _MP_CONTEXT = 'fork'

    _parser = argparse.ArgumentParser()
    _parser.add_argument('config', metavar='CONFIG.gin', type=argparse.FileType('r'))
    _parser.add_argument('-m', '--mpi', action='store_true',
                         help='Runs the experiments with mpi support.')
    _parser.add_argument('-g', '--mpi_groups', nargs='?', type=int,
                         help='The number of MPI groups to create.')
    _parser.add_argument('-j', '--job', nargs='+', type=int, default=None,
                         help='Run only the specified job from the created work (can be used with slurm arrays).'
                              'Note that each repetition counts as a single job.')
    _parser.add_argument('-d', '--delete', action='store_true',
                         help='CAUTION deletes results of previous runs.')
    _parser.add_argument('-o', '--overwrite', action='store_true',
                         help='CAUTION overwrites results of previous runs if config has changed.')
    _parser.add_argument('-e', '--experiment_scopes', nargs='+',
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
    _parser.add_argument('-r', '--repetition', type=int,
                         help='Start only given repetition, assumes that only one experiment will be started.')
    _parser.add_argument('-i', '--iteration', type=int,
                         help='Restart repetition from iteration i, works only together with -r/--repetition.')

    __run_with_mpi = False

    def __init__(self):

        self.__completed = False
        self._COMM = None

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

    # @staticmethod
    # def get_experiment_config(path, config_filename='experiment.yml'):
    #     """ reads the parameters of the experiment (= path) given.
    #     """
    #     with open(os.path.join(path, config_filename), 'r') as f:
    #         config = yaml.load(f)
    #         return config

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

    # @classmethod
    # def load_experiments(cls, config_file, experiment_selectors=None):
    #     """loads all experiment configurations from the given stream, merges them with the default configuration and
    #     expands list or grid parameters
    #
    #     :config_file: file stream of the configuration yaml for the experiments that should be loaded.
    #     :experiment_selectors: list of experiment names. Only the experiments in this list will be loaded.
    #     :return: returns the experiment configurations
    #     """
    #     try:
    #         _config_documents = [*yaml.load_all(config_file)]
    #     except IOError:
    #         raise SystemExit('config file %s not found.' % config_file)
    #
    #     if _config_documents[0]['name'].lower() == 'default':
    #         default_config = _config_documents[0]
    #         experiments_config = _config_documents[1:]
    #     else:
    #         default_config = dict()
    #         experiments_config = _config_documents
    #
    #     # TODO use namedtuple or own ParameterStore??
    #
    #     # iterate over experiments and compute effective configuration and parameters
    #     # TODO add warning if yaml has parameters that do not appear in cls._default_parameters
    #     effective_experiments = []
    #     for _config_e in experiments_config:
    #         if not experiment_selectors or _config_e['name'] in experiment_selectors:
    #             # merge config with default config from yaml file
    #             _effective_config = deepcopy(default_config)
    #             deep_update(_effective_config, _config_e)
    #
    #             # merge params with default params from subclass
    #             _effective_params = dict()
    #             deep_update(_effective_params, cls._default_params)
    #             deep_update(_effective_params, _effective_config['params'])
    #             _effective_config['params'] = _effective_params
    #
    #             effective_experiments.append(_effective_config)
    #
    #             # check for all required param keys
    #             required_keys = ['name', 'path', 'repetitions', 'iterations']
    #             missing_keys = [key for key in required_keys if key not in _effective_config]
    #             if missing_keys:
    #                 raise IncompleteConfigurationError(
    #                     'config does not contain all required keys: {}'.format(missing_keys))
    #
    #     _experiments = cls.__adapt_experiment_path(effective_experiments)
    #     _experiments = cls.__expand_experiments(_experiments)
    #     _experiments = cls.__adapt_experiment_log_path(_experiments)
    #
    #     return _experiments

    # @staticmethod
    # def __adapt_experiment_path(config_list):
    #     """ adapts the path of the experiment
    #     """
    #     # for one single experiment, still wrap it in list
    #     if type(config_list) == dict:
    #         config_list = [config_list]
    #
    #     expanded_config_list = []
    #     for config in config_list:
    #         config['_config_path'] = config['path']
    #         config['path'] = os.path.join(config['path'], config['name'])
    #         expanded_config_list.append(config)
    #
    #     return expanded_config_list

    # @staticmethod
    # def __adapt_experiment_log_path(config_list):
    #     """ adapts the log path of the experiment and sets the log-path
    #     """
    #     # for one single experiment, still wrap it in list
    #     if type(config_list) == dict:
    #         config_list = [config_list]
    #
    #     expanded_config_list = []
    #     for config in config_list:
    #         if 'log_path' in config:
    #             config['log_path'] = os.path.join(config['log_path'], config['name'])
    #         else:
    #             config['log_path'] = os.path.join(config['path'], 'log')
    #         expanded_config_list.append(config)
    #
    #     return expanded_config_list

    # @staticmethod
    # def __expand_experiments(config_list):
    #     """ expands the parameters list according to one of these schemes:
    #         grid: every list item is combined with every other list item
    #         list: every n-th list item of parameter lists are combined
    #     """
    #     if type(config_list) == dict:
    #         config_list = [config_list]
    #
    #     # get all options that are iteratable and build all combinations (grid) or tuples (list)
    #     expanded_config_list = []
    #     for config in config_list:
    #         if 'grid' in config or 'list' in config:
    #             if 'grid' in config:
    #                 # if we want a grid then we choose the product of all parameters
    #                 iter_fun = itertools.product
    #                 key = 'grid'
    #             else:
    #                 # if we want a list then we zip the parameters together
    #                 iter_fun = zip
    #                 key = 'list'
    #
    #             # TODO add support for both list and grid
    #
    #             # convert list/grid dictionary into flat dictionary, where the key is a tuple of the keys and the
    #             # value is the list of values
    #             tuple_dict = flatten_dict_to_tuple_keys(config[key])
    #             _param_names = ['.'.join(t) for t in tuple_dict]
    #
    #             # create a new config for each parameter setting
    #             for values in iter_fun(*tuple_dict.values()):
    #                 # create config file for
    #                 _config = deepcopy(config)
    #                 del _config[key]
    #
    #                 _converted_name = '_'.join("{}{}".format(shorten_param(k), v) for k, v in zip(_param_names, values))
    #                 # _converted_name = re.sub("[' \[\],()]", '', _converted_name)
    #                 _converted_name = re.sub("[' ]", '', _converted_name)
    #                 _converted_name = re.sub('["]', '', _converted_name)
    #                 _converted_name = re.sub("[(\[]", '_', _converted_name)
    #                 _converted_name = re.sub("[)\]]", '', _converted_name)
    #                 _converted_name = re.sub("[,]", '_', _converted_name)
    #                 _config['_experiment_path'] = config['path']
    #                 _config['path'] = os.path.join(config['path'], _converted_name)
    #                 _config['experiment_name'] = _config['name']
    #                 _config['name'] += '__' + _converted_name
    #                 # if 'log_path' in config:
    #                 #     _config['log_path'] = os.path.join(config['log_path'], config['name'], _converted_name, 'log')
    #                 # else:
    #                 #     _config['log_path'] = os.path.join(_config['path'], 'log')
    #                 for i, t in enumerate(tuple_dict.keys()):
    #                     insert_deep_dictionary(_config['params'], t, values[i])
    #                 expanded_config_list.append(_config)
    #         else:
    #             expanded_config_list.append(config)
    #
    #     return expanded_config_list

    @classmethod
    def __init_experiments(cls, config_file, experiments=None, delete_old=False, ignore_config=False,
                           overwrite_old=False, return_all=False):
        """initializes the experiment by loading the configuration file and creating the directory structure.
        :return:
        """
        # TODO move this to ClusterWorkExperiment, ideas: pass class -> instantiate -> check parameters
        #  other idea: store gin_file in experiment folder and only compare gin files (I think not supported by gin.)
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

        if return_all:
            return expanded_experiments

        return run_experiments

    @classmethod
    def __run_mpi(cls, mpi_groups, job_idx=None):
        from mpi4py.futures import MPICommExecutor
        import socket

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

            _logger.debug('[rank {}] [{}] Starting <{}> - Rep {}.'.format(MPI.COMM_WORLD.rank, hostname,
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

            try:
                repetition_results = _instance.__init_rep(_config, _r).__run_rep(_config, _r)
            except Exception:
                _logger.exception('[rank {}] [{}] Caught exception in <{}> - Rep {}. '
                                  'Sending Abort() to my communicator'.format(MPI.COMM_WORLD.rank, hostname,
                                                                              _config['name'], _r + 1))
                _control = 'break'
                MPI.COMM_GROUP.bcast(_control, root=0)
                _instance._COMM.Abort(1)
                raise

            gc.collect()

            _logger.debug('[rank {}] [{}] finished <{}> - Rep {}'.format(MPI.COMM_WORLD.rank, hostname,
                                                                         _config['name'], _r + 1))

            # return repetition_results
            return _config, _r, repetition_results

        def finalize_worker():
            _logger.debug('[rank {}] [{}] finalizing worker...'.format(MPI.COMM_WORLD.rank, hostname))
            _control = 'break'
            MPI.COMM_GROUP.bcast(_control, root=0)
            time.sleep(5)

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
                                                      overwrite_old=options.overwrite,
                                                      return_all=job_idx is not None)

                    _logger.debug("[rank {}] [{}] emitting the following initial work:".format(MPI.COMM_WORLD.rank,
                                                                                               hostname))

                    work_list = []
                    if isinstance(options.repetition, int):
                        exp_config = exp_list[0]
                        if not 0 <= options.repetition < exp_config['repetitions']:
                            # _logger.error('Repetition has to be in range [0, {}]'.format(num_repetitions))
                            raise InvalidParameterArgument(
                                'Repetition has to be in range [0, {}]'.format(exp_config['repetitions']))
                        _logger.info("     - <{}>".format(exp_config['name']))
                        _logger.info("       repetition {}".format(options.repetition + 1))
                        work_list = [(exp_config, options.repetition)]
                    else:
                        for exp_config in exp_list:
                            _logger.info("     - <{}>".format(exp_config['name']))
                            _logger.info("       creating {} repetitions...".format(exp_config['repetitions']))
                            exp_work = [(exp_config, rep) for rep in range(exp_config['repetitions'])]
                            work_list.extend(exp_work)

                    if job_idx is not None:
                        if isinstance(job_idx, int):
                            idx = job_idx
                        elif isinstance(job_idx, list):
                            if len(job_idx) == 1:
                                idx = job_idx[0]
                            else:
                                idx = slice(*job_idx)
                        else:
                            raise NotImplementedError('could not process job_idx of type {}'.format(type(job_idx)))
                        work_list = work_list[idx]
                        if not isinstance(work_list, list):
                            work_list = [work_list]
                        _logger.info('Selecting the following jobs to be executed:')
                        for (conf, r) in work_list:
                            _logger.info('     - <{}> - Rep {}'.format(conf['name'], r + 1))

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

                    if not job_idx:
                        # if we run with job_idx, we will not write the combined results file
                        for exp_name, results in exp_results.items():
                            exp_config = exp_configs[exp_name]
                            with open(os.path.join(exp_config['path'], 'results.csv'), 'w') as results_file:
                                results.to_csv(results_file, **cls._pandas_to_csv_options)

                    executor.submit(finalize_worker)

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
                    _logger.debug('[rank {}] [{}] received command {} '.format(MPI.COMM_WORLD.rank, hostname, control)
                                  + 'will send Abort() and exit.')
                    MPI.COMM_GROUP.Abort(0)
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

        # TODO: add default gin config?
        gin.parse_config(options.config)

        cls._RESTART_FULL_REPETITIONS = options.restart_full_repetitions

        if options.progress or options.full_progress:
            # TODO iterate over experiment scopes if given
            for exp in ClusterWorkExperiment().expand():
                exp.print_progress(full_progress=options.full_progress)

        if options.mpi:
            try:
                from mpi4py import MPI as _MPI
                import cloudpickle
                global MPI
                MPI = _MPI
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

            cls.__run_mpi(mpi_groups=options.mpi_groups, job_idx=options.job)
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
                    log_info_message('Running Repetition {} '.format(repetition[1] + 1), border_start_char='=',
                                     border_end_char='/')
                    result = cls().__init_rep(*repetition).__run_rep(*repetition)
                    _elapsed_time = time.perf_counter() - time_start
                    log_info_message('Finished Repetition {}'.format(repetition[1] + 1))
                    log_info_message('Elapsed time: {}'.format(format_time(_elapsed_time)))
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

        # TODO move this into the logging module
        # # set logging handlers for current repetition
        # file_handler_mode = 'a' if n_finished_its else 'w'
        # file_handler = logging.FileHandler(os.path.join(self._log_path_rep, 'log.txt'), file_handler_mode)
        # file_handler.setLevel(self._LOG_LEVEL)
        # file_handler.setFormatter(_logging_formatter)
        # # file_handler.addFilter(lambda lr: lr.levelno <= logging.ERROR)
        # if self.__run_with_mpi and MPI.COMM_WORLD.size > 1:
        #     logging.root.handlers.clear()
        #     _logger.handlers.clear()
        #     if MPI.COMM_WORLD.rank == 0 and self.__num_mpi_groups == 1:
        #         # if we run just one group, rank 0 can output to stdout
        #         logging.root.handlers = [file_handler,
        #                                  _logging_filtered_std_handler,
        #                                  _logging_err_handler]
        #         _logger.handlers = [file_handler,
        #                             _logging_filtered_std_handler,
        #                             _logging_err_handler,
        #                             _info_border_output_handler]
        #     elif self._COMM.rank == 0:
        #         logging.root.handlers = [file_handler, _logging_err_handler]
        #         _logger.handlers = [file_handler,
        #                             _logging_filtered_std_handler,
        #                             _logging_err_handler,
        #                             _info_content_output_handler]
        #     else:
        #         logging.root.addHandler(_logging_err_handler)
        #         _logger.addHandler(_logging_err_handler)
        #
        # else:
        #     logging.root.handlers.clear()
        #     logging.root.handlers = [file_handler, _logging_filtered_std_handler, _logging_err_handler]
        #     _logger.addHandler(file_handler)

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
            mean_iteration_time = None
            expected_total_time = None
            time_start = time.perf_counter()
            try:
                log_info_message('Starting Iteration {}/{} of Repetition {}/{}'.format(
                    it + 1, self._iterations, rep + 1, self._repetitions),
                    border_start_char='-', border_end_char='-')
                it_result = self.iterate(config, rep, it)
                iteration_time = time.perf_counter() - time_start
                repetition_time += iteration_time
                mean_iteration_time = repetition_time / (it + 1)
                expected_total_time = mean_iteration_time * self._iterations

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
                log_info_message('Finished Iteration {}/{} of Repetition {}/{}'.format(
                    it + 1, self._iterations, rep + 1, self._repetitions), border_start_char='-')
                log_info_message('Iteration time: {} [{}]'.format(format_time(iteration_time),
                                                                  format_time(mean_iteration_time)))
                log_info_message('Repetition time: {} [{}]'.format(format_time(repetition_time),
                                                                   format_time(expected_total_time)))

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
        self._no_gui = (not config['gui'] if 'gui' in config else False) or self.__run_with_mpi
        self._seed_base = zlib.adler32(self._name.encode()) % int(1e6)
        self._seed = self._seed_base + 1000 * rep
        if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
            self._seed += int(1e5 * self._COMM.rank)

        # set params of this repetition
        self._params = config['params']
        self._rep = rep

        self.reset(config, rep)

        return self

    @staticmethod
    def __convert_param_to_dirname(param):
        """ Helper function to convert a parameter value to a valid directory name. """
        if type(param) == str:
            return param
        else:
            return re.sub("0+$", '0', '%f' % param)



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

    @classmethod
    def iterate_config_and_results(cls, config_filename, experiment_selectors=None):
        with open(config_filename, 'r') as config_file:
            experiment_configs = cls.load_experiments(config_file, experiment_selectors)

        def config_and_results_generator():
            for config in experiment_configs:
                results = ClusterWork.load_experiment_results(config)
                yield config, results

        return config_and_results_generator

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


class IncompleteConfigurationError(Exception):
    pass


class InvalidParameterArgument(Exception):
    pass
