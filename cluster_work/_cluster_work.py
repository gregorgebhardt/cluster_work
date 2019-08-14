import abc
import argparse
import os
import sys
import zlib

import gc
import gin
import numpy as np

from ._experiment import Experiment, ExperimentCollection
from ._logging import log_info_message, cw_logger, log_repetition_to_file, init_logging
from ._results import RepetitionResults
from ._time import Timer
from ._tools import format_counter, format_time

# # global MPI handle, will be set to mpi4py.MPI if mpi support is used
# MPI = None
#
#
# def _sigint_handler(sgn, _):
#     cw_logger.warning('Received signal {}, will exit program.'.format(sgn))
#     if MPI:
#         MPI.COMM_WORLD.Abort(0)
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, _sigint_handler)
# signal.signal(signal.SIGTERM, _sigint_handler)

class ClusterWorkMeta(abc.ABCMeta):
    def __call__(cls, experiment: Experiment, rep: int, *args, **kwargs):
        cw_obj = ClusterWork.__new__(cls, experiment=experiment, rep=rep, *args, **kwargs)
        cw_obj.__init__(*args, **kwargs)
        return cw_obj


class ClusterWork(metaclass=ClusterWorkMeta):
    # change this in subclass, if you support restoring state on iteration level
    restore_supported = False
    # _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")
    _RESTART_FULL_REPETITIONS = False
    # _MP_CONTEXT = 'fork'

    _parser = argparse.ArgumentParser()
    _parser.add_argument('gin_file', metavar='CONFIG.gin', type=str)
    # _parser.add_argument('-m', '--mpi', action='store_true',
    #                      help='Runs the experiments with mpi support.')
    # _parser.add_argument('-g', '--mpi_groups', nargs='?', type=int,
    #                      help='The number of MPI groups to create.')
    # _parser.add_argument('-j', '--job', nargs='+', type=int, default=None,
    #                      help='Run only the specified job from the created work (can be used with slurm arrays).'
    #                           'Note that each repetition counts as a single job.')
    _parser.add_argument('-d', '--delete', action='store_true',
                         help='CAUTION deletes results of previous runs.')
    _parser.add_argument('-o', '--overwrite', action='store_true',
                         help='CAUTION overwrites results of previous runs if config has changed.')
    # _parser.add_argument('-e', '--experiment_scopes', nargs='+',
    #                      help='Allows to specify which experiments should be run.')
    _parser.add_argument('-p', '--progress', action='store_true',
                         help='Outputs the progress of the experiment and exits.')
    _parser.add_argument('-P', '--full_progress', action='store_true',
                         help='Outputs a more detailed progress of the experiment and exits.')
    _parser.add_argument('-i', '--ignore_old', action='store_true',
                         help='Ignores changes in the configuration file for skipping or overwriting experiments..')
    _parser.add_argument('--restart_full_repetitions', action='store_true')
    _parser.add_argument('-r', '--repetition', type=int,
                         help='Start only given repetition, assumes that only one experiment will be started.')
    # _parser.add_argument('-i', '--iteration', type=int,
    #                      help='Restart repetition from iteration i, works only together with -r/--repetition.')

    # __run_with_mpi = False

    # idea: maybe we could use this to add MPI functionality?
    # def __init_subclass__(cls, **kwargs):
    #     super(ClusterWork, cls).__init_subclass__(**kwargs)

    def __new__(cls, experiment: Experiment, rep: int, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self.__rep: int = rep
        self.__experiment = experiment
        self.__results: RepetitionResults = RepetitionResults(repetition=rep,
                                                              iterations=experiment.iterations,
                                                              path=experiment.path)
        self.__timer = Timer()

        return self

    def __init__(self, *args, **kwargs):
        if 'experiment' in kwargs:
            del kwargs['experiment']
        if 'rep' in kwargs:
            del kwargs['rep']
        super(ClusterWork, self).__init__(*args, **kwargs)

    @property
    def experiment(self) -> Experiment:
        return self.__experiment

    @property
    def rep(self) -> int:
        return self.__rep

    @property
    def it(self) -> int:
        return self.results.it

    @property
    def results(self) -> RepetitionResults:
        return self.__results

    @property
    def log_path_rep(self):
        return os.path.join(self.experiment.log_path, 'rep_{:02d}'.format(self.__rep), '')

    ######

    # @classmethod
    # def __run_mpi(cls, mpi_groups, job_idx=None):
    #     from mpi4py.futures import MPICommExecutor
    #     import socket
    #
    #     cls.__num_mpi_groups = mpi_groups
    #     hostname = socket.gethostname()
    #
    #     if mpi_groups:
    #         assert mpi_groups <= MPI.COMM_WORLD.size, 'mpi_groups must be <= {}'.format(MPI.COMM_WORLD.size)
    #         if mpi_groups == 1:
    #             MPI.COMM_GROUP = MPI.COMM_WORLD
    #             MPI.COMM_JOB = MPI.COMM_SELF if MPI.COMM_WORLD.rank == 0 else MPI.COMM_NULL
    #         else:
    #             if 0 < MPI.COMM_WORLD.rank < (MPI.COMM_WORLD.size - ((MPI.COMM_WORLD.size - 1) % mpi_groups)):
    #                 MPI.COMM_GROUP = MPI.COMM_WORLD.Split((MPI.COMM_WORLD.rank - 1) % mpi_groups,
    #                                                       (MPI.COMM_WORLD.rank - 1) // mpi_groups)
    #                 MPI.COMM_GROUP.name = "comm_group_{}".format((MPI.COMM_WORLD.rank - 1) % mpi_groups)
    #             else:
    #                 MPI.COMM_GROUP = MPI.COMM_WORLD.Split(MPI.UNDEFINED, MPI.COMM_WORLD.rank)
    #             if MPI.COMM_WORLD.rank == 0 or (MPI.COMM_GROUP and MPI.COMM_GROUP.rank == 0):
    #                 MPI.COMM_JOB = MPI.COMM_WORLD.Split(0, MPI.COMM_WORLD.rank)
    #             else:
    #                 MPI.COMM_JOB = MPI.COMM_WORLD.Split(MPI.UNDEFINED, MPI.COMM_WORLD.rank)
    #     else:
    #         MPI.COMM_JOB = MPI.COMM_WORLD
    #         MPI.COMM_GROUP = MPI.COMM_NULL
    #
    #     _logger.debug('[rank {}] [{}] {}{}'.format(MPI.COMM_WORLD.rank, hostname,
    #                                                'in {} '.format(MPI.COMM_GROUP.name) if MPI.COMM_GROUP else '',
    #                                                'in comm_job' if MPI.COMM_JOB else ''))
    #
    #     def _run_repetition(_config, _r):
    #         """Runs a single repetition of the experiment by calling run_rep(exp_config, r) on a new instance.
    #
    #         :param _config: the configuration document for the experiment
    #         :param _r: the repetition number
    #         """
    #         import socket
    #         hostname = socket.gethostname()
    #         # if 'MPI' in globals():
    #         #     MPI = globals()['MPI']
    #         # else:
    #         #     _logger.warning('no MPI in global variables')
    #         #     from mpi4py import MPI
    #         #     MPI.COMM_GROUP = MPI.COMM_SELF
    #
    #         _logger.debug('[rank {}] [{}] Starting <{}> - Rep {}.'.format(MPI.COMM_WORLD.rank, hostname,
    #                                                                       _config['name'], _r + 1))
    #         if mpi_groups:
    #             _control = 'run_rep'
    #             MPI.COMM_GROUP.bcast(_control, root=0)
    #
    #             _instance = cls()
    #             _instance._COMM = MPI.COMM_GROUP
    #             MPI.COMM_GROUP.bcast(_config, root=0)
    #             MPI.COMM_GROUP.bcast(_r, root=0)
    #         else:
    #             _instance = cls()
    #             _instance._COMM = MPI.COMM_SELF
    #
    #         try:
    #             repetition_results = _instance.__init_rep(_config, _r).__run_rep(_config, _r)
    #         except Exception:
    #             _logger.exception('[rank {}] [{}] Caught exception in <{}> - Rep {}. '
    #                               'Sending Abort() to my communicator'.format(MPI.COMM_WORLD.rank, hostname,
    #                                                                           _config['name'], _r + 1))
    #             _control = 'break'
    #             MPI.COMM_GROUP.bcast(_control, root=0)
    #             _instance._COMM.Abort(1)
    #             raise
    #
    #         gc.collect()
    #
    #         _logger.debug('[rank {}] [{}] finished <{}> - Rep {}'.format(MPI.COMM_WORLD.rank, hostname,
    #                                                                      _config['name'], _r + 1))
    #
    #         # return repetition_results
    #         return _config, _r, repetition_results
    #
    #     def finalize_worker():
    #         _logger.debug('[rank {}] [{}] finalizing worker...'.format(MPI.COMM_WORLD.rank, hostname))
    #         _control = 'break'
    #         MPI.COMM_GROUP.bcast(_control, root=0)
    #         time.sleep(5)
    #
    #     exp_results = dict()
    #     exp_configs = dict()
    #
    #     if MPI.COMM_JOB != MPI.COMM_NULL:
    #         with MPICommExecutor(MPI.COMM_JOB, root=0) as executor:
    #             if executor is not None:
    #                 _logger.debug("[rank {}] [{}] parsing arguments and initializing experiments".format(
    #                     MPI.COMM_WORLD.rank, hostname))
    #
    #                 options = cls._parser.parse_args()
    #                 exp_list = cls.__init_experiments(config_file=options.config, experiments=options.experiments,
    #                                                   delete_old=options.delete,
    #                                                   ignore_config=options.ignore_config,
    #                                                   overwrite_old=options.overwrite,
    #                                                   return_all=job_idx is not None)
    #
    #                 _logger.debug("[rank {}] [{}] emitting the following initial work:".format(MPI.COMM_WORLD.rank,
    #                                                                                            hostname))
    #
    #                 work_list = []
    #                 if isinstance(options.repetition, int):
    #                     exp_config = exp_list[0]
    #                     if not 0 <= options.repetition < exp_config['repetitions']:
    #                         # _logger.error('Repetition has to be in range [0, {}]'.format(num_repetitions))
    #                         raise InvalidParameterArgument(
    #                             'Repetition has to be in range [0, {}]'.format(exp_config['repetitions']))
    #                     _logger.info("     - <{}>".format(exp_config['name']))
    #                     _logger.info("       repetition {}".format(options.repetition + 1))
    #                     work_list = [(exp_config, options.repetition)]
    #                 else:
    #                     for exp_config in exp_list:
    #                         _logger.info("     - <{}>".format(exp_config['name']))
    #                         _logger.info("       creating {} repetitions...".format(exp_config['repetitions']))
    #                         exp_work = [(exp_config, rep) for rep in range(exp_config['repetitions'])]
    #                         work_list.extend(exp_work)
    #
    #                 if job_idx is not None:
    #                     if isinstance(job_idx, int):
    #                         idx = job_idx
    #                     elif isinstance(job_idx, list):
    #                         if len(job_idx) == 1:
    #                             idx = job_idx[0]
    #                         else:
    #                             idx = slice(*job_idx)
    #                     else:
    #                         raise NotImplementedError('could not process job_idx of type {}'.format(type(job_idx)))
    #                     work_list = work_list[idx]
    #                     if not isinstance(work_list, list):
    #                         work_list = [work_list]
    #                     _logger.info('Selecting the following jobs to be executed:')
    #                     for (conf, r) in work_list:
    #                         _logger.info('     - <{}> - Rep {}'.format(conf['name'], r + 1))
    #
    #                 executor_results = executor.starmap(_run_repetition, work_list)
    #
    #                 for _config, _r, _rep_results in executor_results:
    #                     _logger.debug("[rank {}] [{}] adding results from <{}> - Rep {}".format(
    #                         MPI.COMM_WORLD.rank, hostname, _config['name'], _r + 1))
    #
    #                     if _config['name'] not in exp_results:
    #                         index = pd.MultiIndex.from_product([range(_config['repetitions']),
    #                                                             range(_config['iterations'])])
    #                         index.set_names(['r', 'i'], inplace=True)
    #                         exp_results[_config['name']] = pd.DataFrame(index=index, columns=_rep_results.columns)
    #                         exp_configs[_config['name']] = _config
    #                     exp_results[_config['name']].update(_rep_results)
    #
    #                 _logger.info("[rank {}] [{}] saving results to disk".format(MPI.COMM_WORLD.rank, hostname))
    #
    #                 if not job_idx:
    #                     # if we run with job_idx, we will not write the combined results file
    #                     for exp_name, results in exp_results.items():
    #                         exp_config = exp_configs[exp_name]
    #                         with open(os.path.join(exp_config['path'], 'results.csv'), 'w') as results_file:
    #                             results.to_csv(results_file, **cls._pandas_to_csv_options)
    #
    #                 executor.submit(finalize_worker)
    #
    #         if MPI.COMM_GROUP != MPI.COMM_NULL:
    #             _logger.debug('[rank {}] [{}] sending break...'.format(MPI.COMM_WORLD.rank, hostname))
    #             _logger.warning('Will send Abort to {}'.format(MPI.COMM_GROUP.name))
    #             MPI.COMM_GROUP.Abort(0)
    #
    #         if MPI.COMM_WORLD.rank == 0:
    #             # the last one turns off the light
    #             MPI.COMM_WORLD.Abort(0)
    #         sys.exit()
    #
    #     elif MPI.COMM_GROUP != MPI.COMM_NULL:
    #         while True:
    #             _logger.debug('[rank {}] [{}] waiting for control...'.format(MPI.COMM_WORLD.rank, hostname))
    #             control = MPI.COMM_GROUP.bcast(None, root=0)
    #             _logger.debug('[rank {}] [{}] received {}.'.format(MPI.COMM_WORLD.rank, hostname, control))
    #
    #             if control == 'run_rep':
    #                 instance = cls()
    #                 instance._COMM = MPI.COMM_GROUP
    #                 _config = MPI.COMM_GROUP.bcast(None, root=0)
    #                 _r = MPI.COMM_GROUP.bcast(None, root=0)
    #
    #                 instance.__init_rep(_config, _r).__run_rep(_config, _r)
    #                 gc.collect()
    #             else:
    #                 _logger.debug('[rank {}] [{}] received command {} '.format(MPI.COMM_WORLD.rank, hostname, control)
    #                               + 'will send Abort() and exit.')
    #                 MPI.COMM_GROUP.Abort(0)
    #                 break

    # @classmethod
    # def init_from_config(cls, config, rep=0, it=0):
    #     instance = cls().__init_rep_without_checks(config, rep)
    #
    #     instance._log_path_it = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), 'it_{:04d}'.format(it), '')
    #
    #     try:
    #         instance.restore_state(config, rep, it)
    #     except IOError:
    #         _logger.warning('Could not restore experiment {}, rep {} at iteration {}.'.format(config['name'], rep, it))
    #         return None
    #
    #     instance.__iteration = it
    #
    #     def exception_stub(_c, _r, _i):
    #         raise Exception('Experiment not properly initialized. Cannot run.')
    #
    #     instance.iterate = exception_stub
    #     instance.reset = exception_stub
    #
    #     return instance

    @classmethod
    def run(cls):
        """ starts the experiments as given in the config file. """
        arguments = cls._parser.parse_args()

        # idea: add default gin config?
        gin.parse_config_file(arguments.gin_file)

        init_logging()

        cls._RESTART_FULL_REPETITIONS = arguments.restart_full_repetitions

        if arguments.progress or arguments.full_progress:
            # idea: iterate over experiment scopes if given
            for exp in ExperimentCollection(arguments.gin_file).expand():
                exp.print_progress(full_progress=arguments.full_progress)

        # if arguments.mpi:
        #     try:
        #         from mpi4py import MPI as _MPI
        #         import cloudpickle
        #         global MPI
        #         MPI = _MPI
        #         MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)
        #     except ModuleNotFoundError:
        #         _logger.error('ClusterWork requires the mpi4py and cloudpickle packages for distributing jobs via MPI.')
        #         raise
        #     cls.__run_with_mpi = True
        #     cls._MP_CONTEXT = 'forkserver'
        #     import multiprocessing as mp
        #     if not mp.get_start_method(allow_none=True):
        #         mp.set_start_method(cls._MP_CONTEXT)
        #
        #     if MPI.COMM_WORLD.rank == 0:
        #         _logger.info("starting {} with the following arguments:".format(cls.__name__))
        #         for arg, val in vars(arguments).items():
        #             _logger.info("  - {}: {}".format(arg, val))
        #
        #     cls.__run_mpi(mpi_groups=arguments.mpi_groups, job_idx=arguments.job)
        else:
            cw_logger.info("starting {} with the following arguments:".format(cls.__name__))
            for arg, val in vars(arguments).items():
                cw_logger.info("  - {}: {}".format(arg, val))

            for experiment in ExperimentCollection(arguments.gin_file).expand(delete_old_files=arguments.delete):
                # check if experiment exits and if identical or overwrite
                clear_results = False
                if experiment.old_files_exist:
                    if not experiment.old_files_identical:
                        if arguments.ignore_old:
                            cw_logger.warning('Experiment {} has started before, but configuration has '
                                              'changed! '.format(experiment.name)
                                              + 'Starting Experiment anyways due to option -I/--ignore-config')
                        elif arguments.overwrite:
                            # add experiment to clear list
                            cw_logger.warning('Experiment {} has started before, but configuration has '
                                              'changed! Overwriting...'.format(experiment.name))
                            clear_results = True
                        else:
                            # add experiment to skip list
                            cw_logger.warning('Experiment {} has started before, but configuration has '
                                              'changed! Skipping...'.format(experiment.name))
                            cw_logger.warning('--> To overwrite existing results, use the option -o/--overwrite')
                            continue

                # check if only a single repetition should be run
                if isinstance(arguments.repetition, int):
                    if not 0 <= arguments.repetition < experiment.repetitions:
                        raise InvalidParameterArgument('Repetition has to be in range [0, {}]'.format(
                            experiment.repetitions))
                    repetitions = [arguments.repetition]

                else:
                    repetitions = range(experiment.repetitions)

                cw_logger.info("starting experiment {}".format(experiment.name))

                for rep in repetitions:
                    log_info_message('Running Repetition {} '.format(format_counter(rep)),
                                     border_start_char='=', border_end_char='/')
                    exp_timer = Timer()
                    with exp_timer:
                        work = cls(experiment=experiment, rep=rep)
                        if clear_results:
                            work.results.clear()
                        if work.__init_rep():
                            experiment.write_gin_file()
                            work.__run_rep()

                    log_info_message('Finished Repetition {}'.format(format_counter(rep)))
                    log_info_message('Elapsed time: {}'.format(format_time(exp_timer.measured_time)))
                    gc.collect()

        sys.exit(0)

    def __init_rep(self) -> bool:
        """ run a single repetition including directory creation, log files, etc. """
        # set configuration of this repetition
        # idea: move this to some other location (Experiment, Seeder)...
        self._seed_base = zlib.adler32(self.experiment.name.encode()) % int(1e6)
        self._seed = self._seed_base + 1000 * self.rep + 5
        # if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
        #     self._seed += int(1e5 * self._COMM.rank)

        # check if log-file for repetition exists
        rep_has_finished, n_finished_its = self._repetition_has_completed()

        # skip repetition if it has finished
        if rep_has_finished or n_finished_its == self.experiment.iterations:
            cw_logger.info('Repetition {} of experiment {} has finished before. '
                           'Skipping...'.format(format_counter(self.rep), self.experiment.name))
            self.__completed = True
            return False

        log_repetition_to_file(filename=os.path.join(self.experiment.log_path, 'log_rep_{:02}.txt'.format(self.rep)),
                               append=n_finished_its > 0)

        self.reset()

        # if not completed but some iterations have finished, check for restart capabilities
        if self.restore_supported and n_finished_its > 0 and not self._RESTART_FULL_REPETITIONS:
            cw_logger.info('Repetition {} of experiment {} has started before. '
                           'Trying to restore state after iteration {}.'.format(format_counter(self.rep),
                                                                                self.experiment.name, n_finished_its))
            # set start for iterations and restore state in subclass
            try:
                if self.results.it and self.restore_state(self.results.it - 1):
                    cw_logger.info('Restoring iteration succeeded. Restarting after iteration {}.'.format(
                        self.results.it))
                else:
                    cw_logger.warning('Restoring iteration NOT successful. Restarting from iteration 1.')
                    self.results.clear()
            except IOError:
                cw_logger.error('Exception during restore_state of experiment {} in repetition {}.'.format(
                    self.experiment.name, format_counter(self.rep)), exc_info=True)
                self.results.clear()

        else:
            self.results.clear()

        return True

    def __run_rep(self) -> None:
        # log_filename = os.path.join(self._log_path, 'rep_{}.csv'.format(self.repetition))
        if self.results.it > 0:
            self.__repetition_time = self.results[-1].repetition_time

        for it in self.results:
            # update seed
            self._seed = self._seed_base + 1000 * self.rep + it
            # if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
            #     self._seed += int(1e5 * self._COMM.rank)

            log_info_message('Starting Iteration {}/{} of Repetition {}/{}'.format(
                format_counter(it), self.experiment.iterations, format_counter(self.rep), self.experiment.repetitions),
                border_start_char='-', border_end_char='-')

            # run iteration and get results
            try:
                with self.__timer:
                    self.iterate()

                # store timing information
                self.results.store(iteration_time=self.__timer.measured_time,
                                   repetition_time=self.__timer.total_time)

                # save state before results, so that we know the saved state can be restored if we find the results.
                self.save_state(it)

            except ValueError or ArithmeticError or np.linalg.linalg.LinAlgError:
                cw_logger.error('Experiment {} - Repetition {} - Iteration {}'.format(self.experiment.name,
                                                                                      format_counter(self.rep),
                                                                                      format_counter(it)),
                                exc_info=True)
                self.finalize()
                return
            except Exception:
                self.finalize()
                raise
            finally:
                if self.__timer.measured_time is None:
                    with self.__timer:
                        # if no iteration has been measured, create a fake measurement for printing
                        pass
                log_info_message('Finished Iteration {}/{} of Repetition {}/{}'.format(
                    format_counter(it), self.experiment.iterations,
                    format_counter(self.rep), self.experiment.repetitions), border_start_char='-')
                log_info_message('Iteration time: {} [{}]'.format(format_time(self.__timer.measured_time),
                                                                  format_time(self.__timer.mean_time)))
                log_info_message('Repetition time: {} [{}]'.format(format_time(self.__timer.total_time),
                                                                   format_time(self.__timer.expected_total_time(
                                                                       self.experiment.iterations - it))))

        self.finalize()

    # def __init_rep_without_checks(self, config, rep):
    #     # set configuration of this repetition
    #     self._name = config['name']
    #     self._repetitions = config['repetitions']
    #     self._iterations = config['iterations']
    #     self._path = config['path']
    #     self._log_path = config['log_path']
    #     self.log_path_rep = os.path.join(config['log_path'], 'rep_{:02d}'.format(rep), '')
    #     self._plotting = config['plotting'] if 'plotting' in config else True
    #     self._no_gui = (not config['gui'] if 'gui' in config else False) or self.__run_with_mpi
    #     self._seed_base = zlib.adler32(self._name.encode()) % int(1e6)
    #     self._seed = self._seed_base + 1000 * rep
    #     if self.__run_with_mpi and self._COMM.name != 'MPI_COMM_SELF':
    #         self._seed += int(1e5 * self._COMM.rank)
    #
    #     # set params of this repetition
    #     self._params = config['params']
    #     self._rep = rep
    #
    #     self.reset(config, rep)
    #
    #     return self

    def _repetition_has_completed(self) -> (bool, int):
        if self.results.empty:
            return False, 0

        # if repetition has completed
        return len(self.results) == self.experiment.iterations, len(self.results)

    # @staticmethod
    # def __experiment_exists(config):
    #     return os.path.exists(os.path.join(config['path'], 'experiment.yml'))
    #
    # @staticmethod
    # def __experiment_exists_identically(config):
    #     if ClusterWork.__experiment_exists(config):
    #         with open(os.path.join(config['path'], 'experiment.yml'), 'r') as f:
    #             dumped_config = yaml.load(f)
    #             return dumped_config == config
    #
    #     return False
    #
    # @staticmethod
    # def __experiment_has_finished(config):
    #     return os.path.exists(os.path.join(config['path'], 'results.csv'))
    #
    # @staticmethod
    # def __experiment_has_finished_repetitions(config):
    #     for file in os.listdir(config['log_path']):
    #         if fnmatch.fnmatch(file, 'rep_*.csv'):
    #             return True
    #     return False

    @abc.abstractmethod
    def reset(self) -> None:
        """ needs to be implemented by subclass. """
        pass

    @abc.abstractmethod
    def iterate(self) -> None:
        """ needs to be implemented by subclass. """
        pass

    def finalize(self):
        pass

    def save_state(self, iteration: int) -> None:
        """ optionally can be implemented by subclass. """
        pass

    def restore_state(self, iteration: int) -> bool:
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
