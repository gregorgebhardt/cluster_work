import fnmatch
import os
import re
from collections import namedtuple
from typing import List

import gin
import pandas as pd

from ._parameter_iterator import ParameterList, ParameterGrid
from cluster_work._tools import shorten_param
from ._logging import _logger


@gin.configurable('cluster_work', blacklist=['gin_file', '_parent', '_params'])
class ClusterWorkExperiment:
    Options = namedtuple('Options', ('name', 'repetitions', 'iterations', 'path', 'log_path'))
    Progress = namedtuple('Progress', ['name', 'num_repetitions', 'num_iterations', 'exp_progress', 'rep_progress',
                                       'finished_repetitions', 'finished_iterations'])

    def __init__(self, gin_file, name=gin.REQUIRED, iterations=gin.REQUIRED, repetitions=gin.REQUIRED,
                 path=gin.REQUIRED, log_path=None, _parent: 'ClusterWorkExperiment' = None, _params: dict = None):
        self._gin_file = gin_file
        self._log_path_set = True
        if log_path is None:
            self._log_path_set = False
            log_path = os.path.join(self.path, 'log')

        self._options = ClusterWorkExperiment.Options(name=name, repetitions=repetitions, iterations=iterations,
                                                      path=path, log_path=log_path)
        self._parent = _parent

        self._grid = ParameterGrid()
        self._list = ParameterList()
        self._params = _params

        self._results = self._load_results()
        if self._results is None:
            self._results = pd.DataFrame()

        self.__log_path_rep = None
        self.__log_path_rep_exists = False
        self.__log_path_it = None
        self.__log_path_it_exists = False

    @property
    def name(self):
        return self._options.name

    @property
    def repetitions(self):
        return self._options.repetitions

    @property
    def iterations(self):
        return self._options.iterations

    @property
    def path(self):
        return self._options.path

    @property
    def log_path(self):
        return self._options.log_path

    @property
    def is_expanded(self):
        return self._parent is not None

    def expand(self) -> List['ClusterWorkExperiment']:
        if self.is_expanded:
            _logger.warn('Experiment is already expanded!')
            return None
        for g_params in self._grid:
            for l_params in self._list:
                # merge the parameter dicts from list and grid
                params = {**g_params, **l_params}
                # adapt name and path
                adapted_name = '_'.join("{}{}".format(shorten_param(k), v) for k, v in zip(params.keys(),
                                                                                           params.values()))
                adapted_name = re.sub(r"[' ]", '', adapted_name)
                adapted_name = re.sub(r'["]', '', adapted_name)
                adapted_name = re.sub(r"[(\[]", '_', adapted_name)
                adapted_name = re.sub(r"[)\]]", '', adapted_name)
                adapted_name = re.sub(r"[,]", '_', adapted_name)

                adapted_path = os.path.join(self.path, self.name, adapted_name)
                if adapted_name:
                    adapted_name = self.name + '__' + adapted_name
                else:
                    adapted_name = self.name
                if self._log_path_set:
                    adapted_log_path = os.path.join(self.log_path, adapted_name)
                else:
                    adapted_log_path = None

                # yield ClusterWorkOptions with adapted parameters
                yield ClusterWorkExperiment(self._gin_file, name=adapted_name,
                                            iterations=self.iterations,
                                            repetitions=self.repetitions, path=adapted_path,
                                            log_path=adapted_log_path, _parent=self)

    def bind_parameters(self):
        if not self.is_expanded:
            _logger.warn("Experiment is not expanded. Only expanded experiments can be used as context.")
            return
        # bind gin parameters from grid/list
        with gin.unlock_config():
            for k, v in self._params.items():
                gin.bind_parameter(k, v)

    def __enter__(self):
        self.bind_parameters()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO perform exit clean up, e.g. write exceptions to a file, store used gin parameters
        pass

    def create_experiment_directory(self, delete_old_files=False):
        """ creates a subdirectory for the experiment, and deletes existing
            files, if the delete flag is true. Stores the experiment.gin file in the experiment folder.
        """
        # create experiment path and subdir
        os.makedirs(self.path, exist_ok=True)

        # delete old histories if --del flag is active
        if delete_old_files:
            os.system('rm -rf {}/*'.format(self.path))

        # create a directory for the log path
        os.makedirs(self.log_path, exist_ok=True)

        # TODO write a config file for this single exp. in the folder
        #  can only be written after experiment has been instantiated/initialised!!
        #  ideas: perform in experiment context and call at exit gin.operative_config_str()
        #         use finalize and a finalize_hook that reads the parameters in the gin._CONFIG dict.
        # ClusterWork.__write_config_file(config)

    def experiment_exists(self):
        return os.path.exists(os.path.join(self.path, 'experiment.gin'))

    def experiment_exists_identically(self):
        if self.experiment_exists():
            with open(os.path.join(self._options.path, 'experiment.gin'), 'r') as f:
                # TODO implement... How can we obtain all defined gin parameters?
                return False

        return False

    def experiment_has_finished(self):
        return os.path.exists(os.path.join(self.path, 'results.csv'))

    def experiment_has_finished_repetitions(self):
        for file in os.listdir(self.log_path):
            if fnmatch.fnmatch(file, 'rep_*.csv'):
                return True
        return False

    def _load_results(self):
        results_filename = os.path.join(self._options.path, 'results.csv')

        if os.path.exists(results_filename):
            results_df = pd.read_csv(results_filename, sep='\t')
            results_df.set_index(keys=['r', 'i'], inplace=True)
            return results_df
        else:
            results_dfs = [self._load_repetition_results(r) for r in range(self._options.repetitions)]
            results_df = None

            for df in filter(lambda d: isinstance(d, pd.DataFrame), results_dfs):
                if results_df is None:
                    results_df = df
                else:
                    results_df = pd.concat([results_df, df])

            return results_df

    def _load_repetition_results(self, rep):
        rep_results_filename = os.path.join(self._options.log_path, 'rep_{}.csv'.format(rep))

        if os.path.exists(rep_results_filename):
            rep_results_df = pd.read_csv(rep_results_filename, sep='\t')
            rep_results_df.set_index(keys=['r', 'i'], inplace=True)

            return rep_results_df
        else:
            return None

    def repetition_has_completed(self, rep) -> (bool, int, pd.DataFrame):
        log_df = self._load_repetition_results(rep)

        if log_df is None:
            return False, 0, None

        # if repetition has completed
        return log_df.shape[0] == self._options.iterations, log_df.shape[0], log_df

    def experiment_progress(self) -> 'ClusterWorkExperiment.Progress':
        rep_progress = [self.repetition_progress(rep) for rep in range(self._options.repetitions)]
        rep_progress_f, rep_progress_i = map(list, zip(*rep_progress))
        exp_progress_f = sum(rep_progress_f) / self._options.repetitions
        exp_progress_i = sum(map(lambda i: i == self._options.iterations, rep_progress_i))

        return ClusterWorkExperiment.Progress(name=self._options.name, num_iterations=self._options.iterations,
                                              num_repetitions=self._options.repetitions, exp_progress=exp_progress_f,
                                              rep_progress=rep_progress_f, finished_repetitions=exp_progress_i,
                                              finished_iterations=rep_progress_i)

    def repetition_progress(self, rep) -> (float, int):
        # TODO results are already available... use loaded results
        log_df = self._load_repetition_results(rep)
        if log_df is None:
            return .0, 0
        completed_iterations = log_df.shape[0]
        return float(completed_iterations) / self._options.iterations, completed_iterations

    def print_progress(self, full_progress=False):
        """ shows the progress of all experiments defined in the config_file.
        """
        exp_progress = self.experiment_progress()

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

    # @property
    # def log_path_rep(self):
    #     if not self.__log_path_rep_exists:
    #         os.makedirs(self.__log_path_rep, exist_ok=True)
    #         self.__log_path_rep_exists = True
    #     return self.__log_path_rep
    #
    # @log_path_rep.setter
    # def log_path_rep(self, log_path_rep: str):
    #     if os.path.exists(log_path_rep):
    #         if not os.path.isdir(log_path_rep):
    #             raise NotADirectoryError("The log path {} exists but is not a directory".format(log_path_rep))
    #         self.__log_path_rep_exists = True
    #     else:
    #         self.__log_path_rep_exists = False
    #     self.__log_path_rep = log_path_rep
    #
    # @property
    # def log_path_it(self):
    #     if not self.__log_path_it_exists:
    #         os.makedirs(self.__log_path_it, exist_ok=True)
    #         self.__log_path_it_exists = True
    #     return self.__log_path_it
    #
    # @log_path_it.setter
    # def log_path_it(self, log_path_it: str):
    #     if os.path.exists(log_path_it):
    #         if not os.path.isdir(log_path_it):
    #             raise NotADirectoryError("The log path {} exists but is not a directory".format(log_path_it))
    #         self.__log_path_it_exists = True
    #     else:
    #         self.__log_path_it_exists = False
    #     self.__log_path_it = log_path_it
