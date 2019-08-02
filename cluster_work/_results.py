import abc
import os

import pandas as pd

from typing import Union, Iterator, List

from ._logging import cw_logger


class AbstractResults(abc.ABC):
    _pandas_to_csv_options = dict(na_rep='NaN', sep='\t', float_format="%+.8e")

    def __init__(self, path=None, filename=None, _results_df=None):
        self.path = path
        self.filename = filename
        self._results_df = _results_df

        # if no results where passed in to the constructor, try to load them from the file
        if self._results_df is None:
            self._results_df = self.load_from_file()

    @property
    def empty(self):
        if self._results_df is None:
            return True
        return self._results_df.empty

    def set_path(self, path, force_load=False):
        self.path = path
        if self.empty or force_load:
            self._results_df = self.load_from_file()
        else:
            cw_logger.info('results_df not empty, did not load results from new path.')

    @property
    def _full_path(self):
        return os.path.join(self.path, self.filename)

    # def __getattr__(self, item):
    #     if item in self.__dict__:
    #         return self.__dict__[item]
    #     else:
    #         return self._results_df.__getattr__(item)

    def load_from_file(self) -> Union[pd.DataFrame, None]:
        if self.path is None:
            cw_logger.debug('path is not set, will not load results.')
            return None
        if os.path.exists(self._full_path):
            cw_logger.debug('loading results from {}.'.format(self._full_path))
            rep_results_df = pd.read_csv(self._full_path, sep='\t')
            rep_results_df.set_index(keys=['r', 'i'], inplace=True)
            return rep_results_df
        else:
            cw_logger.debug('{} does not exist'.format(self._full_path))
            return None

    def __str__(self):
        return self._results_df.__str__()


class RepetitionResults(AbstractResults):
    def __init__(self, repetition, iterations, filename=None, **kwargs):
        self._repetition = repetition
        self._iterations = iterations
        self._it = 0

        if filename is None:
            filename = 'rep_{}.csv'.format(self._repetition)

        super(RepetitionResults, self).__init__(filename=filename, **kwargs)

        if self._results_df is not None:
            if self._repetition is None:
                self._repetition = self._results_df.index[0][0]
            if self._iterations is None:
                self._iterations = len(self._results_df)
            if not self._results_df.empty:
                self._it = len(self._results_df)

        # if results_df is None create a new data frame
        if self._results_df is None:
            self._results_df = pd.DataFrame(index=pd.MultiIndex.from_product([[self._repetition], range(iterations)],
                                                                             names=['r', 'i']), dtype=float)

    @property
    def it(self):
        return self._it

    def clear(self):
        self._it = 0
        self._results_df = pd.DataFrame(index=pd.MultiIndex.from_product([[self._repetition], range(self._iterations)],
                                                                         names=['r', 'i']), dtype=float)
        if os.path.exists(self._full_path):
            os.remove(self._full_path)

    def __len__(self):
        return len(self._results_df)

    def __iter__(self):
        for i in range(self._it, self._iterations):
            self._it = i
            yield i
            self._write_iteration()

    def __getitem__(self, iteration):
        if iteration >= 0:
            return self._results_df.iloc[iteration]
        else:
            return self._results_df.iloc[self._it - iteration]

    def store(self, **results):
        for k, v in results.items():
            self._results_df.loc[(self._repetition, self._it), k] = v

    def _write_iteration(self):
        if self.path is None:
            cw_logger.warn("path is not set, will not write repetition results to file.")
            return

        # write first line with header
        if self._it == 0:
            # check if path does not exist
            if not os.path.exists(self.path):
                os.makedirs(self.path, exist_ok=True)
            self._results_df.iloc[[self._it]].to_csv(self._full_path, mode='w', header=True,
                                                     **self._pandas_to_csv_options)
        else:
            self._results_df.iloc[[self._it]].to_csv(self._full_path, mode='a', header=False,
                                                     **self._pandas_to_csv_options)


class ExperimentResults(AbstractResults):
    def __init__(self, repetitions=None, iterations=None, filename=None, **kwargs):
        self._repetitions = repetitions
        self._iterations = iterations

        if filename is None:
            filename = 'results.csv'

        super(ExperimentResults, self).__init__(filename=filename, **kwargs)

        if self._results_df is None:
            self._results_df = pd.DataFrame(index=pd.MultiIndex.from_product([range(self._repetitions),
                                                                              range(self._iterations)],
                                                                             names=['r', 'i']), dtype=float)

        self._rep_iter = -1

    @classmethod
    def merge_repetition_results(cls, *repetition_results: AbstractResults) -> 'ExperimentResults':
        data_frames = [rr._results_df for rr in repetition_results]
        merged_df = pd.concat(data_frames)
        iterations = max([len(df) for df in data_frames])
        return ExperimentResults(repetitions=len(repetition_results), iterations=iterations, results_df=merged_df)

    def repetitions(self):
        for rep, df in self._results_df.groupby(level=0):
            yield RepetitionResults(repetition=rep, iterations=None, _results_df=df)

    def __getitem__(self, item: Union[int, list, tuple, slice]) -> Union[RepetitionResults, List[RepetitionResults]]:
        if isinstance(item, int):
            df = self._results_df.loc[[item]]
            return RepetitionResults(repetition=item, iterations=None, _results_df=df)
        else:
            dfs = self._results_df.loc[item]
            return [RepetitionResults(repetition=rep, iterations=None, _results_df=df) for rep, df in dfs.groupby(
                level=0)]

    def __len__(self):
        return self._repetitions

    def __iter__(self) -> Iterator[RepetitionResults]:
        self._rep_iter = -1
        return self

    def __next__(self) -> RepetitionResults:
        self._rep_iter += 1
        if self._rep_iter >= self._repetitions:
            raise StopIteration
        df = self._results_df.loc[[self._rep_iter]]
        return RepetitionResults(repetition=self._rep_iter, iterations=None, _results_df=df)

    def write(self):
        if self.path is None:
            cw_logger.warn("path is not set, will not write experiment results to file.")
            return
        # check if path does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        self._results_df.to_csv(self._full_path, mode='w', header=True,
                                **self._pandas_to_csv_options)
