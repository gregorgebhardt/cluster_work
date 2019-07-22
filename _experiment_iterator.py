import itertools
import gin


class ExperimentIterator:
    _iterator_func = None

    def __init__(self, param_dict: dict = None):
        self._param_dict = param_dict
        self._iterator = None

    def __next__(self):
        with gin.unlock_config():
            parameter_set = list(zip(self._param_dict.keys(), self._iterator.__next__()))
            for k, v in parameter_set:
                gin.bind_parameter(k, v)

        return parameter_set

    def __iter__(self):
        if self._param_dict:
            self._iterator = self._iterator_func(*self._param_dict.values())
            return self
        return iter([None])


@gin.configurable(module='cluster_work')
class List(ExperimentIterator):
    _iterator_func = zip


@gin.configurable(module='cluster_work')
class Grid(ExperimentIterator):
    _iterator_func = itertools.product
