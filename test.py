import gin
from cluster_work import ClusterWork

import _experiment_iterator


@gin.configurable
def test_function(a, b=1, c=2):
    print(a, b, c)


@gin.configurable('test_cluster_work')
class TestClusterWork(ClusterWork):
    def __init__(self, test='test', **kwargs):
        super().__init__(**kwargs)

    def reset(self, config: dict, rep: int) -> None:
        pass

    def iterate(self, config: dict, rep: int, n: int) -> dict:
        return {}


if __name__ == '__main__':
    gin.parse_config('test_function.a = 0')
    gin.parse_config_file('test.gin')

    l = _experiment_iterator.List()
    g = _experiment_iterator.Grid()
    for _ in l:
        for _ in g:
            test_function()
