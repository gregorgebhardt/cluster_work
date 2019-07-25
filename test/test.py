import gin
from cluster_work import ClusterWork
from cluster_work._cluster_work import ClusterWorkOptions

from cluster_work._experiment_iterator import List, Grid


@gin.configurable
def test_function(a, b=1, c=2):
    print(a, b, c)


# @gin.configurable('test_cluster_work')
# class TestClusterWork(ClusterWork):
#     def __init__(self, test='test', **kwargs):
#         super().__init__(**kwargs)
#
#     def reset(self, config: dict, rep: int) -> None:
#         pass
#
#     def iterate(self, config: dict, rep: int, n: int) -> dict:
#         return {}


class A:
    @gin.configurable
    def b(self, name='test'):
        print(name)


if __name__ == '__main__':
    gin.parse_config('test_function.a = 0')
    gin.parse_config_file('test.gin')

    for gwo in ClusterWorkOptions().expand():
        print(gwo.name)
        print(gwo.path)
        print(gwo.log_path)

    pass

    # l = List()
    # g = Grid()
    # for _ in l:
    #     for _ in g:
    #         test_function()
