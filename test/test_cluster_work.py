import gin
from cluster_work import ClusterWork


class TestClusterWork(ClusterWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = None

    def reset(self) -> None:
        self.model = Model()

    def iterate(self) -> None:
        print('iterate')
        res_a, res_b = self.model()
        self.results.store(model_1=res_a, model_2=res_b)


@gin.configurable
class Model:
    def __init__(self, a=gin.REQUIRED, b=gin.REQUIRED):
        self.a = a
        self.b = b

    def __call__(self, *args, **kwargs):
        return self.a, self.b


def finalize_hook(*args, **kwargs):
    pass


gin.config.register_finalize_hook(finalize_hook)


if __name__ == '__main__':
    TestClusterWork.run()
