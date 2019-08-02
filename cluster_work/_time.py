import time


class Timer:
    def __init__(self):
        self._counter = 0
        self._total_time = .0
        self._start_time = None
        self._measured_time = None

    @property
    def total_time(self):
        return self._total_time

    @property
    def mean_time(self):
        return self._total_time / self._counter

    @property
    def measured_time(self):
        return self._measured_time

    def __enter__(self):
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._counter += 1
        self._measured_time = time.perf_counter() - self._start_time
        self._total_time += self._measured_time

    def expected_total_time(self, iterations_to_come):
        return self._total_time + self.mean_time * iterations_to_come
