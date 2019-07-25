import logging
import sys

import gin


class _CWFormatter(logging.Formatter):
    def __init__(self):
        super(_CWFormatter, self).__init__()
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


@gin.configurable('logging', module='cluster_work')
def init_logging(log_level=logging.INFO, cw_log_level=logging.INFO):
    logging.root.setLevel(log_level)
    _logger.setLevel(cw_log_level)
    _logging_filtered_std_handler.setLevel(level=log_level)


def log_info_message(message: str, border_start_char=None, border_end_char=None):
    if border_start_char:
        _logger.log(INFO_BORDER, border_start_char * 52)
    _logger.log(INFO_CONTNT, '>  ' + message)
    if border_end_char:
        _logger.log(INFO_BORDER, border_end_char * 52)


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
