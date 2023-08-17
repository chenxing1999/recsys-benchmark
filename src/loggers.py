import datetime
import os
from abc import ABC

import loguru
from torch.utils.tensorboard import SummaryWriter


class ILogger(ABC):
    def log_metric(self, metric_name: str, value, step=None):
        ...

    def debug(self, msg: str):
        ...

    def info(self, msg: str):
        ...


class Logger(ILogger):
    """Custom logger object to store training log"""

    instance = None

    def __init__(self, level="INFO", log_folder="logs", log_name=None):
        self._logger: loguru.Logger = loguru.logger
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = os.path.join(log_folder, log_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log")

        self._level = level
        self._logger.add(log_file, level=level)

        self._summary_writer = SummaryWriter(log_dir)

        Logger.instance = self

    def log_metric(self, metric_name, value, step=None):
        if isinstance(value, float):
            msg = f"{metric_name}: {value:.4f}"
        else:
            msg = f"{metric_name}: {value}"

        if step:
            msg = f"{step=} - {msg}"
        self._logger.log(self._level, msg)
        self._summary_writer.add_scalar(metric_name, value, step)

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    @classmethod
    def get_logger(cls):
        return cls.instance
