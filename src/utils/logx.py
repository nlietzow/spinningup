from typing import Any, Optional

import numpy as np
import wandb

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def statistics_scalar(
        x: list[float],
        with_min_and_max: bool = False
):
    """
    Get mean/std and optional min/max of scalar x.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)

    if with_min_and_max:
        min_val = np.min(x) if len(x) > 0 else np.inf
        max_val = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, min_val, max_val
    return mean, std


class Logger:
    """
    A general-purpose logger.
    """

    def __init__(self, wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        """
        Initialize a Logger.
        """
        self.wandb = wandb_run

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.step = 0

    def set_step(self, step: int):
        self.step = step

    @staticmethod
    def log(msg: str, color: str = "green"):
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key: str, val: Any):
        """
        Log a value of some diagnostic.
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                    "Trying to introduce a new key %s that you didn't include in the first iteration"
                    % key
            )
        assert key not in self.log_current_row, (
                "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
                % key
        )
        self.log_current_row[key] = val

    def dump_tabular(self):
        """
        Write all the diagnostics from the current iteration.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        key_str = "%" + "%d" % max_key_len
        fmt = "| " + key_str + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            val_str = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, val_str))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    """

    def __init__(self, wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        super().__init__(wandb_run)
        self.epoch_dict: dict[str, list[float]] = dict()

    def store(self, **kwargs: float):
        """
        Save something into the epoch_logger's current state.
        """
        if self.wandb:
            self.wandb.log(kwargs, step=self.step)

        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(
            self,
            key: str,
            val: Any = None,
            with_min_and_max: bool = False,
            average_only: bool = False,
    ):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            epoch_vals = self.epoch_dict[key]
            stats = statistics_scalar(epoch_vals, with_min_and_max=with_min_and_max)

            # Log to text file via parent class
            super().log_tabular(key if average_only else "Average" + key, stats[0])
            if not average_only:
                super().log_tabular("Std" + key, stats[1])
            if with_min_and_max:
                super().log_tabular("Max" + key, stats[3])
                super().log_tabular("Min" + key, stats[2])

        self.epoch_dict[key] = []
