import os.path as osp
import time

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "data")

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up an output directory for logging and returns logger keyword arguments.

    Args:
        exp_name (string): Name for experiment.
        seed (int): Seed for random number generators used by experiment.
        data_dir (string): Path to folder where results should be saved.
            Default is None, which means use the DEFAULT_DATA_DIR.
        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:
        logger_kwargs (dict): Dictionary containing output directory and
            experiment name, to be passed to the logger class constructors.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ""
    relpath = "".join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = "".join([hms_time, "-", exp_name, "_s", str(seed)])
        else:
            subfolder = "".join([exp_name, "_s", str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    return dict(
        output_dir=osp.join(data_dir, relpath),
        exp_name=exp_name,
    )
