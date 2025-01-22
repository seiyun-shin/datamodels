# in this file:
# - take an index
# - take a command to run
# - take an out directory

# - run main() which returns dict mapping: [output] -> [value]
# - run for each output key, find mmap in out directory, write to mmap[index]
#   with the value

from fastargs.decorators import param
from fastargs import Param, Section
import tqdm
from functools import cache

import importlib.util
import sys
from pathlib import Path
from .utils import memmap_path, make_config
from .spec import COMPLETED

import numpy as np
import signal
import sys

def alarm_handler(signal, frame):
    print('!' * 80)
    print('Exiting due to timeout!!')
    print('!' * 80)
    sys.exit(0)

signal.signal(signal.SIGALRM, alarm_handler)

INDEX_NONCE = -99999999
LOGDIR_NONCE = ''

Section('worker').params(
    index=Param(int, 'index of this job', default=INDEX_NONCE),
    main_import=Param(str, 'relativer python import module with main() to run', required=True),
    logdir=Param(str, 'file with main() to run', default=LOGDIR_NONCE),
    do_if_complete=Param(bool, 'ignore the completed array, and do the task anyways', is_flag=True),
    job_timeout=Param(int, 'seconds per job', default=99999999),
    # CHANGED: new param so top-level arrays can live in store_dir
    store_dir=Param(str, 'directory where S, margins, etc. are stored', default=LOGDIR_NONCE)
)

def get_mmap(fn, mode):
    """ Cache np.load(...) calls by filename. """
    return np.load(fn, mmap_mode=mode)

def kv_log(k, v, store_dir, index):
    """
    Writes the value `v` to row [index] of the memmapped array named `k`
    in the directory store_dir.
    """
    this_filename = memmap_path(store_dir, k)
    assert this_filename.exists(), f"File not found: {this_filename}"
    mmap = get_mmap(this_filename, 'r+')
    mmap[index] = v
    mmap.flush()

def kv_read(k, store_dir, index):
    """
    Reads row [index] from the memmapped array named `k` in store_dir.
    """
    this_filename = memmap_path(store_dir, k)
    assert this_filename.exists(), f"File not found: {this_filename}"
    mmap = get_mmap(this_filename, 'r')
    return mmap[index]

################################################################################
# 4) The main do_index function
################################################################################
@param('worker.index')
@param('worker.logdir')
@param('worker.do_if_complete')
@param('worker.job_timeout')
@param('worker.store_dir')
def do_index(*_, index, routine, logdir, do_if_complete, job_timeout, store_dir):
    """
    1) Possibly skip if COMPLETED array shows index is done, unless do_if_complete is True.
    2) run routine(index=index, logdir=..., store_dir=...).
    3) kv_log all returned data to store_dir, then mark COMPLETED=1 at row [index].
    """
    logdir_path = Path(logdir)
    print("logging in", logdir_path)

    worker_logs = logdir_path / 'workers' / str(index)
    worker_logs.mkdir(exist_ok=True, parents=True)

    # check if completed
    if not do_if_complete:
        done_val = kv_read(COMPLETED, store_dir, index)
        if done_val:
            print(f"#{index} is already completed. (use --worker.do_if_complete to override)")
            return False

    # set alarm for job timeout
    signal.alarm(job_timeout)

    # We'll store logs in worker_logs but top-level arrays in store_dir
    run_logdir = worker_logs
    to_log = routine(index=index, logdir=str(run_logdir), store_dir=store_dir)

    # ensure user code doesn't try to override COMPLETED
    from .spec import COMPLETED as COMPLETED_KEY
    assert COMPLETED_KEY not in to_log, f"No '{COMPLETED_KEY}' key allowed in returned data"

    # store returned data in store_dir
    for k, v in to_log.items():
        kv_log(k, v, store_dir=store_dir, index=index)

    # mark completed
    kv_log(COMPLETED_KEY, 1, store_dir=store_dir, index=index)

    # log cmd line
    with open(logdir_path / 'cmds.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    return True

################################################################################
# 5) The main entry point
################################################################################
@param('worker.main_import')
def main(main_import):
    """
    1) import a module that has main(index,logdir,store_dir)
    2) parse config
    3) run do_index(routine=themodule.main)
    """
    module = importlib.import_module(main_import)
    make_config(quiet=True)

    routine = module.main
    status = do_index(routine=routine)


if __name__ == '__main__':
    make_config(quiet=True)
    main()
