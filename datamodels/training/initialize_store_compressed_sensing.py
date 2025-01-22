from fastargs.decorators import param
from fastargs import Param, Section

import numpy as np
import json

from .spec import preprocess_spec
from .utils import memmap_path, make_config

###############################
# 1) Sparse sign helper
###############################
def sample_sparse_sign_matrix(m, n, sparsity):
    """
    Sample an (m x n) sparse sign matrix S where each entry is in {+1, 0, -1}.
    Probabilities:
      +1 with prob 1/(2*sparsity)
       0 with prob 1 - 1/sparsity
      -1 with prob 1/(2*sparsity)
    """
    S = np.random.choice([1, 0, -1],
                         size=(m, n),
                         p=[1/(2*sparsity), 1 - 1/sparsity, 1/(2*sparsity)])
    return S

Section('logging').params(
    logdir=Param(str, 'file with main() to run'),
    spec=Param(str, 'file with spec')
)

@param('logging.logdir')
@param('logging.spec')
def main(logdir, spec):
    """
    1) Load the 'spec.json' describing how many models (num_models) and schema.
    2) Create memmapped arrays for each schema entry, shape=(num_models, ...).
    3) Generate a single S, plus 'margins' and 'completed' arrays, if desired.
    """
    # 1) Read and process spec
    assert logdir is not None
    assert spec is not None
    spec_data = json.loads(open(spec, 'r').read())
    spec_data = preprocess_spec(spec_data)

    num_models = spec_data["num_models"]  # e.g. 50
    schema = spec_data["schema"]          # e.g. {"some_key": {"dtype":..., "shape":[...]}}

    # 2) Suppose you want to fix m=some_value, n=50000, etc.
    #    You can read them from spec_data or define them here. Example:
    m = num_models      # For example, if your spec says 50 models, you'll have 50 rows
    n = 50000           # CIFAR-10 training set size
    s = 50000/3             # expected #nonzeros
    sparsity = 3        # => +1, 0, -1 with {1/6, 2/3, 1/6}

    # 3) Sample S once
    S = sample_sparse_sign_matrix(m, n, sparsity)

    # 4) Also define 'margins' array, shape (m, #test_examples?), e.g. (50, 10000)
    #    or a shape (m,) if you only store scalar differences.
    #    We'll assume 10k test set for demonstration.
    margins_shape = (m, 10000)
    margins = np.zeros(margins_shape, dtype=np.float32)

    # 5) 'completed' array of shape (m,)
    completed = np.zeros((m,), dtype=bool)

    # 6) Create the memmap arrays from your spec
    #    (like your original code, which only sets up placeholders).
    for key, metadata in schema.items():
        dtype = getattr(np, metadata['dtype'])
        shape = (num_models,) + tuple(metadata['shape'])
        this_filename = memmap_path(logdir, key)
        mmap = np.lib.format.open_memmap(this_filename, mode='w+', dtype=dtype, shape=shape)
        mmap.flush()

    # 7) Write S, margins, completed to their own memmap files
    #    so each worker can access them
    s_filename = memmap_path(logdir, 'S')  # e.g. log_dir/S.npy
    np.save(s_filename, S)
    margins_filename = memmap_path(logdir, 'margins')
    np.save(margins_filename, margins)
    completed_filename = memmap_path(logdir, 'completed')
    np.save(completed_filename, completed)

    print(f"Initialized store at {logdir}")
    print(f"  S.shape={S.shape}, margins.shape={margins.shape}, completed.shape={completed.shape}")

if __name__ == '__main__':
    make_config()  # from .utils
    main()
    print('Done!')
