export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NUM_PARALLEL_SIMS=5
SAVE_ALL_SNAPSHOTS=True

python generate_dataset.py save_all_snapshots=$SAVE_ALL_SNAPSHOTS num_parallel_sims=$NUM_PARALLEL_SIMS \
  equation=advdiff2d dataset_save_path=data/advdiff2d_stream

