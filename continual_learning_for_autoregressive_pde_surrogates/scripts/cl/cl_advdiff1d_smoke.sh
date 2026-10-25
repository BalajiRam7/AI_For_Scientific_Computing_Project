#!/usr/bin/env bash
set -eo pipefail

DEVICE=cpu
DATA_ROOT=./data
STREAM=advdiff1d_smoke
MODEL=fno1d
SEED=123
EPOCHS=1
BATCH_SIZE=32
CONTEXT_LENGTH=3
ROLLOUT_STEPS=2
NUM_WORKERS=0
OUTPUT_ROOT=outputs/smoke_tests/advdiff1d_v2
PYTHON=.venv/bin/python

for ALGORITHM in naive replay replay_ts gem a_gem ewc; do
  EXTRA_ARGS=()
  case "$ALGORITHM" in
    replay)
      EXTRA_ARGS=(
        cl_algorithm.algorithm_parameters.memory_size=20
        cl_algorithm.algorithm_parameters.memory_seed=$SEED
        cl_algorithm.algorithm_parameters.memory_mode=experience_balanced
        cl_algorithm.algorithm_parameters.replay_batch_size=32
        cl_algorithm.algorithm_parameters.replay_loss_coef=0.5
      )
      ;;
    replay_ts)
      EXTRA_ARGS=(
        cl_algorithm.algorithm_parameters.memory_size=20
        cl_algorithm.algorithm_parameters.memory_seed=$SEED
        cl_algorithm.algorithm_parameters.memory_mode=experience_balanced
        cl_algorithm.algorithm_parameters.replay_batch_size=32
        cl_algorithm.algorithm_parameters.replay_slice_factor=2
        cl_algorithm.algorithm_parameters.replay_loss_coef=1.0
      )
      ;;
    gem)
      EXTRA_ARGS=(
        cl_algorithm.algorithm_parameters.memory_size=20
        cl_algorithm.algorithm_parameters.memory_mode=experience_balanced
        cl_algorithm.algorithm_parameters.memory_batch_size=32
      )
      ;;
    a_gem)
      EXTRA_ARGS=(
        cl_algorithm.algorithm_parameters.memory_size=20
        cl_algorithm.algorithm_parameters.memory_mode=experience_balanced
        cl_algorithm.algorithm_parameters.memory_batch_size=32
      )
      ;;
  esac

  "$PYTHON" train_cl.py \
    device=$DEVICE \
    seed=$SEED \
    logger.experiment=cl_advdiff1d_smoke_${ALGORITHM} \
    logger.outputs_dir=$OUTPUT_ROOT/$ALGORITHM \
    model=$MODEL \
    model.model_parameters.initial_step=$CONTEXT_LENGTH \
    model.model_parameters.num_embs=6 \
    cl_algorithm=$ALGORITHM \
    cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
    cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
    cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
    "${EXTRA_ARGS[@]}" \
    stream=$STREAM \
    stream.dataset.dataset_params.dataset_root=$DATA_ROOT/advdiff1d_stream \
    stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
    stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
    stream.dataset.dataset_params.rollout_steps_test=$ROLLOUT_STEPS \
    checkpoint_path=null
done
