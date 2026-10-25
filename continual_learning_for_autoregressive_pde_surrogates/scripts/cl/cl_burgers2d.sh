EXPERIMENT_NAME=cl_burgers2d
LR=0.0001
DATA_ROOT=./data
DEVICE=cpu
NUM_WORKERS=0
STREAM_SEED=123,456,789,111,222

# ============================================================================================================
#                                                burgers2d
# ============================================================================================================
STREAM=burgers2d_stream
STREAM_MODE=di
EPOCHS=10
MODEL=fno2d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=64
LR=0.0001
NUM_CHANNELS=2
CONTEXT_LENGTH=10
ROLLOUT_STEPS=30

# fno2d
CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"
SEED=0

# unet2d
# CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"
# SEED=6 

# ----- Naive
python train_cl.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  cl_algorithm=naive \
  cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
  cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
  cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# ----- Replay-TS
python train_cl.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  cl_algorithm=replay_ts \
  cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
  cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
  cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
  cl_algorithm.algorithm_parameters.memory_size=200 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=10 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=5.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# ----- Replay
python train_cl.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  cl_algorithm=replay \
  cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
  cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
  cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
  cl_algorithm.algorithm_parameters.memory_size=20 \
  cl_algorithm.algorithm_parameters.memory_seed=$SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  cl_algorithm.algorithm_parameters.replay_batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.replay_loss_coef=0.5 \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# # ----- GEM
# python train_cl.py --multirun device=$DEVICE \
#   logger.experiment=$EXPERIMENT_NAME \
#   model=$MODEL \
#   model.model_parameters.num_channels=$NUM_CHANNELS \
#   model.model_parameters.initial_step=$CONTEXT_LENGTH \
#   model.model_parameters.num_embs=$NUM_EMBS \
#   cl_algorithm=gem \
#   cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
#   cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
#   cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
#   cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
#   cl_algorithm.algorithm_parameters.memory_size=20 \
#   cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
#   cl_algorithm.algorithm_parameters.memory_batch_size=64 \
#   cl_algorithm.algorithm_parameters.eps_gem=0.000001 \
#   cl_algorithm.algorithm_parameters.gamma=0.9 \
#   stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
#   stream.stream_mode=$STREAM_MODE \
#   stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
#   "stream.dataset.dataset_params.aux_features=[exp_id]" \
#   stream.seed=$STREAM_SEED seed=$SEED
#
# # ----- A-GEM
# python train_cl.py --multirun device=$DEVICE \
#   logger.experiment=$EXPERIMENT_NAME \
#   model=$MODEL \
#   model.model_parameters.num_channels=$NUM_CHANNELS \
#   model.model_parameters.initial_step=$CONTEXT_LENGTH \
#   model.model_parameters.num_embs=$NUM_EMBS \
#   cl_algorithm=a_gem \
#   cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
#   cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
#   cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
#   cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
#   cl_algorithm.algorithm_parameters.memory_size=20 \
#   cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
#   cl_algorithm.algorithm_parameters.memory_batch_size=64 \
#   cl_algorithm.algorithm_parameters.eps_agem=0.000001 \
#   stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
#   stream.stream_mode=$STREAM_MODE \
#   stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
#   "stream.dataset.dataset_params.aux_features=[exp_id]" \
#   stream.seed=$STREAM_SEED seed=$SEED
#
# # ----- EWC
# python train_cl.py --multirun device=$DEVICE \
#   logger.experiment=$EXPERIMENT_NAME \
#   model=$MODEL \
#   model.model_parameters.num_channels=$NUM_CHANNELS \
#   model.model_parameters.initial_step=$CONTEXT_LENGTH \
#   model.model_parameters.num_embs=$NUM_EMBS \
#   cl_algorithm=ewc \
#   cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
#   cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
#   cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
#   cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
#   cl_algorithm.algorithm_parameters.ewc_lambda=50.0 \
#   stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
#   stream.stream_mode=$STREAM_MODE \
#   stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
#   "stream.dataset.dataset_params.aux_features=[exp_id]" \
#   stream.seed=$STREAM_SEED seed=$SEED
#
# # ----- Online-EWC
# python train_cl.py --multirun device=$DEVICE \
#   logger.experiment=$EXPERIMENT_NAME \
#   model=$MODEL \
#   model.model_parameters.num_channels=$NUM_CHANNELS \
#   model.model_parameters.initial_step=$CONTEXT_LENGTH \
#   model.model_parameters.num_embs=$NUM_EMBS \
#   cl_algorithm=online_ewc \
#   cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
#   cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
#   cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
#   cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
#   cl_algorithm.algorithm_parameters.ewc_lambda=50.0 \
#   cl_algorithm.algorithm_parameters.gamma=0.99 \
#   stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
#   stream.stream_mode=$STREAM_MODE \
#   stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
#   "stream.dataset.dataset_params.aux_features=[exp_id]" \
#   stream.seed=$STREAM_SEED seed=$SEED
