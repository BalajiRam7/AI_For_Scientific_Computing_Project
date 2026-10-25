DEVICE=cpu
NUM_WORKERS=0
EPOCHS=1
BATCH_SIZE=64
DATA_ROOT=./data
CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"

LR=0.0001
STREAM_SEED=123,456,789,111,222
STREAM=advdiff1d_stream
STREAM_MODE=slice_mode_experiments
MODEL=fno1d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=64
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=50
SEED=123
MEMORY_SEED=0


# =================================> MEMSIZE:5

EXPERIMENT_NAME=exp_replayts_slicemode

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=50 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=10 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=5.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  cl_algorithm.algorithm_parameters.replay_slice_mode=equidistance,random,lasts,firsts\
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=125 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=25 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=12.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  cl_algorithm.algorithm_parameters.replay_slice_mode=equidistance,random,lasts,firsts\
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=10 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=2 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=1.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  cl_algorithm.algorithm_parameters.replay_slice_mode=equidistance,random,lasts,firsts\
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH


# =================================> MEMSIZE:20
EXPERIMENT_NAME=exp_replayimp_slicefactor_expbalanced_mem20

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=200 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=10 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=5.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=500 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=25 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=12.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=40 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=2 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=1.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=20 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=1 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=0.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# =================================> MEMSIZE:10
EXPERIMENT_NAME=exp_replayimp_slicefactor_expbalanced_mem10

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=100 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=10 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=5.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=250 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=25 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=12.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

python train_cl.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  cl_algorithm=replay_imp_dev \
  cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
  cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
  cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=20 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=2 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=1.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

python train_cl.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  cl_algorithm=replay_imp_dev \
  cl_algorithm.algorithm_parameters.epochs=$EPOCHS \
  cl_algorithm.algorithm_parameters.batch_size=$BATCH_SIZE \
  cl_algorithm.algorithm_parameters.optimizer_params.lr=$LR \
  cl_algorithm.algorithm_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=10 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=1 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=0.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# =================================> MEMSIZE:200
EXPERIMENT_NAME=exp_replayimp_slicefactor_expbalanced_mem200

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=2000 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=10 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=5.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=5000 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=25 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=12.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=400 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=2 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=1.0 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

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
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  cl_algorithm.algorithm_parameters.memory_size=200 \
  cl_algorithm.algorithm_parameters.replay_slice_factor=1 \
  cl_algorithm.algorithm_parameters.replay_loss_coef=0.5 \
  cl_algorithm.algorithm_parameters.memory_seed=$MEMORY_SEED \
  cl_algorithm.algorithm_parameters.memory_mode="experience_balanced" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH
