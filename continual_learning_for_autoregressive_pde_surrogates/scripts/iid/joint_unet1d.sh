EXPERIMENT_NAME=joint
DATA_ROOT=./data
DEVICE=cpu
NUM_WORKERS=0
STREAM_SEED=123,456,789,111,222

# ===============================================> advdiff1d_stream
STREAM=advdiff1d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=unet1d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=50
SEED=66

python train_iid.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  model.model_parameters.emb_dim=$EMB_DIM \
  model.model_parameters.learnable_emb=$LEARNABLE_EMB \
  iid_trainer=joint_trainer \
  iid_trainer.trainer_parameters.epochs=$EPOCHS \
  iid_trainer.trainer_parameters.batch_size=$BATCH_SIZE \
  iid_trainer.trainer_parameters.optimizer_params.lr=$LR \
  iid_trainer.trainer_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED

# ===============================================> burgers1d_stream
STREAM=burgers1d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=unet1d
NUM_EMBS=5
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=50
SEED=66

python train_iid.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  model.model_parameters.emb_dim=$EMB_DIM \
  model.model_parameters.learnable_emb=$LEARNABLE_EMB \
  iid_trainer=joint_trainer \
  iid_trainer.trainer_parameters.epochs=$EPOCHS \
  iid_trainer.trainer_parameters.batch_size=$BATCH_SIZE \
  iid_trainer.trainer_parameters.optimizer_params.lr=$LR \
  iid_trainer.trainer_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED


# ===============================================> mixed_advdiff1d_burgers1d_stream
STREAM=mixed_advdiff1d_burgers1d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=unet1d
NUM_EMBS=10
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=50
SEED=0

python train_iid.py --multirun device=$DEVICE \
  logger.experiment=$EXPERIMENT_NAME \
  model=$MODEL \
  model.model_parameters.num_channels=$NUM_CHANNELS \
  model.model_parameters.initial_step=$CONTEXT_LENGTH \
  model.model_parameters.num_embs=$NUM_EMBS \
  model.model_parameters.emb_dim=$EMB_DIM \
  model.model_parameters.learnable_emb=$LEARNABLE_EMB \
  iid_trainer=joint_trainer \
  iid_trainer.trainer_parameters.epochs=$EPOCHS \
  iid_trainer.trainer_parameters.batch_size=$BATCH_SIZE \
  iid_trainer.trainer_parameters.optimizer_params.lr=$LR \
  iid_trainer.trainer_parameters.num_workers=$NUM_WORKERS \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED