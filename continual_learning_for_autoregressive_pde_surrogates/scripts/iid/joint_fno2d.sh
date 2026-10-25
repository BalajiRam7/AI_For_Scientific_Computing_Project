EXPERIMENT_NAME=joint
DATA_ROOT=./data
DEVICE=cpu
NUM_WORKERS=0
STREAM_SEED=123,456,789,111,222

# ===============================================> advdiff2d_stream
STREAM=advdiff2d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=fno2d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=30
SEED=0

CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"

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
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

# ===============================================> burgers2d_stream
STREAM=burgers2d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=fno2d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=2
CONTEXT_LENGTH=10
ROLLOUT_STEPS=30
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

# ===============================================> navierstokes2d_stream
CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"
STREAM=navierstokes2d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=fno2d
NUM_EMBS=6
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=30
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
  iid_trainer.trainer_parameters.criterion="mse+perceptual" \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH


# ===============================================> mixed_advdiff2d_navierstokes2d_stream
CHECKPOINT_PATH="PRETRAINED_CHECKPOINT_PATH"
STREAM=mixed_advdiff2d_navierstokes2d_stream
STREAM_MODE=di
EPOCHS=20
MODEL=fno2d
NUM_EMBS=10
EMB_DIM=2
LEARNABLE_EMB=false
BATCH_SIZE=128
LR=0.0001
NUM_CHANNELS=1
CONTEXT_LENGTH=10
ROLLOUT_STEPS=30
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
  iid_trainer.trainer_parameters.criterion="mse+perceptual" \
  stream=$STREAM stream.dataset.dataset_params.dataset_root=${DATA_ROOT}/${STREAM} \
  stream.stream_mode=$STREAM_MODE \
  stream.dataset.dataset_params.context_length=$CONTEXT_LENGTH \
  stream.dataset.dataset_params.rollout_steps=$ROLLOUT_STEPS \
  "stream.dataset.dataset_params.aux_features=[exp_id]" \
  stream.seed=$STREAM_SEED seed=$SEED \
  checkpoint_path=$CHECKPOINT_PATH

