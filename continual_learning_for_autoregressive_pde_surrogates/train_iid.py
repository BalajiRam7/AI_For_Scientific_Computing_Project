import hydra
from omegaconf import DictConfig

from src.iid.trainers.trainer_factory import create_trainer
from src.common.stream import create_stream
from src.common.models.model_factory import create_model
from src.common.generic import get_device, set_manual_seed
from src.common.logger import create_logger


@hydra.main(
    config_path="configs",
    config_name="train_iid",
    version_base="1.3",
)
def main(cfg: DictConfig):
    set_manual_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = create_logger(cfg)
    stream = create_stream(cfg.stream)
    model = create_model(device, cfg.model, cfg.checkpoint_path)
    trainer = create_trainer(model, logger, device, cfg.iid_trainer)

    # Train on the entire stream in an iid manner
    print("Training on stream")
    trainer.train(stream)

    print("Evaluating on stream")
    stream_eval_logs = trainer.eval(stream)

    # Finalize training process
    trainer.finalize(stream, stream_eval_logs)


if __name__ == "__main__":
    main()
