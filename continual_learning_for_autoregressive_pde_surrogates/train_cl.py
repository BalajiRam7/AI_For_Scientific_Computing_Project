import hydra
from omegaconf import DictConfig

from src.cl.algorithms import create_algorithm
from src.common.stream import create_stream
from src.common.models.model_factory import create_model
from src.common.generic import get_device, set_manual_seed
from src.common.logger import create_logger


@hydra.main(
    config_path="configs",
    config_name="train_cl",
    version_base="1.3",
)
def main(cfg: DictConfig):
    set_manual_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = create_logger(cfg)
    stream = create_stream(cfg.stream)
    model = create_model(device, cfg.model, cfg.checkpoint_path)
    cl_algorithm = create_algorithm(model, logger, device, cfg.cl_algorithm)

    # Train on sequence of experiences
    all_eval_logs = []
    for experience in stream:
        print(f"Training on {experience.name}")
        cl_algorithm.train_exp(experience)

        print("Evaluating on stream")
        stream_eval_logs = cl_algorithm.eval(stream)
        all_eval_logs.append(stream_eval_logs)
        cl_algorithm.save_checkpoint_after_exp(experience)

    # Finalize training process
    cl_algorithm.finalize(stream, all_eval_logs)


if __name__ == "__main__":
    main()
