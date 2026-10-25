import hydra
from omegaconf import DictConfig

from src.simulation.data_generation.dataset_generator import DatasetGenerator


@hydra.main(
    config_path="configs",
    config_name="simulate",
    version_base="1.3.2",
)
def generate_dataset(cfg: DictConfig):
    data_generator = DatasetGenerator(cfg)
    data_generator.create_dataset()


if __name__ == "__main__":
    generate_dataset()
