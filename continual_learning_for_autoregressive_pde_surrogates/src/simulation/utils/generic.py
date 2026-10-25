import yaml


def load_yaml(file_path) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
