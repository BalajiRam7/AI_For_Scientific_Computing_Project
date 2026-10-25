from typing import List, Dict, Any, final
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from functools import partial
import glob

from src.common.datasets.dataset_factory import create_dataset
from src.common.generic import load_yaml_file


class Experience:
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        exp_idx: int,
        **kwargs,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.exp_idx = exp_idx
        for k, v in kwargs.items():
            setattr(self, k, v)


def condition_checker(param_name, operator, target_value, sample_params):
    param_name_levels = param_name.split(".")

    if len(param_name_levels) == 1:
        value = sample_params[param_name]
    elif len(param_name_levels) == 2:
        value = sample_params[param_name_levels[0]][param_name_levels[1]]
    else:
        raise NotImplementedError()

    if operator == "equals":
        return value == target_value
    else:
        raise NotImplementedError()


def get_sample_indices_by_condition(
    dataset_files: dict, conditions: Dict[str, Any]
) -> List[int]:
    condition_checkers = []
    for k in conditions.keys():
        param_name = conditions[k][0]
        operator = conditions[k][1]
        target_value = conditions[k][2]
        condition_checkers.append(
            partial(
                condition_checker,
                param_name=param_name,
                operator=operator,
                target_value=target_value,
            )
        )

    # Matching indices
    selected_files = []
    for file in dataset_files.items():
        simulation_params = file[1]["parameters"]
        cond_results = [f(sample_params=simulation_params) for f in condition_checkers]

        # if all conditons are met add sample
        if all(cond_results):
            selected_files.append(file)

    return selected_files


class Stream:
    def __init__(
        self,
        dataset_cfg,
        experience_list,
        stream_mode,
        train_test_split_ratio,
        train_val_split_ratio,
        seed,
    ):
        print(f"\nCreating stream with mode {stream_mode} ...")
        self.dataset_cfg = dataset_cfg
        self.experience_list = experience_list[stream_mode]
        self.train_test_split_ratio = train_test_split_ratio
        self.train_val_split_ratio = train_val_split_ratio
        self.seed = seed

        self.experiences = self._create_experience_list()
        print(f"Stream created. Number of experiences: {len(self.experiences)}\n")

    def _eval_rollout_steps(self):
        rollout_steps_test = self.dataset_cfg.dataset_params.get(
            "rollout_steps_test"
        )
        if rollout_steps_test is not None:
            return rollout_steps_test
        return self.dataset_cfg.dataset_params.rollout_steps

    def _create_experience_list(self) -> List[Experience]:
        yaml_files = glob.glob(
            self.dataset_cfg["dataset_params"]["dataset_root"] + "/simulations/*.yaml"
        )
        dataset_files = {
            f.split("/")[-1].split(".")[0]: load_yaml_file(f) for f in yaml_files
        }

        experiences = []
        for exp_idx, exp_conditions in enumerate(self.experience_list):
            files_exp_i = get_sample_indices_by_condition(dataset_files, exp_conditions)
            files_exp_i = [f[0] for f in files_exp_i]

            # Create train/test/splits
            train_files, test_files = train_test_split(
                files_exp_i,
                train_size=self.train_test_split_ratio,
                random_state=self.seed,
            )

            # If train_val_split_ratio is given create also a validation set
            self.dataset_cfg.dataset_params.exp_id = exp_idx
            if self.train_val_split_ratio:
                train_files, val_files = train_test_split(
                    train_files,
                    train_size=self.train_val_split_ratio,
                    random_state=self.seed,
                )
                train_dataset = create_dataset(self.dataset_cfg, file_list=train_files)
                val_dataset = create_dataset(
                    self.dataset_cfg,
                    file_list=val_files,
                    rollout_steps=self._eval_rollout_steps(),
                )
                test_dataset = create_dataset(
                    self.dataset_cfg,
                    file_list=test_files,
                    rollout_steps=self._eval_rollout_steps(),
                )
            else:
                train_dataset = create_dataset(self.dataset_cfg, file_list=train_files)
                val_dataset = None
                test_dataset = create_dataset(
                    self.dataset_cfg,
                    file_list=test_files,
                    rollout_steps=self._eval_rollout_steps(),
                )

            experiences.append(
                Experience(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    exp_idx=exp_idx,
                    name=f"exp_{exp_idx}",
                )
            )
        return experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, item):
        return self.experiences[item]


def create_stream(
    stream_cfg: DictConfig,
) -> Stream:
    if stream_cfg.stream_name.startswith("mixed_"):
        stream_cfg.stream_1.dataset.dataset_params.context_length = (
            stream_cfg.dataset.dataset_params.context_length
        )
        stream_cfg.stream_1.dataset.dataset_params.rollout_steps = (
            stream_cfg.dataset.dataset_params.rollout_steps
        )
        stream_cfg.stream_1.dataset.dataset_params.rollout_steps_test = (
            stream_cfg.dataset.dataset_params.get("rollout_steps_test")
        )
        stream_cfg.stream_1.dataset.dataset_params.aux_features = (
            stream_cfg.dataset.dataset_params.aux_features
        )
        stream_cfg.stream_1.dataset.dataset_params.dataset_root = (
            stream_cfg.dataset.dataset_params.dataset_root.rsplit("/", 1)[0]
            + "/"
            + stream_cfg.stream_1.stream_name
        )
        stream_1 = Stream(
            stream_cfg.stream_1.dataset,
            stream_cfg.stream_1.experience_list,
            stream_cfg.stream_mode,
            stream_cfg.train_test_split_ratio,
            stream_cfg.train_val_split_ratio,
            stream_cfg.seed,
        )

        # TODO: modified datset_cfg is not applied correctly
        stream_cfg.stream_2.dataset.dataset_params.context_length = (
            stream_cfg.dataset.dataset_params.context_length
        )
        stream_cfg.stream_2.dataset.dataset_params.rollout_steps = (
            stream_cfg.dataset.dataset_params.rollout_steps
        )
        stream_cfg.stream_2.dataset.dataset_params.rollout_steps_test = (
            stream_cfg.dataset.dataset_params.get("rollout_steps_test")
        )
        stream_cfg.stream_2.dataset.dataset_params.aux_features = (
            stream_cfg.dataset.dataset_params.aux_features
        )
        stream_cfg.stream_2.dataset.dataset_params.dataset_root = (
            stream_cfg.dataset.dataset_params.dataset_root.rsplit("/", 1)[0]
            + "/"
            + stream_cfg.stream_2.stream_name
        )
        stream_2 = Stream(
            stream_cfg.stream_2.dataset,
            stream_cfg.stream_2.experience_list,
            stream_cfg.stream_mode,
            stream_cfg.train_test_split_ratio,
            stream_cfg.train_val_split_ratio,
            stream_cfg.seed,
        )

        # Create mixed experience list
        stream = stream_1
        final_exp_list = []
        for i, (stream_num, exp_idx) in enumerate(stream_cfg.final_stream_order):
            if stream_num == "stream_1":
                exp = stream_1.experiences[exp_idx]
                exp.exp_idx = i
                exp.name = f"exp_{i}[stream_1]"
            elif stream_num == "stream_2":
                exp = stream_2.experiences[exp_idx]
                exp.exp_idx = i
                exp.name = f"exp_{i}[stream_2]"

            final_exp_list += [exp]

        stream.experiences = final_exp_list
        print(f"Mixed streams with {len(stream)} experiences.\n")

    else:
        stream = Stream(
            stream_cfg.dataset,
            stream_cfg.experience_list,
            stream_cfg.stream_mode,
            stream_cfg.train_test_split_ratio,
            stream_cfg.train_val_split_ratio,
            stream_cfg.seed,
        )

    return stream
