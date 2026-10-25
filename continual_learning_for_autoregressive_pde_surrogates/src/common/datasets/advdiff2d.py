from typing import Tuple
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from src.common.normalizer import Normalizer
from src.common.generic import load_yaml_file


class AdvDiff2D(Dataset):
    def __init__(
        self,
        dataset_root: str,
        exp_id: int = None,
        num_snapshots_per_sim: int = 100,
        context_length: int = 10,
        rollout_steps: int = 30,
        file_list: list = None,
        aux_features: list = None,
    ):
        self.name = "advdiff2d"
        self.dataset_root = dataset_root
        self.exp_id = exp_id
        self.num_snapshots_per_sim = num_snapshots_per_sim
        self.context_length = context_length
        self.rollout_steps = rollout_steps
        self.aux_features = aux_features

        # Normaliaation
        norm_stat_path = os.path.join(self.dataset_root, "normalization.yaml")
        norm_stat_dict = load_yaml_file(norm_stat_path)
        self.normalizer = Normalizer(norm_stat_dict["normalization_stats"])

        # Samples and parameters
        self.samples = self._load_sample_list(file_list)

    def _load_sample_list(self, file_list: dict = None):
        # If file_list is given use it to load the specific files otherwise load all files
        if file_list:
            sim_paths = [self.dataset_root + f"/simulations/{n}.npz" for n in file_list]
            yaml_paths = [
                self.dataset_root + f"/simulations/{n}.yaml" for n in file_list
            ]
        else:
            sim_paths = glob.glob(self.dataset_root + "/simulations/sim_*.npz")
            yaml_paths = glob.glob(self.dataset_root + "/simulations/sim_*.yaml")
        sample_list = [
            {
                "file_path": sim_paths[i],
                "parameters": load_yaml_file(yaml_paths[i])["parameters"],
            }
            for i in range(len(sim_paths))
        ]
        sample_list = [
            s
            for s in sample_list
            for _ in range(
                self.num_snapshots_per_sim - self.context_length - self.rollout_steps
            )
        ]

        return sample_list

    def __len__(self) -> int:
        return len(self.samples)

    def _get_aux_features(self, idx: int):
        # Input feature parameters
        params = self.samples[idx]["parameters"]
        feat_vector = []
        if "exp_id" in self.aux_features:
            feat_vector = [self.exp_id]
            return torch.LongTensor(feat_vector)

        feat_vector = torch.tensor(feat_vector, dtype=torch.float32)

        return feat_vector

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        snapshot_idx = idx % (
            self.num_snapshots_per_sim - self.context_length - self.rollout_steps
        )
        sim_file_path = self.samples[idx]["file_path"]

        aux_vector = None
        if self.aux_features:
            aux_vector = self._get_aux_features(idx)

        data = np.load(sim_file_path)
        inp_snapshots = data["snapshots"][
            snapshot_idx : snapshot_idx + self.context_length
        ]
        out_snapshots = data["snapshots"][
            snapshot_idx + self.context_length : snapshot_idx
            + self.context_length
            + self.rollout_steps
        ]

        inp_snapshots_normalized = self.normalizer.normalize(inp_snapshots, "u")
        out_snapshots_normalized = self.normalizer.normalize(out_snapshots, "u")

        item_dict = {
            "input": torch.FloatTensor(inp_snapshots_normalized),
            "output": torch.FloatTensor(out_snapshots_normalized),
            "aux_input": aux_vector,
        }

        return item_dict

    def get_full_field_solution(self, idx: int) -> np.ndarray:
        snapshot_idx = idx % (
            self.num_snapshots_per_sim - self.context_length - self.rollout_steps
        )
        sim_file_path = self.samples[idx]["file_path"]
        data = np.load(sim_file_path)
        full_solution = data["snapshots"]
        full_solution_normalized = self.normalizer.normalize(full_solution, "u")

        return {"full_field": full_solution_normalized, "snapshot_idx": snapshot_idx}
