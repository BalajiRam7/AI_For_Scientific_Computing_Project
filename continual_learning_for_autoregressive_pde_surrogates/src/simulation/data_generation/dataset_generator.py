from itertools import product
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import os
import numpy as np
import yaml
import copy
import tempfile
import shutil
from omegaconf import ListConfig, DictConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from src.simulation.equations.equation_factory import create_equation


class DatasetGenerator:
    def __init__(self, config: DictConfig):
        self.config = config
        self._literal_list_keys = {"location"}
        self._literal_list_of_dicts_keys: set[str] = set()
        self.uniform_expand: bool = bool(getattr(config, "uniform_expand", False))
        self.uniform_seed = getattr(config, "uniform_seed", None)
        self._rng = np.random.default_rng(self.uniform_seed)
        self.num_parallel_sims = int(getattr(config, "num_parallel_sims", 10))

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds_int = int(max(0, round(seconds)))
        hours, rem = divmod(seconds_int, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _flatten_dict(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, (dict, DictConfig)):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _unflatten_dict(self, d, sep="."):
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_sub = result
            for part in parts[:-1]:
                if part not in d_sub:
                    d_sub[part] = {}
                d_sub = d_sub[part]
            d_sub[parts[-1]] = value
        return result

    def _to_plain_python(self, data):
        if isinstance(data, (dict, DictConfig)):
            return {k: self._to_plain_python(v) for k, v in data.items()}
        if isinstance(data, (list, ListConfig)):
            return [self._to_plain_python(i) for i in data]
        if isinstance(data, np.generic):
            return data.item()
        return data

    @dataclass(frozen=True)
    class UniformSpec:
        low: float
        high: float
        num: int

    @dataclass(frozen=True)
    class UniformDraw:
        low: float
        high: float
        num: int
        index: int

    def _is_function_str(self, value: Any, name: str) -> bool:
        if not isinstance(value, str):
            return False
        s = value.strip().lower().replace(" ", "")
        return s.startswith(name.lower() + "(") and s.endswith(")")

    def _parse_args_3(self, value: str) -> tuple[float, float, int]:
        inside = value.strip()
        lpar = inside.find("(")
        inside = inside[lpar + 1 : -1]
        parts = [p.strip() for p in inside.split(",")]
        if len(parts) != 3:
            raise ValueError("Function must have 3 arguments: start, stop, num")
        start = float(parts[0])
        stop = float(parts[1])
        num = int(float(parts[2]))
        return start, stop, num

    def _parse_range(self, value: str) -> np.ndarray:
        if not self._is_function_str(value, "range"):
            return value
        try:
            start, stop, num = self._parse_args_3(value)
            return np.linspace(start, stop, num)
        except Exception as e:
            raise ValueError(f"Invalid range format: {value}. Error: {e}")

    def _parse_uniform(self, value: str) -> np.ndarray:
        if not self._is_function_str(value, "uniform"):
            return value
        try:
            low, high, num = self._parse_args_3(value)
            if self.uniform_expand:
                return self._rng.uniform(low=low, high=high, size=num)
            return DatasetGenerator.UniformSpec(low=low, high=high, num=num)
        except Exception as e:
            raise ValueError(f"Invalid uniform format: {value}. Error: {e}")

    def _recompose_inline_function_tokens(
        self, mapping: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        items = list(mapping.items())
        i = 0
        new_items: List[tuple[Any, Any]] = []
        while i < len(items):
            key_i, val_i = items[i]
            if isinstance(val_i, str):
                token = val_i.strip()
                starts_func = token.lower().startswith(
                    "range("
                ) or token.lower().startswith("uniform(")
                is_complete = token.endswith(")")
                if starts_func and not is_complete:
                    tokens = [token]
                    j = i + 1
                    while j < len(items):
                        key_j, val_j = items[j]
                        key_tok = str(key_j).strip()
                        tokens.append(key_tok)
                        if isinstance(val_j, (str, int, float)) and val_j not in (
                            None,
                            "",
                        ):
                            val_tok = str(val_j).strip()
                            tokens.append(val_tok)
                        j += 1
                        if tokens[-1].endswith(")") or (
                            len(tokens) >= 2 and tokens[-2].endswith(")")
                        ):
                            break
                    combined = ", ".join(tokens)
                    new_items.append((key_i, combined))
                    i = j
                    continue
            new_items.append((key_i, val_i))
            i += 1

        fixed: Dict[Any, Any] = {}
        for k, v in new_items:
            fixed[k] = v
        return fixed

    def _sanitize_config_structure(self, data: Any) -> Any:
        if isinstance(data, (dict, DictConfig)):
            raw = dict(data)
            raw = self._recompose_inline_function_tokens(raw)
            return {k: self._sanitize_config_structure(v) for k, v in raw.items()}
        if isinstance(data, (list, ListConfig)):
            return [self._sanitize_config_structure(v) for v in list(data)]
        return data

    def _validate_function_strings(self, data: Any, path: Tuple[Any, ...] = ()) -> None:
        if isinstance(data, str):
            s = data.strip().lower()
            if (s.startswith("range(") or s.startswith("uniform(")) and not s.endswith(
                ")"
            ):
                dotted = ".".join(map(str, path)) if path else "<root>"
                raise ValueError(
                    f"Malformed function value at '{dotted}': '{data}'. "
                    'YAML likely split the unquoted function. Please quote it (e.g., "Range(0.1, 0.2, 3)") '
                    "or use block mapping format instead of inline braces."
                )
            s_compact = s.replace(" ", "")
            if s_compact.endswith(")") and "(" in s_compact:
                name = s_compact.split("(", 1)[0]
                if any(c.isalpha() for c in name) and name not in ("range", "uniform"):
                    dotted = ".".join(map(str, path)) if path else "<root>"
                    raise ValueError(
                        f"Unknown function '{name}' at '{dotted}': '{data}'. Supported: Range, Uniform"
                    )
        elif isinstance(data, (dict, DictConfig)):
            for k, v in dict(data).items():
                self._validate_function_strings(v, path + (k,))
        elif isinstance(data, (list, ListConfig)):
            for idx, v in enumerate(list(data)):
                self._validate_function_strings(v, path + (idx,))

    def _expand_value(self, key: str, value: Any) -> List[Any]:
        if isinstance(value, str):
            r = self._parse_range(value)
            if isinstance(r, np.ndarray):
                return list(r)
            u = self._parse_uniform(value)
            if isinstance(u, DatasetGenerator.UniformSpec):
                return [
                    DatasetGenerator.UniformDraw(u.low, u.high, u.num, i)
                    for i in range(u.num)
                ]
            if isinstance(u, np.ndarray):
                return list(u)
            return [value]

        if isinstance(value, (dict, DictConfig)):
            return self._expand_dict(value)

        if isinstance(value, (list, ListConfig)):
            if len(value) == 0:
                return [list(value)]

            if all(isinstance(elem, (dict, DictConfig)) for elem in value):
                expanded: List[Any] = []
                for elem in value:
                    expanded.extend(self._expand_dict(elem))
                return expanded

            expanded_elems: List[Any] = []
            for elem in value:
                if isinstance(elem, str):
                    r = self._parse_range(elem)
                    if isinstance(r, np.ndarray):
                        expanded_elems.extend(list(r))
                        continue
                    u = self._parse_uniform(elem)
                    if isinstance(u, DatasetGenerator.UniformSpec):
                        expanded_elems.extend(
                            [
                                DatasetGenerator.UniformDraw(u.low, u.high, u.num, i)
                                for i in range(u.num)
                            ]
                        )
                        continue
                expanded_elems.append(elem)

            if key in self._literal_list_keys:
                return [self._to_plain_python(expanded_elems)]

            if all(
                isinstance(
                    elem, (int, float, np.number, str, DatasetGenerator.UniformDraw)
                )
                for elem in expanded_elems
            ):
                return list(expanded_elems)

            return [self._to_plain_python(expanded_elems)]

        return [value]

    def _expand_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        expanded_per_key: Dict[str, List[Any]] = {}
        for sub_key, sub_value in data.items():
            expanded_per_key[sub_key] = self._expand_value(sub_key, sub_value)

        keys: List[str] = list(expanded_per_key.keys())
        value_lists: List[List[Any]] = [expanded_per_key[k] for k in keys]

        expanded_dicts: List[Dict[str, Any]] = []
        for combo in product(*value_lists):
            new_item: Dict[str, Any] = {}
            for k, v in zip(keys, combo):
                new_item[k] = v
            expanded_dicts.append(new_item)

        return expanded_dicts

    def _create_param_grid(self, params: DictConfig) -> List[DictConfig]:
        sanitized = self._sanitize_config_structure(params)
        self._validate_function_strings(sanitized)
        return [self._unflatten_dict(p) for p in self._expand_dict(sanitized)]

    def _collect_uniform_specs(
        self, data: Any, base_path: Tuple[Any, ...] = ()
    ) -> List[Tuple[Tuple[Any, ...], "DatasetGenerator.UniformSpec"]]:
        specs: List[Tuple[Tuple[Any, ...], DatasetGenerator.UniformSpec]] = []
        if isinstance(data, DatasetGenerator.UniformSpec):
            specs.append((base_path, data))
        elif isinstance(data, (dict, DictConfig)):
            for k, v in dict(data).items():
                specs.extend(self._collect_uniform_specs(v, base_path + (k,)))
        elif isinstance(data, (list, ListConfig)):
            for idx, v in enumerate(list(data)):
                specs.extend(self._collect_uniform_specs(v, base_path + (idx,)))
        return specs

    def _set_value_at_path(self, obj: Any, path: Tuple[Any, ...], value: Any) -> None:
        current = obj
        for key in path[:-1]:
            current = current[key]
        last = path[-1]
        current[last] = value

    def _materialize_uniform_placeholders(self, data: Any) -> Any:
        if isinstance(data, DatasetGenerator.UniformDraw):
            return float(self._rng.uniform(low=data.low, high=data.high))
        if isinstance(data, (dict, DictConfig)):
            return {
                k: self._materialize_uniform_placeholders(v)
                for k, v in dict(data).items()
            }
        if isinstance(data, (list, ListConfig)):
            return [self._materialize_uniform_placeholders(v) for v in list(data)]
        return data

    def _materialize_uniform_specs_once(
        self, base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        specs = self._collect_uniform_specs(base_params)
        if not specs:
            return base_params

        params_copy = copy.deepcopy(base_params)
        for path, spec in specs:
            sample = float(self._rng.uniform(low=spec.low, high=spec.high))
            self._set_value_at_path(params_copy, path, sample)
        return params_copy

    def _run_single_simulation(self, i, base_params):
        cache_dir = tempfile.mkdtemp(prefix=f"fenics_cache_worker_{i}_")
        os.environ["XDG_CACHE_HOME"] = cache_dir

        try:
            if self.uniform_seed is not None:
                worker_seed = self.uniform_seed + i
            else:
                worker_seed = i + os.getpid()

            self._rng = np.random.default_rng(worker_seed)

            params = self._materialize_uniform_placeholders(base_params)

            equation = create_equation(self.config.equation.equation_name, params)
            snapshots = equation.solve()

            if snapshots is None or not isinstance(snapshots, np.ndarray):
                raise ValueError(f"Simulation {i} did not produce a snapshots ndarray.")

            save_all_snapshots = bool(getattr(self.config, "save_all_snapshots", True))
            if not save_all_snapshots and snapshots.shape[0] >= 2:
                snapshots_to_save = np.stack([snapshots[0], snapshots[-1]], axis=0)
            else:
                snapshots_to_save = snapshots

            simulations_dir = os.path.join(self.config.dataset_save_path, "simulations")
            os.makedirs(simulations_dir, exist_ok=True)
            base_filename = f"sim_{i}"
            npz_path = os.path.join(simulations_dir, f"{base_filename}.npz")

            np.savez_compressed(npz_path, snapshots=snapshots_to_save)

            plain_params = self._to_plain_python(params)
            sim_yaml = {
                "equation": self.config.equation.equation_name,
                "file_path": os.path.basename(npz_path),
                "parameters": plain_params,
            }
            yaml_path = os.path.join(simulations_dir, f"{base_filename}.yaml")
            with open(yaml_path, "w") as yf:
                yaml.safe_dump(sim_yaml, yf, sort_keys=False)

            return {"file_path": os.path.basename(npz_path), "parameters": plain_params}
        finally:
            shutil.rmtree(cache_dir)

    def create_dataset(self):
        dataset_save_path = self.config.dataset_save_path
        simulations_dir = os.path.join(dataset_save_path, "simulations")
        os.makedirs(simulations_dir, exist_ok=True)

        base_param_grid = self._create_param_grid(
            self.config.equation.equation_parameters
        )
        total_sims = len(base_param_grid)
        sim_metadata = []

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[bold blue]Generating simulations...", total=total_sims
            )

            with ProcessPoolExecutor(max_workers=self.num_parallel_sims) as executor:
                futures = {
                    executor.submit(self._run_single_simulation, i, base): i
                    for i, base in enumerate(base_param_grid)
                }

                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        meta = future.result()
                        sim_metadata.append(meta)
                    except Exception as e:
                        print(f"Simulation {i} failed: {e}")

                    progress.update(task, advance=1)

        sim_metadata.sort(
            key=lambda m: int(
                os.path.basename(m["file_path"]).split("_")[-1].split(".")[0]
            )
        )

        normalization_stats = {}
        flat_param_grid = [
            self._flatten_dict(meta["parameters"]) for meta in sim_metadata
        ]
        all_keys = set(k for p in flat_param_grid for k in p.keys())

        for key in all_keys:
            values = [
                p[key]
                for p in flat_param_grid
                if key in p and isinstance(p[key], (int, float, np.number))
            ]
            if values:
                normalization_stats[f"inputs.{key}"] = {
                    "min": float(min(values)),
                    "max": float(max(values)),
                }

        u_all_min = float("inf")
        u_all_max = float("-inf")
        for meta in sim_metadata:
            try:
                full_path = os.path.join(simulations_dir, meta["file_path"])
                with np.load(full_path) as npz:
                    frames = npz["snapshots"]
                T = frames.shape[0]
                frames_flat = frames.reshape(T, -1)
                u_all_min = min(u_all_min, float(np.min(frames_flat)))
                u_all_max = max(u_all_max, float(np.max(frames_flat)))
            except Exception as e:
                print(
                    f"Warning: Could not read file {meta['file_path']} for normalization. Error: {e}"
                )

        normalization_stats["u"] = {
            "min": (0.0 if u_all_min == float("inf") else u_all_min),
            "max": (0.0 if u_all_max == float("-inf") else u_all_max),
        }

        norm_path = os.path.join(dataset_save_path, "normalization.yaml")
        with open(norm_path, "w") as nf:
            yaml.safe_dump(
                {"normalization_stats": normalization_stats}, nf, sort_keys=False
            )

        print(f"\nData generation complete. All data saved in '{dataset_save_path}'.")
