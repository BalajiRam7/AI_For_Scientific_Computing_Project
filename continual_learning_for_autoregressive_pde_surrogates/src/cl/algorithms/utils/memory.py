from __future__ import annotations

from typing import Dict, List, Tuple

import random
import torch
from torch.utils.data import DataLoader


class Memory:
    """
    Shared replay buffer supporting two modes:
    - 'reservoir': classic reservoir sampling over all seen samples
    - 'experience_balanced': per-experience balanced buffers with equal quotas

    Stores full targets; algorithms can override sampling to transform them
    (e.g., ReplayImproved slices targets after sampling).
    """

    def __init__(
        self,
        memory_size: int,
        device: torch.device = torch.device("cpu"),
        memory_seed: int = 1234,
        mode: str = "reservoir",
    ) -> None:
        assert mode in {"reservoir", "experience_balanced"}
        self.memory_size = int(memory_size)
        self.device = device
        self.mode = mode

        # Global RNG and counters for reservoir
        self._rng = random.Random(memory_seed)
        self._num_seen_examples = 0

        # Storage
        # Reservoir: list of (x, y, aux, exp_id)
        self._memory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

        # Experience-balanced: per-experience buffers
        self._per_exp: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

    def __len__(self) -> int:
        if self.mode == "reservoir":
            return len(self._memory)
        total = 0
        for buf in self._per_exp.values():
            total += len(buf)
        return total

    @torch.no_grad()
    def add_samples_from_experience(self, experience, batch_size: int = 32) -> None:
        loader = DataLoader(
            experience.train_dataset, batch_size=batch_size, shuffle=True
        )
        exp_id = int(experience.exp_idx)
        for batch in loader:
            x_b, y_b = batch["input"].detach().cpu(), batch["output"].detach().cpu()
            aux_b = (
                batch["aux_input"].detach().cpu()
                if batch.get("aux_input") is not None
                else torch.empty(0)
            )
            if self.mode == "reservoir":
                self._reservoir_add(x_b, y_b, aux_b, exp_id)
            else:
                self._balanced_add(x_b, y_b, aux_b, exp_id)

    def sample_from_memory(
        self, num_samples: int, return_exp_ids: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        def _pack(xs, ys, auxs, labels=None):
            out = (xs.to(self.device), ys.to(self.device), auxs.to(self.device))
            if return_exp_ids:
                if labels is None:
                    labels = torch.empty(0, dtype=torch.long, device=self.device)
                else:
                    labels = labels.to(device=self.device, dtype=torch.long)
                return out + (labels,)
            return out

        empty = (
            torch.empty(0, device=self.device),
            torch.empty(0, device=self.device),
            torch.empty(0, device=self.device),
        )
        if len(self) == 0 or num_samples <= 0:
            return _pack(*empty)

        if self.mode == "reservoir":
            n = min(num_samples, len(self._memory))
            idxs = [self._rng.randrange(len(self._memory)) for _ in range(n)]
            xs = torch.stack([self._memory[i][0] for i in idxs])
            ys = torch.stack([self._memory[i][1] for i in idxs])
            auxs = torch.stack([self._memory[i][2] for i in idxs])
            labels = torch.tensor(
                [self._memory[i][3] for i in idxs], dtype=torch.long
            )
            return _pack(xs, ys, auxs, labels)

        exp_ids = list(self._per_exp.keys())
        E = len(exp_ids)
        if E == 0:
            return _pack(*empty)
        base = max(1, num_samples // E)
        remainder = max(0, num_samples - base * E)

        alloc: Dict[int, int] = {e: base for e in exp_ids}
        for e in exp_ids[:remainder]:
            alloc[e] += 1

        chosen: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        chosen_ids: List[int] = []
        deficit = 0
        for e in exp_ids:
            buf = self._per_exp[e]
            k = min(alloc[e], len(buf))
            if k < alloc[e]:
                deficit += alloc[e] - k
            if k > 0:
                idxs = [self._rng.randrange(len(buf)) for _ in range(k)]
                chosen.extend([buf[i] for i in idxs])
                chosen_ids.extend([e] * k)

        if deficit > 0:
            donors = [e for e in exp_ids if len(self._per_exp[e]) > alloc[e]]
            for _ in range(deficit):
                if not donors:
                    break
                e = donors[self._rng.randrange(len(donors))]
                buf = self._per_exp[e]
                chosen.append(buf[self._rng.randrange(len(buf))])
                chosen_ids.append(e)

        xs = torch.stack([t[0] for t in chosen])
        ys = torch.stack([t[1] for t in chosen])
        auxs = torch.stack([t[2] for t in chosen])
        labels = torch.tensor(chosen_ids, dtype=torch.long)
        return _pack(xs, ys, auxs, labels)

    # -------------------- internal helpers --------------------
    def _reservoir_add(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        aux_batch: torch.Tensor,
        exp_id: int,
    ) -> None:
        bsz = x_batch.shape[0]
        for i in range(bsz):
            self._num_seen_examples += 1
            item = (x_batch[i], y_batch[i], aux_batch[i], exp_id)
            if len(self._memory) < self.memory_size:
                self._memory.append(item)
            else:
                j = self._rng.randint(0, self._num_seen_examples - 1)
                if j < self.memory_size:
                    self._memory[j] = item

    def _balanced_add(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        aux_batch: torch.Tensor,
        exp_id: int,
    ) -> None:
        if exp_id not in self._per_exp:
            self._per_exp[exp_id] = []
            # When a new experience appears, rebalance by trimming if needed
            self._rebalance_capacities()

        buf = self._per_exp[exp_id]
        cap_map = self._capacity_map()
        cap = cap_map[exp_id]
        bsz = x_batch.shape[0]
        for i in range(bsz):
            item = (x_batch[i], y_batch[i], aux_batch[i])
            if len(buf) < cap:
                buf.append(item)
            else:
                # per-experience reservoir
                j = self._rng.randrange(cap)
                buf[j] = item

    def _capacity_map(self) -> Dict[int, int]:
        exps = sorted(self._per_exp.keys())
        if not exps:
            return {}
        E = len(exps)
        base = self.memory_size // E
        rem = self.memory_size - base * E
        cap = {e: base for e in exps}
        for e in exps[:rem]:
            cap[e] += 1
        return cap

    def _rebalance_capacities(self) -> None:
        cap = self._capacity_map()
        # Trim buffers to current capacities
        for e, buf in list(self._per_exp.items()):
            c = cap.get(e, 0)
            if c <= 0:
                del self._per_exp[e]
                continue
            if len(buf) > c:
                idxs = list(range(len(buf)))
                self._rng.shuffle(idxs)
                keep = set(idxs[:c])
                self._per_exp[e] = [buf[i] for i in range(len(buf)) if i in keep]
