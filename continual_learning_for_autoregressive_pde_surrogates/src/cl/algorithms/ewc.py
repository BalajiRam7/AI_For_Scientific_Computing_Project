import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .algorithm import Algorithm
from src.common.stream import Experience
from src.common.logger import Logger


class EWC(Algorithm):
    def __init__(
        self,
        model: nn.Module,
        device: torch.DeviceObjType = torch.device("cpu"),
        num_workers: int = 0,
        logger: Logger = None,
        optimizer: str = "adam",
        optimizer_params: dict = {},
        criterion: str = "mse",
        criterion_params: dict = {},
        batch_size: int = 64,
        epochs: int = 5,
        ewc_lambda: float = 1.0,
    ):
        super().__init__(
            model,
            device,
            logger=logger,
            num_workers=num_workers,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            criterion=criterion,
            criterion_params=criterion_params,
            batch_size=batch_size,
            epochs=epochs,
        )
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices = []
        self.optimal_params_list = []

    def _compute_fisher_matrix(self, dataset: Dataset):
        self.model.eval()
        fisher_matrix = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Use the same data loading settings as training for consistency
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        num_batches = 0

        for batch in tqdm(
            loader, desc="Computing Fisher Matrix", leave=False
        ):  # Added tqdm for visibility
            x = batch["input"].to(self.device)
            y = batch["output"].to(self.device)
            aux_input = batch["aux_input"].to(self.device)
            n_rollout_steps = y.shape[1]

            # Clear any existing gradients
            self.model.zero_grad(set_to_none=True)

            # Accumulate mean-reduced loss over rollouts for Fisher estimation
            # This ensures the gradient magnitude is comparable to the training loss.
            fisher_loss = 0.0
            x_cur = x.clone()
            for i in range(n_rollout_steps):
                pred_i = self.model(x_cur, aux_input=aux_input)

                # *** KEY CHANGE: Use reduction="mean" ***
                # The mean is taken over (B * L * C) elements per time step.
                fisher_loss += nn.functional.mse_loss(
                    pred_i, y[:, i, :].unsqueeze(1), reduction="mean"
                )
                x_cur = torch.cat([x_cur, pred_i], dim=1)[:, 1:, :]

            # Backpropagate through the accumulated mean-reduced loss
            fisher_loss.backward()

            # accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    # handle complex gradients
                    if torch.is_complex(grad):
                        grad = grad.real
                    fisher_matrix[name] += grad.pow(2)
            num_batches += 1

        # *** KEY CHANGE: Normalize by the total number of batches (N) ***
        # F = (1/N) * sum_n( (grad L_mean)^2 ) is the proper empirical estimate for EWC
        if num_batches > 0:
            for name in fisher_matrix:
                fisher_matrix[name] /= float(num_batches)

        return fisher_matrix

    def _ewc_penalty(self):
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)

        total_penalty = torch.tensor(0.0, device=self.device)

        for fisher_matrix, optimal_params in zip(
            self.fisher_matrices, self.optimal_params_list
        ):
            penalty_for_exp = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if name in fisher_matrix:
                    p = param
                    p_opt = optimal_params[name]
                    f = fisher_matrix[name]

                    # handle complex tensors safely
                    if torch.is_complex(p):
                        p = p.real
                    if torch.is_complex(p_opt):
                        p_opt = p_opt.real
                    if torch.is_complex(f):
                        f = f.real

                    # EWC Penalty: 0.5 * F * (p - p_opt)^2
                    penalty_for_exp += (f * (p - p_opt).pow(2)).sum()
            total_penalty += penalty_for_exp

        # The factor of 0.5 from the EWC penalty formula is included in the EWC_lambda
        # in the training loop: total_loss = total_loss + self.ewc_lambda * penalty
        return total_penalty

    def train_exp(self, experience: Experience):
        self.model.train()
        train_loader = DataLoader(
            experience.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epochs):
            train_progress = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
            )

            for batch in train_progress:
                x = batch["input"].to(self.device)
                y = batch["output"].to(self.device)
                aux_input = batch["aux_input"].to(self.device)
                n_rollout_steps = y.shape[1]

                self.optimizer.zero_grad()

                # autoregressive loss (accumulates mean-reduced step losses)
                total_loss = 0.0
                x_cur = x.clone()
                for i in range(n_rollout_steps):
                    pred_i = self.model(x_cur, aux_input=aux_input)
                    total_loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x_cur = torch.cat([x_cur, pred_i], dim=1)[:, 1:, :]

                # EWC penalty (already includes the 0.5 factor from the formula)
                penalty = self._ewc_penalty()
                total_loss = total_loss + self.ewc_lambda * penalty

                total_loss.backward()
                self.optimizer.step()

                # Logging
                self.log_metric("train", "step_loss", total_loss.item())
                self.log_metric("train", "ewc_penalty", penalty.item())
                self.global_step += 1

        self.log_metric("train", "exp_idx", experience.exp_idx)

        # After training, compute Fisher info and save parameters
        self._update_ewc_params(experience)

    def _update_ewc_params(self, experience: Experience):
        current_fisher_matrix = self._compute_fisher_matrix(experience.train_dataset)
        self.fisher_matrices.append(current_fisher_matrix)

        # Parameter saving remains correct
        current_optimal_params = {
            name: (
                param.clone().detach().real
                if torch.is_complex(param)
                else param.clone().detach()
            )
            for name, param in self.model.named_parameters()
        }
        self.optimal_params_list.append(current_optimal_params)
