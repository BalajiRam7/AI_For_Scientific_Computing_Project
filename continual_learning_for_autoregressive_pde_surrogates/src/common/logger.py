from aim.sdk.objects import image
from omegaconf import DictConfig, OmegaConf
import os
from aim import Run, Image
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.init_aim()
        self.init_run_dirs()

    def init_aim(self):
        repo = os.path.join(self.cfg.logger.outputs_dir, self.cfg.logger.repo)
        print("\nInitializing logger ...")
        self.run = Run(
            repo=repo,
            experiment=self.cfg.logger.experiment,
            # system_tracking_interval=True,
        )
        runs_dir = os.path.join(self.cfg.logger.outputs_dir, "runs")
        self.run_root = os.path.join(runs_dir, self.run.hash)
        os.makedirs(self.run_root, exist_ok=True)

        # Set tags if given
        if self.cfg.logger.tags:
            for tag in self.cfg.logger.tags:
                self.run.add_tag(tag)

        self.run["hparams"] = OmegaConf.to_object(self.cfg)

        config_path = os.path.join(self.run_root, "config.yaml")
        OmegaConf.save(self.cfg, config_path)

    def init_run_dirs(self):
        self.plots_dir = os.path.join(self.run_root, "plots")
        self.checkpoints_dir = os.path.join(self.run_root, "checkpoints")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def finalize_run(self):
        pass

    def log_metric(
        self, metric_value: float, metric_name: str, metric_context: str, step: int
    ):
        self.run.track(
            metric_value,
            name=metric_name,
            step=step,
            context={"subset": metric_context},
        )

    def log_figure(self, fig, fig_name: str, fig_context: str, step: int):
        image_path = os.path.join(self.plots_dir, fig_name + ".png")
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

        # Load and log to aim
        img = Image(image_path)
        self.run.track(img, name=fig_name, step=step, context={"subset": fig_context})


def create_logger(cfg: DictConfig) -> Logger:
    return Logger(cfg)
