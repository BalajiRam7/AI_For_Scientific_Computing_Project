import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aim import Repo, Run


METHODS = ("naive", "replay", "replay_ts", "gem", "a_gem", "ewc")


def collect_metrics(root: Path):
    rows = []
    for method in METHODS:
        method_root = root / method
        run_dirs = sorted((method_root / "runs").iterdir())
        run_dir = next(path for path in run_dirs if path.is_dir())
        repo = Repo(str(method_root))
        repo._recreate_index()
        run = Run(repo=str(method_root), run_hash=run_dir.name, read_only=True)
        values = {}
        for metric in run.metrics():
            if metric.name in ("test_loss_stream", "test_loss_rell2_stream"):
                values[metric.name] = float(metric.dataframe().iloc[-1]["value"])
        rows.append(
            {
                "algorithm": method,
                "final_test_loss_stream": values["test_loss_stream"],
                "final_test_relative_l2_stream": values["test_loss_rell2_stream"],
                "run_id": run_dir.name,
            }
        )
    return rows


def write_outputs(root: Path, rows):
    summary_path = root / "summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        row["algorithm"].replace("_", "-").upper()
        if row["algorithm"] in ("replay_ts", "a_gem")
        else row["algorithm"].capitalize()
        for row in rows
    ]
    loss = np.array([row["final_test_loss_stream"] for row in rows])
    relative_l2 = np.array(
        [row["final_test_relative_l2_stream"] for row in rows]
    )
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"]
    plot_dir = root / "comparison_plots"
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = axis.bar(labels, loss, color=colors)
    axis.set_yscale("log")
    axis.set_ylabel("Final stream MSE (log scale)")
    axis.set_title("Mixed AdvDiff1D + Burgers1D CPU smoke test")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, loss):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2g}", ha="center", fontsize=8)
    fig.savefig(plot_dir / "mixed_method_mse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = axis.bar(labels, relative_l2, color=colors)
    axis.set_ylabel("Final stream relative $L^2$ error")
    axis.set_title("Mixed-stream continual-learning comparison")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, relative_l2):
        axis.text(bar.get_x() + bar.get_width() / 2, value + max(relative_l2) * 0.025, f"{value:.3f}", ha="center", fontsize=8)
    fig.savefig(plot_dir / "mixed_method_relative_l2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_path, plot_dir


if __name__ == "__main__":
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/smoke_tests/mixed_advdiff1d_burgers1d")
    summary, plots = write_outputs(output_root, collect_metrics(output_root))
    print(f"Wrote {summary}")
    print(f"Wrote plots to {plots}")
