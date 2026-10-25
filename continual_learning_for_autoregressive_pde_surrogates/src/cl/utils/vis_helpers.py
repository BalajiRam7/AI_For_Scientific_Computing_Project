import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np


def plot_learning_vs_final_performance(
    all_eval_logs: List[Dict[str, List]],
):
    learning_losses = [all_eval_logs[i]["values"][i] for i in range(len(all_eval_logs))]
    final_losses = [all_eval_logs[-1]["values"][i] for i in range(len(all_eval_logs))]
    n = max(len(learning_losses), len(final_losses))
    indices = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(indices - width / 2, learning_losses, width, label="Learning")
    ax.bar(indices + width / 2, final_losses, width, label="Final")
    ax.set_xlabel("Experience")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Learning vs Final Loss per Experience")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Exp {i + 1}" for i in range(n)])
    ax.legend()
    fig.tight_layout()

    return fig
