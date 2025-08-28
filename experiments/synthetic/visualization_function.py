"""Functions for the visualization."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

pm = PathManager()
pm.register_handler(ManifoldPathHandler())


manifold_rootdir = ""
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def visualize_training_curve(
    n_seed: int = 1,
    experiment_name: str = "default",
    logging_type: str = "uniform",
    manifold_rootdir: str = manifold_rootdir,
):
    naive_cf_val_losses = torch.zeros((500 + 1, n_seed))
    moe_cf_val_losses = torch.zeros((500 + 1, n_seed))
    late_stage_cf_val_losses = torch.zeros((500 + 1, n_seed))

    for seed in range(n_seed):
        with pm.open(
            f"manifold://{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={seed}/naive_cf/training_process/val_losses.pt",
            "rb",
        ) as f:
            naive_cf_val_losses[:, seed] = torch.load(f)
        with pm.open(
            f"manifold://{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={seed}/moe_cf/training_process/val_losses.pt",
            "rb",
        ) as f:
            moe_cf_val_losses[:, seed] = torch.load(f)
        with pm.open(
            f"manifold://{manifold_rootdir}/{experiment_name}/{experiment_name},logging={logging_type},seed={seed}/naive_cf/late_stage_training_process/val_losses.pt",
            "rb",
        ) as f:
            late_stage_cf_val_losses[:, seed] = torch.load(f)

    x = torch.linspace(0, 50000 + 1, 500 + 1)[1:-1]
    early_stage_single_val_loss = naive_cf_val_losses[1:-1]
    early_stage_moe_val_loss = moe_cf_val_losses[1:-1]
    late_stage_val_loss = late_stage_cf_val_losses[1:-1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(
        x,
        early_stage_single_val_loss.mean(dim=-1),
        linewidth=2,
        color=colors[2],
        label="early stage (naive CF)",
    )
    ax.plot(
        x,
        early_stage_moe_val_loss.mean(dim=-1),
        linewidth=2,
        color=colors[3],
        label="early stage (MoE CF)",
    )
    ax.plot(
        x,
        late_stage_val_loss.mean(dim=-1),
        linewidth=2,
        color="gray",
        label="late stage",
    )

    # for seed in range(n_seed):
    ax.fill_between(
        x,
        early_stage_single_val_loss.min(dim=-1)[0],
        early_stage_single_val_loss.max(dim=-1)[0],
        color=colors[2],
        alpha=0.2,
    )
    ax.fill_between(
        x,
        early_stage_moe_val_loss.min(dim=-1)[0],
        early_stage_moe_val_loss.max(dim=-1)[0],
        color=colors[3],
        alpha=0.2,
    )
    ax.fill_between(
        x,
        late_stage_val_loss.min(dim=-1)[0],
        late_stage_val_loss.max(dim=-1)[0],
        color="gray",
        alpha=0.2,
    )

    ax.set_title("validation mean-squared-error (MSE)", fontsize=16)
    ax.set_xlabel("training steps", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    # plt.savefig(f"{save_path}/training_curve.png", dpi=300, bbox_inches="tight")


def visualize_baseline_performance(
    experiment_name: str = "default",
    logging_type: str = "uniform",
    manifold_rootdir: str = manifold_rootdir,
):
    with pm.open(
        f"manifold://{manifold_rootdir}/{experiment_name}/reference_performance.csv",
        "r",
    ) as f:
        reference_df = pd.read_csv(f)
    with pm.open(
        f"manifold://{manifold_rootdir}/{experiment_name}/baseline_performance_{logging_type}.csv",
        "r",
    ) as f:
        df = pd.read_csv(f)

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 3))
    n_candidate_action = reference_df["n_candidate_action_eval"]

    for j, n_cand_ in enumerate(n_candidate_action):
        reference_df_ = reference_df[reference_df["n_candidate_action_eval"] == n_cand_]
        df_ = df[df["n_candidate_action_eval"] == n_cand_]

        optimal = reference_df_["optimal"].values
        uniform = reference_df_["uniform"].values
        naive_cf = df_["naive-cf"].values
        moe = df_["moe-cf"].values
        greedy = df_["greedy-algo"].values

        axes[j].bar(
            [0.0],
            optimal.mean(),
            yerr=optimal.std(),
            alpha=0.8,
            linewidth=2,
            error_kw={"linewidth": 2},
        )
        axes[j].bar(
            [1.0],
            greedy.mean(),
            yerr=greedy.std(),
            alpha=0.8,
            linewidth=2,
            error_kw={"linewidth": 2},
        )
        axes[j].bar(
            [2.0],
            naive_cf.mean(),
            yerr=naive_cf.std(),
            alpha=0.8,
            linewidth=2,
            error_kw={"linewidth": 2},
        )
        axes[j].bar(
            [3.0],
            moe.mean(),
            yerr=moe.std(),
            alpha=0.8,
            linewidth=2,
            error_kw={"linewidth": 2},
        )
        axes[j].bar(
            [4.0],
            uniform.mean(),
            yerr=uniform.std(),
            alpha=0.8,
            linewidth=2,
            error_kw={"linewidth": 2},
        )

        axes[j].set_title(f"candidate size: k={n_cand_}", fontsize=16)
        axes[j].set_xticks(
            [0, 1, 2, 3, 4],
            ["optimal", "greedy", "naive CF", "MoE CF", "uniform"],
            rotation=45,
        )
        axes[j].set_ylabel("policy expected reward")
        axes[j].set_ylim(0, 8.0)
        # plt.savefig(f"{save_path}/training_curve.png", dpi=300, bbox_inches="tight")


def scatter_plot():
    # tentative values (from prelim results)
    epoch_1e1 = [1119, 62, 415]
    epoch_5e2 = [341, 523, 524]
    epoch_1e2 = [449, 162, 411]
    epoch_5e3 = [1140, 742, 1041]
    epoch_ca = [5000, 5000, 5000]

    performance_1e1 = [4.92, 3.81, 4.33]
    performance_5e2 = [4.05, 4.15, 4.27]
    performance_1e2 = [3.83, 3.77, 3.81]
    performance_5e3 = [3.81, 3.79, 3.78]
    performance_ca = [6.04, 6.51, 6.05]

    plt.figure(figsize=(12, 6))
    plt.scatter(
        epoch_ca,
        performance_ca,
        marker="*",
        s=1000,
        label="credit-assigned",
        c="black",
        zorder=2,
    )
    plt.scatter(
        epoch_1e1,
        performance_1e1,
        marker="o",
        s=1000,
        label="vanilla (lr=1e-1)",
        zorder=2,
    )
    plt.scatter(
        epoch_5e2,
        performance_5e2,
        marker="s",
        s=1000,
        label="vanilla (lr=5e-2)",
        zorder=2,
    )
    plt.scatter(
        epoch_1e2,
        performance_1e2,
        marker="^",
        s=1000,
        label="vanilla (lr=1e-2)",
        zorder=2,
    )
    plt.scatter(
        epoch_5e3,
        performance_5e3,
        marker="v",
        s=1000,
        label="vanilla (lr=5e-3)",
        zorder=2,
    )

    plt.hlines(
        [7.54],
        xmin=[0],
        xmax=[5000],
        colors="gray",
        linestyle="dashed",
        alpha=0.8,
        zorder=1,
    )
    plt.hlines(
        [3.95],
        xmin=[0],
        xmax=[5000],
        colors="gray",
        linestyle="--",
        alpha=0.8,
        zorder=1,
    )

    plt.legend(fontsize=20, loc="upper left")
