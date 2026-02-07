"""Visualization functions for RTC evaluation results."""

import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _short_level_name(level: str) -> str:
    """Convert 'worlds/l/grasp_easy.json' -> 'grasp_easy'."""
    return pathlib.Path(level).stem


def load_results(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load and preprocess evaluation results CSV."""
    df = pd.read_csv(csv_path)
    df["level_short"] = df["level"].map(_short_level_name)
    return df


def plot_solve_rate_by_delay_horizon(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: average metric vs (inference_delay, execute_horizon) for a method."""
    subset = df[df["method"] == method].copy()
    agg = subset.groupby(["delay", "execute_horizon"])[metric].mean().reset_index()
    pivot = agg.pivot(index="delay", columns="execute_horizon", values=metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{metric} by inference_delay & execute_horizon ({method})")
    ax.set_xlabel("execute_horizon")
    ax.set_ylabel("inference_delay (delay)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_method_comparison(
    df: pd.DataFrame,
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    aggregate: Literal["all", "per_level"] = "all",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Compare methods: bar chart of metric aggregated across (delay, execute_horizon)."""
    if aggregate == "all":
        agg = df.groupby("method")[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=agg, x="method", y=metric, hue="method", legend=False, ax=ax)
        ax.set_title(f"Mean {metric} by method (all delay/horizon)")
    else:
        agg = df.groupby(["method", "level_short"])[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=agg,
            x="level_short",
            y=metric,
            hue="method",
            ax=ax,
        )
        ax.set_title(f"Mean {metric} by method and level")
        ax.tick_params(axis="x", rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_delay_effect(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: metric vs inference_delay, one line per execute_horizon."""
    subset = df[df["method"] == method].copy()
    agg = subset.groupby(["delay", "execute_horizon"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for eh in sorted(agg["execute_horizon"].unique()):
        sub = agg[agg["execute_horizon"] == eh]
        ax.plot(sub["delay"], sub[metric], marker="o", label=f"execute_horizon={eh}")
    ax.set_xlabel("inference_delay")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs inference_delay ({method})")
    ax.legend()
    ax.set_xticks(sorted(agg["delay"].unique()))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_delay_vs_success_by_method(
    df: pd.DataFrame,
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: x=inference_delay, y=success rate; datapoints where execute_horizon=max(1, delay); one line per method."""
    df = df.copy()
    df["s"] = df["delay"].apply(lambda d: max(1, d))
    subset = df[df["execute_horizon"] == df["s"]].copy()
    agg = subset.groupby(["delay", "method"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("delay")
        ax.plot(sub["delay"], sub[metric], marker="o", label=method)
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row[metric]:.2f}",
                (row["delay"], row[metric]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
    ax.set_xlabel("inference_delay")
    ax.set_ylabel("success rate" if metric == "returned_episode_solved" else metric)
    ax.set_title(f"Success rate vs inference_delay (execute_horizon=max(1, delay))")
    ax.legend()
    ax.set_xticks(sorted(agg["delay"].unique()))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_per_level_heatmap(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: level vs (delay, execute_horizon) for a method."""
    subset = df[df["method"] == method].copy()
    subset["d/h"] = subset["delay"].astype(str) + "/" + subset["execute_horizon"].astype(str)
    agg = subset.groupby(["level_short", "d/h"])[metric].mean().reset_index()
    pivot = agg.pivot(index="level_short", columns="d/h", values=metric)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{metric} by level and (delay, execute_horizon) ({method})")
    ax.set_xlabel("(delay, execute_horizon)")
    ax.set_ylabel("Level")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_all(
    csv_path: str | pathlib.Path = "eval_logs/results.csv",
    output_dir: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Generate all standard visualizations and optionally save to output_dir."""
    df = load_results(csv_path)
    out = pathlib.Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def _save(name: str) -> pathlib.Path | None:
        return out / f"{name}.png" if out else None

    plot_solve_rate_by_delay_horizon(
        df, "realtime", output_path=_save("solve_rate_heatmap_realtime"), show=show
    )
    plot_method_comparison(df, aggregate="all", output_path=_save("method_comparison"), show=show)
    plot_method_comparison(df, aggregate="per_level", output_path=_save("method_comparison_per_level"), show=show)
    plot_delay_effect(df, "realtime", output_path=_save("delay_effect_realtime"), show=show)
    plot_delay_vs_success_by_method(df, output_path=_save("delay_vs_success_by_method"), show=show)
    plot_per_level_heatmap(df, "realtime", output_path=_save("per_level_heatmap_realtime"), show=show)


def main() -> None:
    """CLI entry point for generating all visualizations."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="eval_logs/results.csv",
        help="Path to results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures (if not set, only display)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (useful when saving to file on headless systems)",
    )
    args = parser.parse_args()
    plot_all(args.csv, args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
