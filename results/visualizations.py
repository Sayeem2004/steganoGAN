import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

data = pd.read_csv("results/steganogan_results.csv")

data["model_type"] = data["model_type"].str.lower()

data_filtered = data[data["data_depth"] <= 6]

data_filtered.to_csv("output/steganogan_data.csv", index=False)

metrics = ["accuracy", "rs_bpp", "psnr", "ssim"]
model_types = ["basic", "residual", "dense"]


def create_4panel_plots():
    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        ax = axes[0]
        for model in model_types:
            model_data = data_filtered[data_filtered["model_type"] == model]
            ax.plot(
                model_data["data_depth"],
                model_data[metric],
                "o-",
                label=model.capitalize(),
            )
        ax.set_xlabel("Data Depth")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"All Models: {metric.upper()} vs Data Depth")
        ax.legend()
        ax.grid(True)

        for i, model in enumerate(model_types):
            ax = axes[i + 1]
            model_data = data_filtered[data_filtered["model_type"] == model]
            ax.plot(
                model_data["data_depth"],
                model_data[metric],
                "o-",
                color=f"C{i}",
                label=model.capitalize(),
            )
            ax.set_xlabel("Data Depth")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{model.capitalize()}: {metric.upper()} vs Data Depth")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(
            f"output/steganogan_{metric}_4panel.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def create_individual_metric_plots():
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for i, model in enumerate(model_types):
            model_data = data_filtered[data_filtered["model_type"] == model]
            plt.plot(
                model_data["data_depth"],
                model_data[metric],
                "o-",
                label=model.capitalize(),
            )

        plt.xlabel("Data Depth")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Data Depth (First 6 Depths)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"output/steganogan_{metric}_combined.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def create_full_depth_plots():
    for metric in metrics:
        plt.figure(figsize=(12, 7))

        for i, model in enumerate(model_types):
            model_data = data[data["model_type"] == model]
            plt.plot(
                model_data["data_depth"],
                model_data[metric],
                "o-",
                label=model.capitalize(),
                linewidth=2,
                markersize=8,
            )

        plt.xlabel("Data Depth", fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.title(f"{metric.upper()} vs Data Depth (All 12 Depths)", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"output/steganogan_{metric}_all_depths.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    print("Exporting CSV data...")

    print("Creating individual metric plots (all 12 depths)...")
    create_full_depth_plots()

    print("Done! All visualizations saved to 'output' directory.")
