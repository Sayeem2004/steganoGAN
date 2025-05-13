import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_steganogan_data(csv_path):
    data = pd.read_csv(csv_path)
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("Set2")

    plt.figure(figsize=(12, 8))
    model_types = data["model_type"].unique()

    for model in model_types:
        model_data = data[data["model_type"] == model]
        model_data = model_data.sort_values(by="data_depth")
        plt.plot(model_data["data_depth"], model_data["rs_bpp"], "o-",
                 linewidth=2, markersize=8, label=f"{model.capitalize()}",)

    plt.title("SteganoGAN Models: Data Depth vs RS_BPP (Bits Per Pixel)", fontsize=16)
    plt.xlabel("Data Depth", fontsize=14)
    plt.ylabel("RS_BPP (Bits Per Pixel)", fontsize=14)
    plt.xticks(range(1, data["data_depth"].max() + 1))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.figtext(0.02, 0.02, "Higher RS_BPP values indicate more data being hidden in the image",
                fontsize=10, style="italic")

    output_path = "results/steganogan_model_depth_vs_rsbpp.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.figure(figsize=(15, 10))
    plt.figure(figsize=(16, 10))

    for i, model in enumerate(model_types, 1):
        plt.subplot(2, 2, i)
        model_data = data[data["model_type"] == model].sort_values(by="data_depth")
        plt.plot(model_data["data_depth"], model_data["rs_bpp"], "o-",
            linewidth=2, markersize=8, color=sns.color_palette("Set2")[i - 1])

        plt.title(f"{model.capitalize()} Model", fontsize=14)
        plt.xlabel("Data Depth", fontsize=12)
        plt.ylabel("RS_BPP", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(range(1, data["data_depth"].max() + 1))

    plt.tight_layout()
    individual_path = "results/steganogan_individual_models.png"
    plt.savefig(individual_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    csv_path = "results/steganogan_data.csv"
    analyze_steganogan_data(csv_path)
