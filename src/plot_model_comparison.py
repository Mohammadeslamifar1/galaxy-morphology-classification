from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_PATH = "outputs/figures/model_comparison.png"

Path("outputs/figures").mkdir(parents=True, exist_ok=True)


models = ["Baseline CNN", "ResNet18"]
test_accuracies = [75.31, 86.08]


plt.figure(figsize=(8, 5))
bars = plt.bar(models, test_accuracies)

plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison on Galaxy10 SDSS Test Set")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, test_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        fontsize=11,
    )

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()

print(f"Model comparison plot saved to {OUTPUT_PATH}")