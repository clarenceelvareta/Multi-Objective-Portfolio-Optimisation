from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "scaling_results" / "gurobi_scaling_full_frontier_summary.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load new scaling results
df = pd.read_csv(CSV_PATH)

# Sort by universe size
df = df.sort_values("M")

M = df["M"]
total_time = df["total_runtime_sec"]
median_time = df["median_runtime_sec"]

# Relative runtime
base_time = total_time.iloc[0]
relative = total_time / base_time

# Plot total full-frontier runtime
plt.figure(figsize=(7, 4.5))
plt.plot(M, total_time, marker="o", linewidth=2)

for x, y, r in zip(M, total_time, relative):
    plt.annotate(
        f"{y:.2f}s\n({r:.1f}x)",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9
    )

plt.xlabel("Universe Size $M$")
plt.ylabel("Total Runtime for 20-Point Frontier (s)")
plt.title("Stratified Gurobi $\\epsilon$-Constraint Scaling\n"
          "Full 20-Point CVaR Frontier per Universe Size")
plt.grid(True, alpha=0.3)
plt.tight_layout()

png_path = FIG_DIR / "gurobi_scaling_stratified.png"
pdf_path = FIG_DIR / "gurobi_scaling_stratified.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")