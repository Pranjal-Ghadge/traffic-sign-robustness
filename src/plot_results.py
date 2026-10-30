
import os
import pandas as pd
import matplotlib.pyplot as plt

results=pd.read_csv("results/gtsrb_corruption_results.csv")

os.makedirs("results/plots", exist_ok=True)

results["severity"]=pd.to_numeric(results["severity"])
results["accuracy"]=pd.to_numeric(results["accuracy"])*100
results["macro_f1"]=pd.to_numeric(results["macro_f1"])

plt.figure(figsize=(12, 7))

for corruption in results["corruption"].unique():
    data=results[results["corruption"]==corruption].sort_values("severity")
    plt.plot(data["severity"], data["accuracy"], marker="o", label=corruption)

plt.xlabel("Corruption Severity")
plt.ylabel("Accuracy (%)")
plt.title("GTSRB Classification Accuracy Under Different Corruptions")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("results/plots/accuracy_vs_severity.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 7))

for corruption in results["corruption"].unique():
    data=results[results["corruption"]==corruption].sort_values("severity")
    plt.plot(data["severity"], data["macro_f1"], marker="o", label=corruption)

plt.xlabel("Corruption Severity")
plt.ylabel("Macro-F1 Score")
plt.title("GTSRB Macro-F1 Under Different Corruptions")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("results/plots/macro_f1_vs_severity.pdf", bbox_inches="tight")
plt.close()

print("Graphs saved successfully!")
print("results/plots/accuracy_vs_severity.pdf")
print("results/plots/macro_f1_vs_severity.pdf")
