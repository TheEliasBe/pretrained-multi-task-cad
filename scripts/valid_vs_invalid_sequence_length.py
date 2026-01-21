import json
import pathlib

from scipy.stats import mannwhitneyu

from models.config import Config

# --- Config ---
config = Config()
cwd = pathlib.Path(__file__).parent.parent.resolve()
results_path = cwd / "results" / "cadcoder-seqcompl" / "vivid-sky-38_epoch_25.json"


# --- Load results ---
with open(results_path) as f:
    results = json.load(f)

results = results["train_samples"]
total_length_valid = 0
total_length_invalid = 0
for res in results:
    if res["epoch"] < 24:
        continue
    if res["is_valid"]:
        total_length_valid += len(res["true_code"])
    else:
        total_length_invalid += len(res["true_code"])
num_valid = sum(1 for res in results if res["is_valid"])
num_invalid = sum(1 for res in results if not res["is_valid"])
avg_length_valid = total_length_valid / num_valid if num_valid > 0 else 0
avg_length_invalid = total_length_invalid / num_invalid if num_invalid > 0 else 0
print(f"Number of valid sequences: {num_valid}")
print(f"Number of invalid sequences: {num_invalid}")
print(f"Average length of valid sequences: {avg_length_valid}")
print(f"Average length of invalid sequences: {avg_length_invalid}")

valid_lengths = [len(res["true_code"]) for res in results if res["is_valid"]]
invalid_lengths = [len(res["true_code"]) for res in results if not res["is_valid"]]
stat, p_value = mannwhitneyu(valid_lengths, invalid_lengths, alternative="less")
print(f"Mann-Whitney U test p-value (valid < invalid): {p_value}")
if p_value < 0.05:
    print(
        "Statistically significant: valid sequences are shorter than invalid sequences."
    )
else:
    print("Not statistically significant.")
