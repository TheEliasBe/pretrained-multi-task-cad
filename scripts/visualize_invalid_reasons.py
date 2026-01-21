import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def normalize_reason(reason: str) -> str:
    # replace "on line X"
    reason = re.sub(r"on line \d+", "on line N", reason)
    # replace "(detected at line X)"
    reason = re.sub(r"\(detected at line \d+\)", "(detected at line N)", reason)
    # replace "(<string>, line X)"
    reason = re.sub(r"\(<string>, line \d+\)", "(<string>, line N)", reason)
    return reason.strip()


def categorize_reason(reason: str) -> str:
    r = reason.lower()
    if "brep" in r:
        return "Invalid Geometry"
    if "::" in r:
        return "Invalid Geometry"
    if "expected" in r and "float" in r:
        return "Invalid Geometry"
    if "should be non" in r or "non null" in r:
        return "Invalid Geometry"
    if "not defined" in r:
        return "Undefined Variables"
    if "has no attribute" in r:
        return "Undefined Variables"
    if "missing" in r and "parenthesis" in r:
        return "Missing Parenthesis"
    if "unmatched" in r or "was never closed" in r:
        return "Missing Parenthesis"
    if "unterminated string" in r:
        return "Unterminated String"
    if "decimal" in r:
        return "Invalid Decimal"
    if "unexpected indent" in r:
        return "Unexpected Indent"
    if (
        "invalid syntax" in r
        or "perhaps you forgot" in r
        or "float division by zero" in r
    ):
        return "Invalid Syntax"
    if "does not match opening parenthesis" in r:
        return "Missing Parenthesis"
    return "Invalid Syntax"


error_categories = {
    "Invalid Geometry": 0,
    "Missing Parenthesis": 0,
    "Unterminated String": 0,
    "Invalid Decimal": 0,
    "Unexpected Indent": 0,
    "Invalid Syntax": 0,
    "Undefined Variables": 0,
}

# Load JSON data
with open("results/cadcoder-image/faithful-river-14_epoch_70.json", "r") as f:
    data = json.load(f)

# Normalize invalid reasons
reasons = [
    normalize_reason(s["invalid_reason"])
    for s in data["train_samples"]
    if not s["is_valid"]
]

# Categorize reasons into error categories
category_counts = Counter(categorize_reason(r) for r in reasons)

print(category_counts)

# Plot relative share of categories sorted descending
sorted_counts = category_counts.most_common()
labels = [cat for cat, _ in sorted_counts]
values = [count for _, count in sorted_counts]
total = sum(values)
shares = [v / total for v in values]
plt.figure(figsize=(8, 4))
bars = plt.bar(range(len(shares)), shares, color="black", tick_label=labels)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Percentage of total")
# Format y-axis as percentages
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

# Save plots
plt.savefig("invalid_reason_histogram_categories.pgf", bbox_inches="tight")
plt.savefig("invalid_reason_histogram_categories.png", bbox_inches="tight")
