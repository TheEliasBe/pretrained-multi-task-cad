#!/usr/bin/env python3
import sys
from pathlib import Path

import pyarrow.dataset as ds

if len(sys.argv) != 2:
    print("Usage: python get_cadquery_code.py <deepcad_id>")
    sys.exit(1)

needle = sys.argv[1].strip()
# Update this path to your local Omni-CAD dataset directory
base = Path("./data/GenCAD-Code/data")

# Scan all Parquet files in the dir
dataset = ds.dataset(str(base), format="parquet")

print(f"Found {dataset.count_rows()} total entries.")

# Normalize deepcad_id to string in the filter by casting
f = ds.field("deepcad_id").cast("string") == needle
cols = ["deepcad_id", "cadquery"]

tbl = dataset.to_table(filter=f, columns=cols)
if tbl.num_rows:
    # Return the first match
    print(tbl.column("cadquery")[0].as_py())
    sys.exit(0)

print(f"No entry found for deepcad_id: {needle}")
sys.exit(2)
