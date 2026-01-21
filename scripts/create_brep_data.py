#!/usr/bin/env python3
"""
Script to create BREP data by processing CAD code from parquet files.
Loads parquet files from config.data.root_dir / cad_code and appends
shape.to_step() calls with the appropriate step file names.
"""

import pathlib

import pandas as pd

from models.config import Config


def process_cad_code_files():
    """Process CAD code parquet files and append shape.to_step() calls."""

    # Load configuration
    config = Config()

    # Set up data directory path
    cad_code_dir = pathlib.Path(config.data.root_dir) / "cad_code"

    if not cad_code_dir.exists():
        print(f"Error: CAD code directory does not exist: {cad_code_dir}")
        return

    # Find all parquet files in the directory
    parquet_files = list(cad_code_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {cad_code_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files to process")

    # Process each parquet file
    for parquet_file in parquet_files:
        print(f"Processing: {parquet_file}")

        try:
            # Load the parquet file
            df = pd.read_parquet(parquet_file)

            # Check if deepcad_id column exists
            if "deepcad_id" not in df.columns:
                print(f"Warning: 'deepcad_id' column not found in {parquet_file}")
                print(f"Available columns: {list(df.columns)}")
                continue

            # Check if there's a code column (could be 'cadquery', 'code', etc.)
            code_column = None
            for col_name in ["cadquery", "code", "cad_code"]:
                if col_name in df.columns:
                    code_column = col_name
                    break

            if code_column is None:
                print(f"Warning: No code column found in {parquet_file}")
                print(f"Available columns: {list(df.columns)}")
                continue

            # Process each row
            processed_rows = []
            for idx, row in df.iterrows():
                deepcad_id = row["deepcad_id"]
                original_code = row[code_column]

                # Create step file name from deepcad_id
                step_file = f"{deepcad_id}.step"

                # Append the shape.to_step() call
                modified_code = original_code + f"\nshape.to_step('{step_file}')"

                # Create new row with modified code
                new_row = row.copy()
                new_row[code_column] = modified_code
                processed_rows.append(new_row)

            # Create new dataframe with processed data
            processed_df = pd.DataFrame(processed_rows)

            # Save the processed file (add suffix to avoid overwriting)
            output_file = (
                parquet_file.parent / f"{parquet_file.stem}_with_step_export.parquet"
            )
            processed_df.to_parquet(output_file, index=False)

            print(f"Saved processed data to: {output_file}")
            print(f"Processed {len(processed_df)} records")

        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue

    print("Processing complete!")


if __name__ == "__main__":
    process_cad_code_files()
