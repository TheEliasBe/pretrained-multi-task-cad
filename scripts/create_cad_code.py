#!/usr/bin/env python3
"""
Script to create CAD code from DeepCAD JSON files using pycadseq.
Processes JSON files from config.data.root_dir / omni_cad / json and
saves the generated CAD code to parquet files.
"""

import pathlib

import pandas as pd
from pycadseq.json_importer.process_deepcad_to_pycadseq import (
    generate_pycadseq_code_from_deepcad_json,
)

from models.config import Config


def process_json_to_cad_code():
    """Process DeepCAD JSON files and generate CAD code using pycadseq."""

    # Load configuration
    config = Config()

    # Set up source and output directories
    json_source_dir = pathlib.Path(config.data.root_dir) / "omni_cad" / "json"
    output_dir = pathlib.Path(config.data.root_dir) / "cad_code"

    if not json_source_dir.exists():
        print(f"Error: JSON source directory does not exist: {json_source_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subdirectories in the JSON source directory
    subdirectories = [d for d in json_source_dir.iterdir() if d.is_dir()]

    if not subdirectories:
        print(f"No subdirectories found in {json_source_dir}")
        return

    print(f"Found {len(subdirectories)} subdirectories to process")

    # Process each subdirectory
    for subdir in subdirectories:
        subdir_name = subdir.name
        print(f"Processing subdirectory: {subdir_name}")

        # Find all JSON files in the subdirectory
        json_files = list(subdir.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {subdir}")
            continue

        print(f"Found {len(json_files)} JSON files in {subdir_name}")

        # Process each JSON file and collect results
        results = []
        processed_count = 0
        failed_count = 0

        for json_file in json_files:
            try:
                # Generate CAD code from JSON file
                code = generate_pycadseq_code_from_deepcad_json(str(json_file))

                # Create a record for this file
                record = {
                    "filename": json_file.name,
                    "filepath": str(json_file.relative_to(json_source_dir)),
                    "subdirectory": subdir_name,
                    "cad_code": code,
                    "deepcad_id": f"{subdir_name}/{json_file.stem}",  # Following the pattern from other scripts
                }

                results.append(record)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files from {subdir_name}")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                failed_count += 1
                continue

        if results:
            # Create DataFrame and save to parquet
            df = pd.DataFrame(results)
            output_file = output_dir / f"{subdir_name}.parquet"

            try:
                df.to_parquet(output_file, index=False)
                print(f"Saved {len(results)} records to: {output_file}")
                print(
                    f"Successfully processed: {processed_count}, Failed: {failed_count}"
                )
            except Exception as e:
                print(f"Error saving parquet file {output_file}: {e}")
        else:
            print(f"No valid records generated for subdirectory {subdir_name}")

    print("Processing complete!")


def main():
    """Main function to run the CAD code generation process."""
    try:
        process_json_to_cad_code()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
