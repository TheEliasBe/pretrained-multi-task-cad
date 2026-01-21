import json
import pathlib
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from models.config import Config
from modules.execute_cad_code import execute_cad_code

# --- Config ---
config = Config()
results_path = "path/to/your/results.json"
data_root = pathlib.Path(config.data.root_dir)
data_dir = data_root / "GenCAD-Code" / "data"

# --- Load dataset ---
cadquery_df = pd.concat(
    [
        pd.read_parquet(data_dir / "train-00000-of-00002.parquet"),
        pd.read_parquet(data_dir / "train-00001-of-00002.parquet"),
        pd.read_parquet(data_dir / "test-00000-of-00001.parquet"),
        pd.read_parquet(data_dir / "validation-00000-of-00001.parquet"),
    ],
    ignore_index=True,
)
cadquery_df = cadquery_df.set_index("deepcad_id")

# --- Load results ---
with open(results_path) as f:
    results = json.load(f)

results = results["train_samples"]
for res in results:
    if res["is_valid"]:
        deepcad_id = res["id"]
        row = cadquery_df.loc[deepcad_id]
        # Load ground truth image
        img_bytes = row["image"]["bytes"]
        if isinstance(img_bytes, str):
            img_bytes = img_bytes.encode("latin1")
        gt_img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Render generated CAD
        try:
            gen_img = execute_cad_code(res["generated_code"], result_var="solid")
        except Exception:
            gen_img = Image.new("RGB", gt_img.size, (255, 0, 0))  # Red for error

        # --- Visualization ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_img)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(gen_img)
        axes[1].set_title("Generated CAD")
        axes[1].axis("off")
        plt.suptitle(
            f"ID: {deepcad_id}\nChamfer: {res['chamfer_distance']}, F-score: {res['fscore']}, CodeBLEU: {res['codebleu']:.2f}"
        )
        plt.show()
