import json
import os
import pathlib
import tempfile
from io import BytesIO

import cairosvg
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from models.config import Config
from modules.execute_cad_code import execute_cad_code

# --- Config ---
config = Config()
project_root = pathlib.Path(__file__).parent.parent
results_path = (
    project_root / "results" / "cadcoder-image" / "vital-water-27_epoch_5_batch_0.json"
)
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


# Helper: render CadQuery Workplane to PIL Image
def render_workplane(wp, size=(512, 512)):
    """Convert CadQuery Workplane to PIL Image via STL mesh and trimesh rendering."""
    # Export workplane to temporary SVG
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
        wp.export(
            tmp_svg.name,
            opt={
                "width": size[0],
                "height": size[1],
                "marginLeft": 0,
                "marginTop": 0,
                "showAxes": False,
                "fillColor": (32, 32, 32),
            },
        )
    # Convert SVG to PNG
    try:
        png_bytes = cairosvg.svg2png(url=tmp_svg.name, background_color="white")
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
    finally:
        os.remove(tmp_svg.name)
    return img


results = results["train_samples"]

for res in results:
    if res["is_valid"] or True:
        deepcad_id = res["id"]
        row = cadquery_df.loc[deepcad_id]
        # Load ground truth image
        img_bytes = row["image"]["bytes"]
        true_code = row["cadquery"]
        if isinstance(img_bytes, str):
            img_bytes = img_bytes.encode("latin1")
        gt_img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Render generated CAD
        try:
            # gen_obj = execute_cad_code(res["generated_code"], result_var="solid")
            true_source_obj = execute_cad_code(true_code, result_var="solid")
            true_res_obj = execute_cad_code(res["true_code"], result_var="solid")
            # generated_obj = execute_cad_code(res["generated_code"], result_var="solid")
            print("Source Code")
            print(true_code)
            print("True Result Code")
            print(res["true_code"])
            # if isinstance(gen_obj, str) or gen_obj is None:
            #     raise ValueError("Invalid CAD object")
            # gen_img = render_workplane(gen_obj, size=gt_img.size)
            true_source_img = render_workplane(true_source_obj, size=gt_img.size)
            true_res_img = render_workplane(true_res_obj, size=gt_img.size)
            # generated_img = render_workplane(generated_obj, size=gt_img.size)
        except Exception:
            # red image on error
            gen_img = Image.new("RGB", gt_img.size, (255, 0, 0))

        # --- Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(gt_img)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(true_res_img)
        axes[1].set_title("True Result CAD")
        axes[1].axis("off")

        # axes[2].imshow(generated_img)
        # axes[2].set_title("Generated CAD")
        # axes[2].axis("off")
        # plt.suptitle(f"ID: {deepcad_id}\nChamfer: {res['chamfer_distance']}, F-score: {res['fscore']}, CodeBLEU: {res['codebleu']:.2f}")
        save_file = (
            project_root / "results" / "viz" / f"cadcoder_image_{deepcad_id[5:]}.png"
        )
        print(f"Saving to {save_file}")
        plt.savefig(
            save_file,
        )
        plt.close()
