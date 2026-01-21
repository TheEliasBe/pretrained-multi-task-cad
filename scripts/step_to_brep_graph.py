import pathlib
import random

from models.config import Config
from modules.brep_to_graph import BrepToGraphConverter

solid_model_converter = BrepToGraphConverter()

config = Config()
step_file_dir = pathlib.Path(config.data.root_dir).parent.parent.parent / "deepcad_step"
dgl_graph_dir = (
    pathlib.Path(config.data.root_dir).parent.parent.parent / "deepcad_graphs"
)

if not dgl_graph_dir.exists():
    dgl_graph_dir.mkdir(parents=True)

files = list(step_file_dir.rglob("*.step"))
random.shuffle(files)
for step_file in files:
    # Get relative path from step_file_dir
    relative_subpath = step_file.relative_to(step_file_dir).parent
    output_subdir = dgl_graph_dir / relative_subpath

    # Create subdir if it doesn't exist
    output_subdir.mkdir(parents=True, exist_ok=True)

    output_path = output_subdir / (step_file.stem + ".bin")
    if not output_path.exists():
        try:
            solid_model_converter.process_one_file(
                file_path=step_file,
                output_path=output_path,
            )
        except:
            print(f"Failed to process {step_file}")
