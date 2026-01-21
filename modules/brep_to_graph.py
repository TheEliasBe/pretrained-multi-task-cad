import argparse
import pathlib
import signal
from itertools import repeat
from multiprocessing.pool import Pool
from typing import Optional

import dgl
import numpy as np
import torch
from occwl.compound import Compound
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm


class BrepToGraphConverter:
    def __init__(
        self,
        curv_u_samples=10,
        surf_u_samples=10,
        surf_v_samples=10,
        num_processes=8,
    ):
        self.curv_u_samples = curv_u_samples
        self.surf_u_samples = surf_u_samples
        self.surf_v_samples = surf_v_samples
        self.num_processes = num_processes

    def build_graph(self, solid):
        # Build face adjacency graph with B-rep entities as node and edge features
        graph = face_adjacency(solid)

        # Compute the UV-grids for faces
        graph_face_feat = []
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]
            # Compute UV-grids
            points = uvgrid(
                face,
                method="point",
                num_u=self.surf_u_samples,
                num_v=self.surf_v_samples,
            )
            normals = uvgrid(
                face,
                method="normal",
                num_u=self.surf_u_samples,
                num_v=self.surf_v_samples,
            )
            visibility_status = uvgrid(
                face,
                method="visibility_status",
                num_u=self.surf_u_samples,
                num_v=self.surf_v_samples,
            )
            mask = np.logical_or(
                visibility_status == 0, visibility_status == 2
            )  # 0: Inside, 1: Outside, 2: On boundary
            # Concatenate channel-wise to form face feature tensor
            face_feat = np.concatenate((points, normals, mask), axis=-1)
            graph_face_feat.append(face_feat)
        graph_face_feat = np.asarray(graph_face_feat)

        # Compute the U-grids for edges
        graph_edge_feat = []
        for edge_idx in graph.edges:
            # Get the B-rep edge
            edge = graph.edges[edge_idx]["edge"]
            # Ignore degenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # Compute U-grids
            points = ugrid(edge, method="point", num_u=self.curv_u_samples)
            tangents = ugrid(edge, method="tangent", num_u=self.curv_u_samples)
            # Concatenate channel-wise to form edge feature tensor
            edge_feat = np.concatenate((points, tangents), axis=-1)
            graph_edge_feat.append(edge_feat)
        graph_edge_feat = np.asarray(graph_edge_feat)

        # Convert face-adj graph to DGL format
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
        dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
        dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
        return dgl_graph

    def process_one_file(
        self, file_path: pathlib.Path, output_path: Optional[pathlib.Path] = None
    ):
        solid = Compound.load_from_step(file_path)
        graph = self.build_graph(solid)
        if output_path is None:
            return graph
        dgl.data.utils.save_graphs(str(output_path), [graph])

    def initializer(self):
        """Ignore CTRL+C in the worker process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def process(self, input_folder, output_folder, use_pool=True):
        input_path = pathlib.Path(input_folder)
        output_path = pathlib.Path(output_folder)
        total_fails = 0
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        step_files = list(input_path.glob("*.st*p"))

        if use_pool:
            pool = Pool(processes=self.num_processes, initializer=self.initializer)
            r = []
            try:
                r = list(
                    pool.imap(self.process_one_file, zip(step_files, repeat(self)))
                )
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            tqdm.write(f"Processed {len(r)} files.")
        else:
            for step_file in tqdm(step_files):
                try:
                    self.process_one_file(step_file, output_path)
                except Exception as e:
                    tqdm.write(f"Failed to process {step_file}: {e}")
                    total_fails += 1
            tqdm.write(f"Processed {len(step_files)} files. {total_fails} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument(
        "--curv_u_samples", type=int, default=10, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()

    # get sub dirs
    # Update these paths to your local data directories
    input_folder = "./data/step_files"
    output_folder = "./data/brep_graphs"

    converter = BrepToGraphConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        curv_u_samples=args.curv_u_samples,
        surf_u_samples=args.surf_u_samples,
        surf_v_samples=args.surf_v_samples,
        num_processes=args.num_processes,
    )
    converter.process(use_pool=False)
