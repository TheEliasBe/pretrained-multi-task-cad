import ast
import os
import re
import tempfile
from collections import Counter
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

from modules.execute_cad_code import execute_cad_code

try:
    from codebleu import calc_codebleu

    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False


class CadEvaluator:
    """
    This class calculates the metrics given two CAD shapes (ground truth and prediction).
    """

    def __init__(self, sampler_points=1024, threshold=0.05):
        self.n_points = sampler_points
        self.threshold = threshold
        np.random.seed(42)  # For reproducibility

    def evaluate_cadquery(self, code_gt: str, code_pred: str) -> dict:
        # Execute code → get CAD shape
        try:
            cad_gt = execute_cad_code(code_gt, result_var="solid")
            cad_pred = execute_cad_code(code_pred, result_var="solid")
            if isinstance(cad_pred, str):
                invalid_reason = cad_pred
                chamfer_distance = None
                f1 = None
                nc = None
                intersection_over_union = None
            else:
                invalid_reason = None
                shape_gt = cad_gt.val().wrapped
                shape_pred = cad_pred.val().wrapped
                # Mesh shapes → point clouds
                pts_gt, normals_gt = self._shape_to_points_ocp(shape_gt)
                pts_pred, normals_pred = self._shape_to_points_ocp(shape_pred)
                # Metrics
                chamfer_distance = self._chamfer_distance(pts_gt, pts_pred)
                f1 = self._fscore(pts_gt, pts_pred)
                nc = self._normal_consistency(
                    normals_gt, normals_pred, pts_gt, pts_pred
                )
                intersection_over_union = CadEvaluator.align_shapes(
                    shape_gt, shape_pred
                )
                if intersection_over_union is False or intersection_over_union == 0.0:
                    intersection_over_union = None
        except Exception as e:
            chamfer_distance = None
            f1 = None
            nc = None
            intersection_over_union = None
            invalid_reason = str(e)

        # CodeBLEU metric
        codebleu_score = self._calculate_codebleu(code_gt, code_pred)

        return {
            "chamfer": chamfer_distance,
            "fscore": f1,
            "normal_consistency": nc,
            "codebleu": codebleu_score,
            "invalid_reason": invalid_reason,
            "intersection_over_union": intersection_over_union,
        }

    def evaluate(self, code_gt: str, code_pred: str) -> dict:
        # Execute code → get CAD shape
        try:
            cad_gt = execute_cad_code(code_gt)
            cad_pred = execute_cad_code(code_pred)
            if isinstance(cad_pred, str):
                invalid_reason = cad_pred
                cd = None
                f1 = None
                nc = None
            else:
                invalid_reason = None
                shape_gt = cad_gt.to_occ()
                shape_pred = cad_pred.to_occ()
                # Mesh shapes → point clouds
                pts_gt, normals_gt = self._shape_to_points(shape_gt)
                pts_pred, normals_pred = self._shape_to_points(shape_pred)
                # Metrics
                cd = self._chamfer_distance(pts_gt, pts_pred)
                f1 = self._fscore(pts_gt, pts_pred)
                nc = self._normal_consistency(
                    normals_gt, normals_pred, pts_gt, pts_pred
                )
        except Exception as e:
            cd = None
            f1 = None
            nc = None
            invalid_reason = str(e)

        # CodeBLEU metric
        codebleu_score = self._calculate_codebleu(code_gt, code_pred)

        return {
            "chamfer": cd,
            "fscore": f1,
            "normal_consistency": nc,
            "codebleu": codebleu_score,
            "invalid_reason": invalid_reason,
        }

    def _shape_to_points_ocp(self, shape: "TopoDS_Shape"):
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.StlAPI import StlAPI_Writer

        # Mesh the shape
        mesh = BRepMesh_IncrementalMesh(shape, 0.05)
        mesh.Perform()

        # Write to temporary STL file
        writer = StlAPI_Writer()
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_file:
            stl_path = tmp_file.name
        writer.Write(shape, stl_path)

        # Load with trimesh and sample
        tri = trimesh.load_mesh(stl_path, force="mesh")
        tri.apply_translation(-tri.centroid)
        tri.apply_scale(1.0 / np.max(tri.extents))

        points, face_indices = trimesh.sample.sample_surface(
            tri, self.n_points, seed=42
        )
        normals = tri.face_normals[face_indices]

        os.remove(stl_path)
        return points, normals

    def _shape_to_points(self, shape: "TopoDS_Shape"):
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Extend.DataExchange import write_stl_file

        # Mesh the shape
        mesh = BRepMesh_IncrementalMesh(shape, 0.05)
        mesh.Perform()

        # Write to temporary STL file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_file:
            stl_path = tmp_file.name
        write_stl_file(shape, stl_path)

        # Load with trimesh and sample
        tri = trimesh.load_mesh(stl_path, force="mesh")
        tri.apply_translation(-tri.centroid)
        tri.apply_scale(1.0 / np.max(tri.extents))

        points, face_indices = trimesh.sample.sample_surface(
            tri, self.n_points, seed=42
        )
        normals = tri.face_normals[face_indices]

        os.remove(stl_path)
        return points, normals

    def _chamfer_distance(self, a, b):
        tree_a = cKDTree(a)
        tree_b = cKDTree(b)
        dists_a, _ = tree_a.query(b)
        dists_b, _ = tree_b.query(a)
        return np.mean(dists_a**2) + np.mean(dists_b**2)

    def _fscore(self, a, b):
        tree_a = cKDTree(a)
        tree_b = cKDTree(b)
        dists_a, _ = tree_a.query(b)
        dists_b, _ = tree_b.query(a)
        precision = np.mean((dists_b < self.threshold).astype(float))
        recall = np.mean((dists_a < self.threshold).astype(float))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _normal_consistency(self, normals_a, normals_b, pts_a, pts_b):
        tree_a = cKDTree(pts_a)
        tree_b = cKDTree(pts_b)
        _, idx_a = tree_b.query(pts_a)  # Find closest pts_b for each pt in pts_a
        _, idx_b = tree_a.query(pts_b)  # Find closest pts_a for each pt in pts_b

        # Average both directions:
        consistency_a = np.abs((normals_a * normals_b[idx_a]).sum(axis=1))
        consistency_b = np.abs((normals_b * normals_a[idx_b]).sum(axis=1))
        return (np.mean(consistency_a) + np.mean(consistency_b)) / 2.0

    def _calculate_codebleu(self, code_gt: str, code_pred: str) -> float:
        """
        Calculate CodeBLEU score between ground truth and predicted code.

        Args:
            code_gt: Ground truth code string
            code_pred: Predicted code string

        Returns:
            CodeBLEU score (float) or None if calculation fails
        """

        try:
            # Clean and normalize code strings
            code_gt_clean = self._normalize_code(code_gt)
            code_pred_clean = self._normalize_code(code_pred)

            # Calculate CodeBLEU using the library
            result = calc_codebleu(
                references=[code_gt_clean], predictions=[code_pred_clean], lang="python"
            )

            return result["codebleu"]

        except Exception as e:
            print(f"Error calculating CodeBLEU: {e}")
            return None

    def _normalize_code(self, code: str) -> str:
        """
        Normalize code by removing extra whitespace, comments, and standardizing formatting.

        Args:
            code: Raw code string

        Returns:
            Normalized code string
        """
        try:
            # Parse and unparse to normalize formatting
            tree = ast.parse(code)
            normalized = ast.unparse(tree)
            return normalized
        except Exception:
            # If parsing fails, do basic normalization
            lines = code.split("\n")
            # Remove empty lines and comments
            normalized_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    normalized_lines.append(line)
            return "\n".join(normalized_lines)

    def _calculate_manual_codebleu(self, code_gt: str, code_pred: str) -> dict:
        """
        Manual implementation of CodeBLEU-like metrics when the library is not available.
        This is a simplified version that focuses on:
        - Token-level BLEU
        - AST node similarity
        - Dataflow similarity (simplified)

        Returns:
            Dictionary with individual metric scores
        """
        try:
            # Tokenize code
            tokens_gt = self._tokenize_code(code_gt)
            tokens_pred = self._tokenize_code(code_pred)

            # Calculate n-gram BLEU scores
            bleu_scores = {}
            for n in range(1, 5):
                bleu_scores[f"bleu_{n}"] = self._calculate_ngram_bleu(
                    tokens_gt, tokens_pred, n
                )

            # Calculate AST similarity
            ast_similarity = self._calculate_ast_similarity(code_gt, code_pred)

            # Calculate dataflow similarity (simplified)
            dataflow_similarity = self._calculate_dataflow_similarity(
                code_gt, code_pred
            )

            # Weighted combination (CodeBLEU weights)
            codebleu_score = (
                0.25 * bleu_scores["bleu_4"]
                + 0.25 * ast_similarity
                + 0.25 * dataflow_similarity
                + 0.25 * bleu_scores["bleu_1"]
            )

            return {
                "codebleu": codebleu_score,
                "ast_similarity": ast_similarity,
                "dataflow_similarity": dataflow_similarity,
                **bleu_scores,
            }

        except Exception as e:
            print(f"Error in manual CodeBLEU calculation: {e}")
            return {"codebleu": 0.0}

    def _tokenize_code(self, code: str) -> list:
        """Tokenize Python code into meaningful tokens"""
        try:
            tree = ast.parse(code)
            tokens = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    tokens.append(node.id)
                elif isinstance(node, ast.Constant):
                    tokens.append(str(node.value))
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    tokens.append(node.name)
            return tokens
        except Exception:
            # Fallback to simple regex tokenization
            tokens = re.findall(r"\w+|\S", code)
            return [t for t in tokens if t.strip()]

    def _calculate_ngram_bleu(
        self, ref_tokens: list, pred_tokens: list, n: int
    ) -> float:
        """Calculate n-gram BLEU score"""
        if len(pred_tokens) < n:
            return 0.0

        ref_ngrams = Counter(
            [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
        )
        pred_ngrams = Counter(
            [tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)]
        )

        overlap = 0
        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                overlap += min(count, ref_ngrams[ngram])

        if len(pred_ngrams) == 0:
            return 0.0
        return overlap / len(pred_ngrams)

    def _calculate_ast_similarity(self, code_gt: str, code_pred: str) -> float:
        """Calculate AST node similarity"""
        try:
            tree_gt = ast.parse(code_gt)
            tree_pred = ast.parse(code_pred)

            nodes_gt = [type(node).__name__ for node in ast.walk(tree_gt)]
            nodes_pred = [type(node).__name__ for node in ast.walk(tree_pred)]

            nodes_gt_counter = Counter(nodes_gt)
            nodes_pred_counter = Counter(nodes_pred)

            # Calculate Jaccard similarity
            intersection = sum((nodes_gt_counter & nodes_pred_counter).values())
            union = sum((nodes_gt_counter | nodes_pred_counter).values())

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_dataflow_similarity(self, code_gt: str, code_pred: str) -> float:
        """Calculate simplified dataflow similarity based on variable usage patterns"""
        try:
            # Extract variable definitions and usages
            def extract_variables(code):
                tree = ast.parse(code)
                variables = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        variables.add(node.id)
                return variables

            vars_gt = extract_variables(code_gt)
            vars_pred = extract_variables(code_pred)

            # Calculate Jaccard similarity for variables
            intersection = len(vars_gt & vars_pred)
            union = len(vars_gt | vars_pred)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def visualize_point_clouds(
        self,
        pts1,
        pts2,
        normals1=None,
        normals2=None,
        title1="Ground Truth",
        title2="Prediction",
        show_normals=True,
        normal_scale=0.05,
    ):
        """
        Visualize two point clouds side by side with optional normals
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=[title1, title2],
            horizontal_spacing=0.05,
        )

        # Point cloud 1
        fig.add_trace(
            go.Scatter3d(
                x=pts1[:, 0],
                y=pts1[:, 1],
                z=pts1[:, 2],
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.7),
                name=f"{title1} Points",
                hovertemplate="<b>Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Point cloud 2
        fig.add_trace(
            go.Scatter3d(
                x=pts2[:, 0],
                y=pts2[:, 1],
                z=pts2[:, 2],
                mode="markers",
                marker=dict(size=3, color="red", opacity=0.7),
                name=f"{title2} Points",
                hovertemplate="<b>Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add normals if provided
        if show_normals and normals1 is not None:
            # Sample subset of normals for visualization (too many arrows clutters)
            sample_idx = np.random.choice(len(pts1), min(200, len(pts1)), replace=False)

            for i in sample_idx:
                pt = pts1[i]
                normal = normals1[i] * normal_scale
                fig.add_trace(
                    go.Scatter3d(
                        x=[pt[0], pt[0] + normal[0]],
                        y=[pt[1], pt[1] + normal[1]],
                        z=[pt[2], pt[2] + normal[2]],
                        mode="lines",
                        line=dict(color="lightblue", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

        if show_normals and normals2 is not None:
            sample_idx = np.random.choice(len(pts2), min(200, len(pts2)), replace=False)

            for i in sample_idx:
                pt = pts2[i]
                normal = normals2[i] * normal_scale
                fig.add_trace(
                    go.Scatter3d(
                        x=[pt[0], pt[0] + normal[0]],
                        y=[pt[1], pt[1] + normal[1]],
                        z=[pt[2], pt[2] + normal[2]],
                        mode="lines",
                        line=dict(color="lightcoral", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title="Point Cloud Comparison",
            height=600,
            scene1=dict(aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            scene2=dict(aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        )

        fig.show()
        return fig

    def visualize_overlay_comparison(
        self, pts1, pts2, normals1=None, normals2=None, title="Overlaid Point Clouds"
    ):
        """
        Overlay both point clouds in the same plot to see differences
        """
        fig = go.Figure()

        # Add first point cloud
        fig.add_trace(
            go.Scatter3d(
                x=pts1[:, 0],
                y=pts1[:, 1],
                z=pts1[:, 2],
                mode="markers",
                marker=dict(size=4, color="blue", opacity=0.6),
                name="First Sample",
                hovertemplate="<b>Sample 1</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
            )
        )

        # Add second point cloud
        fig.add_trace(
            go.Scatter3d(
                x=pts2[:, 0],
                y=pts2[:, 1],
                z=pts2[:, 2],
                mode="markers",
                marker=dict(size=4, color="red", opacity=0.6),
                name="Second Sample",
                hovertemplate="<b>Sample 2</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            height=600,
        )

        fig.show()
        return fig

    def visualize_normal_differences(
        self, pts1, pts2, normals1, normals2, max_points=1000
    ):
        """
        Visualize normal alignment quality between two point clouds
        """
        # Find correspondences
        from scipy.spatial import cKDTree

        tree = cKDTree(pts2)
        dists, idx = tree.query(pts1)

        # Calculate normal alignment (dot product)
        matched_normals = normals2[idx]
        dot_products = np.abs((normals1 * matched_normals).sum(axis=1))

        # Sample for visualization
        if len(pts1) > max_points:
            sample_idx = np.random.choice(len(pts1), max_points, replace=False)
            pts1_vis = pts1[sample_idx]
            dot_products_vis = dot_products[sample_idx]
            dists_vis = dists[sample_idx]
        else:
            pts1_vis = pts1
            dot_products_vis = dot_products
            dists_vis = dists

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=["Normal Alignment Quality", "Point Distance Error"],
        )

        # Color by normal alignment
        fig.add_trace(
            go.Scatter3d(
                x=pts1_vis[:, 0],
                y=pts1_vis[:, 1],
                z=pts1_vis[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=dot_products_vis,
                    colorscale="RdYlBu",
                    colorbar=dict(title="Normal Alignment", x=0.45),
                    cmin=0,
                    cmax=1,
                    opacity=0.8,
                ),
                name="Normal Alignment",
                hovertemplate="<b>Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<br>Alignment: %{marker.color:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Color by distance
        fig.add_trace(
            go.Scatter3d(
                x=pts1_vis[:, 0],
                y=pts1_vis[:, 1],
                z=pts1_vis[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=dists_vis,
                    colorscale="Viridis",
                    colorbar=dict(title="Distance Error", x=1.05),
                    opacity=0.8,
                ),
                name="Distance Error",
                hovertemplate="<b>Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<br>Distance: %{marker.color:.6f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Point Cloud Alignment Analysis",
            height=600,
            scene1=dict(aspectmode="cube"),
            scene2=dict(aspectmode="cube"),
        )

        # Print statistics
        print("Normal Alignment Statistics:")
        print(f"  Mean: {np.mean(dot_products):.4f}")
        print(f"  Min:  {np.min(dot_products):.4f}")
        print(f"  Max:  {np.max(dot_products):.4f}")
        print(f"  % > 0.9: {np.mean(dot_products > 0.9) * 100:.1f}%")
        print(f"  % < 0.5: {np.mean(dot_products < 0.5) * 100:.1f}%")

        print("\nDistance Statistics:")
        print(f"  Mean: {np.mean(dists):.6f}")
        print(f"  Max:  {np.max(dists):.6f}")
        print(f"  Std:  {np.std(dists):.6f}")

        fig.show()
        return fig

    @staticmethod
    def compute_mass_properties(
        shape: TopoDS_Shape,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        from OCP.BRepGProp import BRepGProp

        """Compute mass properties such as volume (interpreted as mass for unit density)
        and the center of mass of the given shape."""
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        mass = (
            props.Mass()
        )  # For solids, this gives the volume (mass = volume * density)
        center_of_mass = props.CentreOfMass()
        center_of_mass = np.array(
            [center_of_mass.X(), center_of_mass.Y(), center_of_mass.Z()]
        )
        matrix_of_inertia = props.MatrixOfInertia()
        matrix_of_inertia = np.array(
            [
                [
                    matrix_of_inertia.Value(1, 1),
                    matrix_of_inertia.Value(1, 2),
                    matrix_of_inertia.Value(1, 3),
                ],
                [
                    matrix_of_inertia.Value(2, 1),
                    matrix_of_inertia.Value(2, 2),
                    matrix_of_inertia.Value(2, 3),
                ],
                [
                    matrix_of_inertia.Value(3, 1),
                    matrix_of_inertia.Value(3, 2),
                    matrix_of_inertia.Value(3, 3),
                ],
            ]
        )
        return mass, center_of_mass, matrix_of_inertia

    def align_shapes(source: TopoDS_Shape, target: TopoDS_Shape) -> float:
        """
        Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU
        Code adapted from Cad-Coder (Doris et al., 2025)
        """

        intersection = BRepAlgoAPI_Common(source, target).Shape()
        union = BRepAlgoAPI_Fuse(source, target).Shape()

        if intersection == 0.0:
            return 0.0

        V_I, _, _ = CadEvaluator.compute_mass_properties(intersection)
        V_U, _, _ = CadEvaluator.compute_mass_properties(union)

        IOU = V_I / V_U

        print(f"Intersection Volume: {V_I}, Union Volume: {V_U}, IOU: {IOU}")

        return IOU
