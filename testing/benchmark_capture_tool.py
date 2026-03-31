#!/usr/bin/env python3
"""Interactive benchmark-capture tool for vesicle edge datasets.

This script helps build and maintain a benchmark dataset for vesicle edge
Detection work. It provides an interactive annotator for:

- selecting the vesicle center
- tracing the vesicle edge with clicks
- fitting a radial contour sampled on a fixed angular grid
- recording annotation metadata and review status
- exporting overlays for quick inspection

The tool stores one JSON annotation per image plus optional overlay PNGs.
It is intentionally simple, deterministic, and easy to extend.

It supports both ordinary image files and specific frames extracted from
multi-frame .nd2 stacks.

Examples
--------
Annotate ordinary image files:

python benchmark_capture_tool.py annotate \
    --images ./images \
    --output ./benchmark \
    --ext .png .tif

Annotate all frames from ND2 files in a directory:

python benchmark_capture_tool.py annotate \
    --images ./stacks \
    --output ./benchmark \
    --nd2-mode all

Annotate selected ND2 frames only:

python benchmark_capture_tool.py annotate \
    --images ./stacks \
    --output ./benchmark \
    --nd2-mode selected \
    --nd2-frames 0 5 10 25

Annotate ND2 frames listed in a manifest CSV with columns path,frame:

python benchmark_capture_tool.py annotate \
    --images ./stacks \
    --output ./benchmark \
    --nd2-manifest ./frames.csv

Regenerate overlays:

python benchmark_capture_tool.py export-overlays \
    --images ./images \
    --output ./benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:  # pragma: no cover
    tk = None
    messagebox = None

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install it with: pip install pillow"
    ) from exc

try:
    import nd2
except ImportError:  # pragma: no cover
    nd2 = None


DEFAULT_NUM_ANGLES = 360
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Disable Matplotlib's built-in "save figure" keyboard shortcut so that the
# annotator can safely use the plain "s" key for saving annotation JSON.
mpl.rcParams["keymap.save"] = []


@dataclass(slots=True)
class AnnotationMetadata:
    """Metadata recorded alongside a vesicle annotation."""

    annotator: str = ""
    reviewer: str = ""
    notes: str = ""
    status: str = "draft"
    source_image: str = ""
    source_frame: int | None = None
    created_utc: str = ""
    updated_utc: str = ""
    image_shape: tuple[int, int] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VesicleAnnotation:
    """Serializable vesicle annotation.

    Attributes
    ----------
    schema_version:
        Version for the on-disk JSON format.
    image_id:
        Stable identifier for the image or image frame.
    center_xy:
        Annotated center in image pixel coordinates as [x, y].
    edge_points_xy:
        Raw clicked edge points in image pixel coordinates.
    theta_deg:
        Angular samples in degrees for the fitted radial contour.
    radius_px:
        Radius values sampled on ``theta_deg``.
    contour_xy:
        Reconstructed contour in image pixel coordinates sampled on the
        fixed angular grid.
    metadata:
        Freeform bookkeeping and review metadata.
    """

    schema_version: int
    image_id: str
    center_xy: list[float] | None
    edge_points_xy: list[list[float]]
    theta_deg: list[float]
    radius_px: list[float]
    contour_xy: list[list[float]]
    metadata: AnnotationMetadata


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Address of an annotatable image source.

    For ordinary image files, ``frame_index`` is ``None``.
    For ND2 stacks, ``frame_index`` identifies the specific time/frame slice.
    """

    path: Path
    frame_index: int | None = None

    @property
    def image_id(self) -> str:
        """Return a stable on-disk identifier for this source."""
        if self.frame_index is None:
            return self.path.stem.replace(" ", "_")
        stem = self.path.stem.replace(" ", "_")
        return f"{stem}__frame_{self.frame_index:05d}"

    @property
    def display_name(self) -> str:
        """Return a human-readable name for UI display."""
        if self.frame_index is None:
            return self.path.name
        return f"{self.path.name} [frame {self.frame_index}]"


class AnnotationStore:
    """Load and save benchmark artifacts in a benchmark directory."""

    INDEX_COLUMNS = [
        "image_id",
        "source_image",
        "source_frame",
        "annotation_path",
        "overlay_path",
        "test_image_path",
        "status",
        "annotator",
        "reviewer",
        "updated_utc",
    ]

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.annotations_dir = output_dir / "annotations"
        self.overlays_dir = output_dir / "overlays"
        self.test_images_dir = output_dir / "test_images"
        self.index_path = output_dir / "index.csv"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.test_images_dir.mkdir(parents=True, exist_ok=True)

    def annotation_path(self, record: FrameRecord) -> Path:
        """Return the JSON path for an image or image-frame annotation."""
        return self.annotations_dir / f"{record.image_id}.json"

    def overlay_path(self, record: FrameRecord) -> Path:
        """Return the overlay PNG path for an image or image-frame annotation."""
        return self.overlays_dir / f"{record.image_id}.png"

    def test_image_path(self, record: FrameRecord) -> Path:
        """Return the path for saving the raw test image."""
        return self.test_images_dir / f"{record.image_id}.png"

    def load(self, record: FrameRecord) -> VesicleAnnotation | None:
        """Load an existing annotation, if present."""
        path = self.annotation_path(record)
        if not path.exists():
            return None

        payload = json.loads(path.read_text())
        metadata = AnnotationMetadata(**payload["metadata"])
        return VesicleAnnotation(
            schema_version=payload["schema_version"],
            image_id=payload["image_id"],
            center_xy=payload["center_xy"],
            edge_points_xy=payload["edge_points_xy"],
            theta_deg=payload["theta_deg"],
            radius_px=payload["radius_px"],
            contour_xy=payload["contour_xy"],
            metadata=metadata,
        )

    def save(self, record: FrameRecord, annotation: VesicleAnnotation) -> Path:
        """Write an annotation JSON file and return its path."""
        path = self.annotation_path(record)
        payload = asdict(annotation)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def update_index(
        self,
        *,
        record: FrameRecord,
        annotation: VesicleAnnotation,
        annotation_path: Path,
        overlay_path: Path | None,
        test_image_path: Path | None,
    ) -> Path:
        """Upsert one row in the benchmark index for the current annotated frame."""
        rows_by_id: dict[str, dict[str, str]] = {}

        if self.index_path.exists():
            with self.index_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    image_id = row.get("image_id", "")
                    if image_id:
                        rows_by_id[image_id] = {
                            column: row.get(column, "") for column in self.INDEX_COLUMNS
                        }

        rows_by_id[record.image_id] = {
            "image_id": record.image_id,
            "source_image": str(record.path),
            "source_frame": "" if record.frame_index is None else str(record.frame_index),
            "annotation_path": os.path.relpath(annotation_path, self.output_dir),
            "overlay_path": "" if overlay_path is None else os.path.relpath(overlay_path, self.output_dir),
            "test_image_path": "" if test_image_path is None else os.path.relpath(test_image_path, self.output_dir),
            "status": annotation.metadata.status,
            "annotator": annotation.metadata.annotator,
            "reviewer": annotation.metadata.reviewer,
            "updated_utc": annotation.metadata.updated_utc,
        }

        sorted_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
        with self.index_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(sorted_rows)

        return self.index_path


class ImageCatalog:
    """Discover annotatable ordinary images and ND2 frames."""

    def __init__(
        self,
        root: Path,
        extensions: Sequence[str],
        include_nd2: bool,
        nd2_mode: str,
        nd2_frames: Sequence[int] | None,
        nd2_manifest: Path | None,
    ) -> None:
        self.root = root
        self.records = self._discover_records(
            root=root,
            extensions=extensions,
            include_nd2=include_nd2,
            nd2_mode=nd2_mode,
            nd2_frames=nd2_frames,
            nd2_manifest=nd2_manifest,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameRecord:
        return self.records[index]

    @staticmethod
    def _discover_records(
        root: Path,
        extensions: Sequence[str],
        include_nd2: bool,
        nd2_mode: str,
        nd2_frames: Sequence[int] | None,
        nd2_manifest: Path | None,
    ) -> list[FrameRecord]:
        normalized_exts = {ext.lower() for ext in extensions}
        records: list[FrameRecord] = []

        if nd2_manifest is not None:
            records.extend(load_nd2_manifest(nd2_manifest, root if root.is_dir() else root.parent))

        if root.is_file():
            path = root
            suffix = path.suffix.lower()

            if suffix == ".nd2":
                if not include_nd2:
                    return []
                records.extend(
                    discover_nd2_records(
                        nd2_path=path,
                        nd2_mode=nd2_mode,
                        nd2_frames=nd2_frames,
                    )
                )
                return sorted(records, key=lambda record: (str(record.path), record.frame_index or -1))

            if suffix in normalized_exts:
                records.append(FrameRecord(path=path, frame_index=None))
                return records

            return []

        ordinary_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_exts
        )
        records.extend(FrameRecord(path=path, frame_index=None) for path in ordinary_paths)

        if include_nd2 and nd2_manifest is None:
            nd2_paths = sorted(
                path for path in root.rglob("*.nd2") if path.is_file()
            )
            for nd2_path in nd2_paths:
                records.extend(
                    discover_nd2_records(
                        nd2_path=nd2_path,
                        nd2_mode=nd2_mode,
                        nd2_frames=nd2_frames,
                    )
                )

        deduped = sorted(set(records), key=lambda record: (str(record.path), record.frame_index or -1))
        return deduped


class ContourModel:
    """Utilities for converting sparse clicked points into a radial contour."""

    @staticmethod
    def fit_radial_contour(
        center_xy: np.ndarray,
        edge_points_xy: np.ndarray,
        num_angles: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a contour sampled on a fixed angular grid."""
        if edge_points_xy.shape[0] < 3:
            raise ValueError("At least 3 edge points are required to fit a contour.")

        dx = edge_points_xy[:, 0] - center_xy[0]
        dy = edge_points_xy[:, 1] - center_xy[1]
        theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        radius = np.hypot(dx, dy)

        order = np.argsort(theta)
        theta_sorted = theta[order]
        radius_sorted = radius[order]

        theta_extended = np.concatenate(
            [theta_sorted - 2.0 * np.pi, theta_sorted, theta_sorted + 2.0 * np.pi]
        )
        radius_extended = np.tile(radius_sorted, 3)

        theta_grid = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        radius_grid = np.interp(theta_grid, theta_extended, radius_extended)

        contour_x = center_xy[0] + radius_grid * np.cos(theta_grid)
        contour_y = center_xy[1] + radius_grid * np.sin(theta_grid)
        contour_xy = np.column_stack([contour_x, contour_y])
        return theta_grid, radius_grid, contour_xy

    @staticmethod
    def polygon_centroid(contour_xy: np.ndarray) -> np.ndarray:
        """Return the area centroid of a closed contour polygon.

        The centroid is computed from the fitted contour rather than trusting
        the user's initial center click as final ground truth. If the polygon is
        degenerate, the arithmetic mean of the contour vertices is used as a
        fallback.
        """
        if contour_xy.shape[0] < 3:
            raise ValueError("At least 3 contour points are required to compute a centroid.")

        x = contour_xy[:, 0]
        y = contour_xy[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        cross = x * y_next - x_next * y
        twice_area = np.sum(cross)

        if np.isclose(twice_area, 0.0):
            return np.mean(contour_xy, axis=0)

        cx = np.sum((x + x_next) * cross) / (3.0 * twice_area)
        cy = np.sum((y + y_next) * cross) / (3.0 * twice_area)
        return np.array([cx, cy], dtype=float)


class BenchmarkAnnotator:
    """Interactive matplotlib-based benchmark annotator."""

    def __init__(
        self,
        records: Sequence[FrameRecord],
        store: AnnotationStore,
        annotator: str,
        num_angles: int,
        autosave_overlay: bool,
    ) -> None:
        self.records = list(records)
        self.store = store
        self.annotator = annotator
        self.num_angles = num_angles
        self.autosave_overlay = autosave_overlay

        self.index = 0
        self.image_array: np.ndarray | None = None
        self.current_record: FrameRecord | None = None
        self.center_xy: np.ndarray | None = None
        self.edge_points_xy: list[list[float]] = []
        self.annotation: VesicleAnnotation | None = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Vesicle Benchmark Annotator")

        self.status_text = self.fig.text(0.01, 0.01, "", fontsize=10)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def run(self) -> None:
        """Start the interactive session."""
        if not self.records:
            raise SystemExit("No matching images or ND2 frames were found.")

        self.load_record(0)
        plt.show()

    def load_record(self, index: int) -> None:
        """Load an image source and any existing annotation into the editor."""
        self.index = max(0, min(index, len(self.records) - 1))
        self.current_record = self.records[self.index]
        self.image_array = load_frame_array(self.current_record)

        self.center_xy = None
        self.edge_points_xy = []
        self.annotation = self.store.load(self.current_record)

        if self.annotation is not None:
            if self.annotation.center_xy is not None:
                self.center_xy = np.asarray(self.annotation.center_xy, dtype=float)
            self.edge_points_xy = [list(point) for point in self.annotation.edge_points_xy]

        self.redraw()
        self.update_status(
            f"[{self.index + 1}/{len(self.records)}] {self.current_record.display_name}"
        )

    def redraw(self) -> None:
        """Refresh the image and annotation overlays."""
        assert self.current_record is not None
        assert self.image_array is not None

        self.ax.clear()
        self.ax.imshow(self.image_array, cmap="gray")
        self.ax.set_title(self.current_record.display_name)
        self.ax.set_axis_off()

        if self.center_xy is not None:
            self.ax.plot(
                self.center_xy[0],
                self.center_xy[1],
                marker="+",
                markersize=14,
                markeredgewidth=2.0,
            )

        if self.edge_points_xy:
            pts = np.asarray(self.edge_points_xy, dtype=float)
            self.ax.plot(pts[:, 0], pts[:, 1], linestyle="", marker="o", markersize=3)

        if self.center_xy is not None and len(self.edge_points_xy) >= 3:
            _, _, contour_xy = ContourModel.fit_radial_contour(
                self.center_xy,
                np.asarray(self.edge_points_xy, dtype=float),
                self.num_angles,
            )
            refined_center_xy = ContourModel.polygon_centroid(contour_xy)
            closed_xy = np.vstack([contour_xy, contour_xy[0]])
            self.ax.plot(closed_xy[:, 0], closed_xy[:, 1], linewidth=1.5)
            self.ax.plot(
                refined_center_xy[0],
                refined_center_xy[1],
                marker="x",
                markersize=10,
                markeredgewidth=2.0,
            )

        self.fig.canvas.draw_idle()

    def update_status(self, message: str) -> None:
        """Update the status line shown below the figure."""
        self.status_text.set_text(
            message
            + " | left click: center/edge, right click: undo point, "
            + "s: save, n/p: next/prev, c: clear, r: reset center, o: save overlay"
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event: MouseEvent) -> None:
        """Handle mouse clicks for center selection and edge tracing."""
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        if event.button == MouseButton.LEFT:
            if self.center_xy is None:
                self.center_xy = np.array([event.xdata, event.ydata], dtype=float)
                self.update_status("Center selected.")
            else:
                self.edge_points_xy.append([float(event.xdata), float(event.ydata)])
                self.update_status(f"Added edge point {len(self.edge_points_xy)}.")
            self.redraw()
            return

        if event.button == MouseButton.RIGHT and self.edge_points_xy:
            self.edge_points_xy.pop()
            self.update_status("Removed last edge point.")
            self.redraw()

    def on_key_press(self, event: KeyEvent) -> None:
        """Handle keyboard shortcuts."""
        key = (event.key or "").lower()
        try:
            if key == "s":
                assert self.current_record is not None
                needs_overwrite_confirmation = any(
                    path.exists()
                    for path in (
                        self.store.annotation_path(self.current_record),
                        self.store.overlay_path(self.current_record),
                        self.store.test_image_path(self.current_record),
                    )
                )
                overwrite_ok = True
                if needs_overwrite_confirmation:
                    overwrite_ok = prompt_yes_no(
                        f"Overwrite existing benchmark files for {self.current_record.image_id}?"
                    )
                if overwrite_ok:
                    ann_path = self.save_current_annotation(confirm_overwrite=False)
                    if ann_path is not None:
                        print(f"Saved annotation: {ann_path}")
                        overlay_path = self.save_overlay(confirm_overwrite=False)
                        if overlay_path is not None:
                            print(f"Saved overlay: {overlay_path}")
                        test_img_path = self.save_test_image(confirm_overwrite=False)
                        if test_img_path is not None:
                            print(f"Saved test image: {test_img_path}")
                        assert self.annotation is not None
                        index_path = self.store.update_index(
                            record=self.current_record,
                            annotation=self.annotation,
                            annotation_path=ann_path,
                            overlay_path=overlay_path,
                            test_image_path=test_img_path,
                        )
                        print(f"Updated benchmark index: {index_path}")
                else:
                    self.update_status("Save cancelled.")
            elif key == "o":
                overlay_path = self.save_overlay()
                if overlay_path is not None:
                    print(f"Saved overlay: {overlay_path}")
            elif key == "n":
                self.next_record()
            elif key == "p":
                self.previous_record()
            elif key == "c":
                self.edge_points_xy = []
                self.update_status("Cleared edge points.")
                self.redraw()
            elif key == "r":
                self.center_xy = None
                self.edge_points_xy = []
                self.update_status("Reset center and edge points.")
                self.redraw()
            elif key == "q":
                plt.close(self.fig)
        except Exception as exc:
            self.update_status(f"Error: {exc}")
            print(f"Error handling key {key!r}: {exc}")

    def save_current_annotation(self, confirm_overwrite: bool = True) -> Path | None:
        """Save the current annotation to disk.

        If an annotation already exists for the current source, prompt before
        overwriting it unless confirmation has already been granted for the
        current save operation.
        """
        if self.current_record is None or self.image_array is None:
            raise RuntimeError("No image is currently loaded.")

        if self.center_xy is None:
            raise RuntimeError("Cannot save without a selected center.")

        if len(self.edge_points_xy) < 3:
            raise RuntimeError("Cannot save without at least 3 edge points.")

        edge_points = np.asarray(self.edge_points_xy, dtype=float)
        theta_grid, radius_grid, contour_xy = ContourModel.fit_radial_contour(
            self.center_xy,
            edge_points,
            self.num_angles,
        )
        refined_center_xy = ContourModel.polygon_centroid(contour_xy)

        # Re-fit the contour around the refined centroid so that the saved
        # radial representation is consistent with the stored ground-truth
        # center rather than the user's initial click.
        theta_grid, radius_grid, contour_xy = ContourModel.fit_radial_contour(
            refined_center_xy,
            edge_points,
            self.num_angles,
        )

        now = utc_timestamp()
        existing = self.store.load(self.current_record)
        created_utc = now
        reviewer = ""
        notes = ""
        tags: list[str] = []
        status = "draft"

        if existing is not None:
            annotation_path = self.store.annotation_path(self.current_record)
            if confirm_overwrite and not prompt_yes_no(f"Overwrite existing benchmark files for {self.current_record.image_id}?"):
                self.update_status("Save cancelled.")
                return None
            created_utc = existing.metadata.created_utc or now
            reviewer = existing.metadata.reviewer
            notes = existing.metadata.notes
            tags = list(existing.metadata.tags)
            status = existing.metadata.status

        annotation = VesicleAnnotation(
            schema_version=1,
            image_id=self.current_record.image_id,
            center_xy=refined_center_xy.astype(float).tolist(),
            edge_points_xy=edge_points.astype(float).tolist(),
            theta_deg=np.rad2deg(theta_grid).astype(float).tolist(),
            radius_px=radius_grid.astype(float).tolist(),
            contour_xy=contour_xy.astype(float).tolist(),
            metadata=AnnotationMetadata(
                annotator=self.annotator,
                reviewer=reviewer,
                notes=notes,
                status=status,
                source_image=str(self.current_record.path),
                source_frame=self.current_record.frame_index,
                created_utc=created_utc,
                updated_utc=now,
                image_shape=(int(self.image_array.shape[0]), int(self.image_array.shape[1])),
                tags=tags,
            ),
        )

        path = self.store.save(self.current_record, annotation)
        self.annotation = annotation
        self.center_xy = refined_center_xy

        self.update_status(f"Saved annotation to {path}.")
        return path

    def save_overlay(self, confirm_overwrite: bool = True) -> Path | None:
        """Write an overlay PNG for the current image.

        If an overlay already exists for the current source, prompt before
        overwriting it unless confirmation has been disabled.
        """
        if self.current_record is None:
            raise RuntimeError("No image is currently loaded.")

        output_path = self.store.overlay_path(self.current_record)
        if confirm_overwrite and output_path.exists():
            if not prompt_yes_no(f"Overwrite existing benchmark files for {self.current_record.image_id}?"):
                self.update_status("Save cancelled.")
                return None

        self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
        self.update_status(f"Saved overlay to {output_path}.")
        return output_path

    def save_test_image(self, confirm_overwrite: bool = True) -> Path | None:
        if self.current_record is None or self.image_array is None:
            raise RuntimeError("No image is currently loaded.")

        output_path = self.store.test_image_path(self.current_record)

        if confirm_overwrite and output_path.exists():
            if not prompt_yes_no(f"Overwrite existing benchmark files for {self.current_record.image_id}?"):
                self.update_status("Save cancelled.")
                return None

        from PIL import Image
        arr = self.image_array
        if arr.dtype != np.uint8:
            arr_to_save = arr.astype(np.uint16)
        else:
            arr_to_save = arr
        Image.fromarray(arr_to_save).save(output_path)

        self.update_status(f"Saved test image to {output_path}.")
        return output_path

    def next_record(self) -> None:
        """Move to the next image source."""
        if self.index < len(self.records) - 1:
            self.load_record(self.index + 1)

    def previous_record(self) -> None:
        """Move to the previous image source."""
        if self.index > 0:
            self.load_record(self.index - 1)


def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prompt_yes_no(prompt: str) -> bool:
    """Prompt for a yes/no answer.

    Prefer a GUI popup when Tk is available. Fall back to a terminal prompt
    otherwise. The default answer is "no".
    """
    if messagebox is not None and tk is not None:
        root = tk.Tk()
        root.withdraw()
        try:
            return bool(messagebox.askyesno("Confirm overwrite", prompt, parent=root))
        finally:
            root.destroy()

    while True:
        response = input(f"{prompt} [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def require_nd2() -> None:
    """Raise a clear error if ND2 support is requested but unavailable."""
    if nd2 is None:
        raise SystemExit(
            "ND2 support requires the 'nd2' package. Install it with: pip install nd2"
        )


def load_image_array(path: Path) -> np.ndarray:
    """Load a regular image file into a numpy array."""
    image = Image.open(path)
    if image.mode not in ("L", "I;16", "I"):
        image = image.convert("L")
    return np.asarray(image, dtype=np.float32)


def read_nd2_frame(path: Path, frame_index: int) -> np.ndarray:
    """Load a particular frame from an ND2 file.

    This implementation assumes frame selection along the leading axis returned
    by ``nd2.imread`` / ``ND2File.asarray``. For more complex ND2 datasets with
    multiple non-spatial axes, you will likely want to expand this to support an
    explicit axis mapping such as t/z/c.
    """
    require_nd2()
    with nd2.ND2File(path) as stack:
        data = stack.asarray()

    if data.ndim < 2:
        raise ValueError(f"Unexpected ND2 dimensionality for {path}: {data.shape}")

    if data.ndim == 2:
        if frame_index != 0:
            raise IndexError(f"Frame {frame_index} is out of range for single-frame ND2 {path}.")
        frame = data
    else:
        n_frames = data.shape[0]
        if not 0 <= frame_index < n_frames:
            raise IndexError(
                f"Frame {frame_index} is out of range for {path} with {n_frames} frames."
            )
        frame = data[frame_index]
        while frame.ndim > 2:
            frame = frame[0]

    return np.asarray(frame, dtype=np.float32)


def load_frame_array(record: FrameRecord) -> np.ndarray:
    """Load either a normal image or a specific ND2 frame."""
    if record.frame_index is None:
        return load_image_array(record.path)
    return read_nd2_frame(record.path, record.frame_index)


def discover_nd2_records(
    nd2_path: Path,
    nd2_mode: str,
    nd2_frames: Sequence[int] | None,
) -> list[FrameRecord]:
    """Return annotatable frame records for one ND2 file."""
    require_nd2()

    if nd2_mode == "none":
        return []

    with nd2.ND2File(nd2_path) as stack:
        shape = stack.shape

    if len(shape) < 2:
        raise ValueError(f"Unexpected ND2 dimensionality for {nd2_path}: {shape}")

    n_frames = 1 if len(shape) == 2 else int(shape[0])

    if nd2_mode == "first":
        return [FrameRecord(path=nd2_path, frame_index=0)]

    if nd2_mode == "all":
        return [FrameRecord(path=nd2_path, frame_index=i) for i in range(n_frames)]

    if nd2_mode == "selected":
        if not nd2_frames:
            raise SystemExit("--nd2-mode selected requires --nd2-frames.")
        bad = [frame for frame in nd2_frames if frame < 0 or frame >= n_frames]
        if bad:
            raise SystemExit(
                f"Requested ND2 frames out of range for {nd2_path.name}: {bad}; valid range is 0..{n_frames - 1}."
            )
        return [FrameRecord(path=nd2_path, frame_index=frame) for frame in nd2_frames]

    raise RuntimeError(f"Unhandled ND2 mode: {nd2_mode}")


def load_nd2_manifest(manifest_path: Path, root: Path) -> list[FrameRecord]:
    """Load explicit ND2 path/frame pairs from a CSV manifest.

    The CSV must contain columns named ``path`` and ``frame``. Paths may be
    absolute or relative to ``root``.
    """
    require_nd2()
    records: list[FrameRecord] = []

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "frame"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit("ND2 manifest must contain columns: path, frame")

        for row in reader:
            raw_path = row["path"].strip()
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            records.append(FrameRecord(path=path, frame_index=int(row["frame"])))

    return records


def iter_annotation_files(annotations_dir: Path) -> Iterable[Path]:
    """Yield annotation JSON files in sorted order."""
    return sorted(annotations_dir.glob("*.json"))


def export_overlays(images_dir: Path, output_dir: Path) -> None:
    """Regenerate overlay images from saved annotations."""
    store = AnnotationStore(output_dir)
    annotations_dir = store.annotations_dir

    for annotation_path in iter_annotation_files(annotations_dir):
        payload = json.loads(annotation_path.read_text())
        source_image = Path(payload["metadata"]["source_image"])
        source_frame = payload["metadata"].get("source_frame")
        record = FrameRecord(path=source_image, frame_index=source_frame)

        if not source_image.exists() and source_frame is None:
            candidate = images_dir / source_image.name
            if candidate.exists():
                record = FrameRecord(path=candidate, frame_index=None)

        image_array = load_frame_array(record)
        center_xy = np.asarray(payload["center_xy"], dtype=float)
        edge_points_xy = np.asarray(payload["edge_points_xy"], dtype=float)
        contour_xy = np.asarray(payload["contour_xy"], dtype=float)
        closed_xy = np.vstack([contour_xy, contour_xy[0]])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_array, cmap="gray")
        ax.set_title(record.display_name)
        ax.set_axis_off()
        ax.plot(center_xy[0], center_xy[1], marker="+", markersize=14, markeredgewidth=2)
        ax.plot(edge_points_xy[:, 0], edge_points_xy[:, 1], linestyle="", marker="o", markersize=3)
        ax.plot(closed_xy[:, 0], closed_xy[:, 1], linewidth=1.5)

        overlay_path = store.overlay_path(record)
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {overlay_path}")


def summarize_annotations(output_dir: Path) -> None:
    """Print a simple summary of benchmark-capture progress."""
    store = AnnotationStore(output_dir)
    rows: list[dict[str, Any]] = []

    for annotation_path in iter_annotation_files(store.annotations_dir):
        payload = json.loads(annotation_path.read_text())
        rows.append(
            {
                "image_id": payload["image_id"],
                "status": payload["metadata"]["status"],
                "annotator": payload["metadata"]["annotator"],
                "reviewer": payload["metadata"]["reviewer"],
                "source_frame": payload["metadata"].get("source_frame"),
                "n_edge_points": len(payload["edge_points_xy"]),
                "n_angles": len(payload["theta_deg"]),
                "updated_utc": payload["metadata"]["updated_utc"],
            }
        )

    if not rows:
        print("No annotations found.")
        return

    print(f"Annotations: {len(rows)}")
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    print("Most recently updated:")
    rows_sorted = sorted(rows, key=lambda row: row["updated_utc"], reverse=True)
    for row in rows_sorted[:10]:
        print(
            f"  {row['image_id']}: frame={row['source_frame']}, status={row['status']}, "
            f"points={row['n_edge_points']}, updated={row['updated_utc']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive tool for capturing vesicle benchmark annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Open the interactive annotation UI.",
    )
    annotate_parser.add_argument("--images", type=Path, required=True, help="Image root directory.")
    annotate_parser.add_argument("--output", type=Path, required=True, help="Benchmark output directory.")
    annotate_parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Ordinary image file extensions to include.",
    )
    annotate_parser.add_argument(
        "--annotator",
        default="",
        help="Annotator name stored in annotation metadata.",
    )
    annotate_parser.add_argument(
        "--num-angles",
        type=int,
        default=DEFAULT_NUM_ANGLES,
        help="Number of angular samples in the radial contour representation.",
    )
    annotate_parser.add_argument(
        "--autosave-overlay",
        action="store_true",
        help="Also save an overlay PNG whenever an annotation is saved.",
    )
    annotate_parser.add_argument(
        "--include-nd2",
        action="store_true",
        help="Include .nd2 files discovered under --images.",
    )
    annotate_parser.add_argument(
        "--nd2-mode",
        choices=("none", "first", "all", "selected"),
        default="first",
        help=(
            "Which frames to expose from each discovered .nd2 file: none, first, all, or selected. "
            "Ignored unless --include-nd2 is set or --nd2-manifest is provided."
        ),
    )
    annotate_parser.add_argument(
        "--nd2-frames",
        nargs="+",
        type=int,
        default=None,
        help="Specific frame indices to use when --nd2-mode selected is chosen.",
    )
    annotate_parser.add_argument(
        "--nd2-manifest",
        type=Path,
        default=None,
        help="CSV manifest of explicit ND2 path/frame pairs with columns path,frame.",
    )

    overlay_parser = subparsers.add_parser(
        "export-overlays",
        help="Regenerate overlay PNGs from saved annotations.",
    )
    overlay_parser.add_argument("--images", type=Path, required=True, help="Image root directory.")
    overlay_parser.add_argument("--output", type=Path, required=True, help="Benchmark output directory.")

    summary_parser = subparsers.add_parser(
        "summary",
        help="Print a summary of existing annotations.",
    )
    summary_parser.add_argument("--output", type=Path, required=True, help="Benchmark output directory.")

    return parser.parse_args()


def main() -> None:
    """Entry point for the command-line interface."""
    args = parse_args()

    if args.command == "annotate":
        inferred_include_nd2 = (
            args.include_nd2
            or args.nd2_manifest is not None
            or args.images.suffix.lower() == ".nd2"
            or args.nd2_mode != "none"
        )
        catalog = ImageCatalog(
            root=args.images,
            extensions=args.ext,
            include_nd2=inferred_include_nd2,
            nd2_mode=args.nd2_mode,
            nd2_frames=args.nd2_frames,
            nd2_manifest=args.nd2_manifest,
        )
        store = AnnotationStore(args.output)
        annotator = BenchmarkAnnotator(
            records=catalog.records,
            store=store,
            annotator=args.annotator,
            num_angles=args.num_angles,
            autosave_overlay=args.autosave_overlay,
        )
        annotator.run()
        return

    if args.command == "export-overlays":
        export_overlays(args.images, args.output)
        return

    if args.command == "summary":
        summarize_annotations(args.output)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
