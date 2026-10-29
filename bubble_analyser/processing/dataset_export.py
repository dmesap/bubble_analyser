"""Export corrected bubble segmentations as a dataset in a user-chosen format.

Every format starts from the same per-bubble geometry: the manual polygon when a
bubble has one, otherwise a polygon sampled from its fitted ellipse. That geometry
lives in the processed (possibly resampled) image, so it is scaled to the original
image before anything is written.

Formats
-------
``yolo_seg``   Ultralytics YOLO segmentation: ``labels/*.txt`` with normalised polygons.
``yolo_det``   Ultralytics YOLO detection: ``labels/*.txt`` with normalised boxes.
``coco``       COCO instance segmentation: one ``annotations.json`` (pixel polygons).
``voc``        Pascal VOC: ``Annotations/*.xml`` (pixel boxes, no masks).
``masks_npy``  Integer label images (``.npy``), as exported by earlier versions.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import numpy.typing as npt

Ellipse = tuple[tuple[float, float], tuple[float, float], float]
Polygon = Sequence[tuple[float, float]]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DatasetFormat:
    """A selectable export format."""

    key: str
    label: str
    description: str


DATASET_FORMATS: tuple[DatasetFormat, ...] = (
    DatasetFormat(
        "yolo_seg",
        "YOLO segmentation",
        "images/ + labels/*.txt (one line per bubble: class x1 y1 x2 y2 ..., normalised) + data.yaml. "
        "For training Ultralytics YOLO segmentation models.",
    ),
    DatasetFormat(
        "yolo_det",
        "YOLO detection (boxes)",
        "images/ + labels/*.txt (class x_center y_center width height, normalised) + data.yaml. "
        "Bounding boxes only, no masks.",
    ),
    DatasetFormat(
        "coco",
        "COCO instance segmentation",
        "images/ + annotations.json (pixel polygons, boxes and areas). "
        "For Detectron2, MMDetection, pycocotools and similar tools.",
    ),
    DatasetFormat(
        "voc",
        "Pascal VOC (boxes)",
        "JPEGImages/ + Annotations/*.xml (pixel bounding boxes) + ImageSets/Main/trainval.txt. "
        "Bounding boxes only, no masks.",
    ),
    DatasetFormat(
        "masks_npy",
        "Masks (.npy), as in earlier versions",
        "images/ + masks/*_mask.npy (integer label image at the processed resolution, one value per bubble).",
    ),
)

DEFAULT_FORMAT_KEY = "yolo_seg"
_FORMAT_KEYS = tuple(item.key for item in DATASET_FORMATS)


def get_format(key: str) -> DatasetFormat:
    """Return the format registered under ``key``."""
    for item in DATASET_FORMATS:
        if item.key == key:
            return item
    raise ValueError(f"Unknown export format '{key}'. Available formats: {', '.join(_FORMAT_KEYS)}.")


@dataclass(frozen=True)
class FrameAnnotation:
    """The corrected segmentation of one image.

    Attributes:
        image_path: Original image on disk.
        ellipses: Fitted ellipses as ``((cx, cy), (width, height), angle)``, in processed-image pixels.
        polygons: Optional manual polygon per ellipse (same order); ``None`` keeps the ellipse.
        annotation_shape: ``(height, width)`` of the processed image the geometry belongs to.
            ``None`` means the geometry is already in original-image pixels.
        labelled_mask: Integer label image, only needed by the ``masks_npy`` format.
    """

    image_path: Path
    ellipses: Sequence[Ellipse]
    polygons: Sequence[Polygon | None] = ()
    annotation_shape: tuple[int, int] | None = None
    labelled_mask: npt.NDArray[np.int_] | None = None


@dataclass
class ExportSummary:
    """What an export wrote and what it left out."""

    format_key: str
    output_root: Path
    frames_exported: int = 0
    instances_exported: int = 0
    frames_without_bubbles: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _PreparedFrame:
    frame: FrameAnnotation
    size: tuple[int, int]  # (height, width) of the original image
    polygons: list[npt.NDArray[np.float64]]  # pixel coordinates in the original image


def ellipse_to_polygon(ellipse: Ellipse, points: int | None = None) -> npt.NDArray[np.float64]:
    """Sample an ellipse as a closed polygon in float pixel coordinates.

    ``ellipse`` follows the OpenCV convention: full axis lengths and a rotation angle in degrees.
    Small bubbles get fewer points than large ones, so tiny masks are not oversampled.
    """
    (cx, cy), (width, height), angle = ellipse
    half_a = max(float(width) / 2.0, 0.5)
    half_b = max(float(height) / 2.0, 0.5)
    if points is None:
        points = int(np.clip(round(math.pi * (half_a + half_b) / 3.0), 16, 72))
    theta = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    rotation = math.radians(float(angle))
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    x = half_a * np.cos(theta)
    y = half_b * np.sin(theta)
    return np.column_stack((float(cx) + x * cos_r - y * sin_r, float(cy) + x * sin_r + y * cos_r))


def bubble_polygons(frame: FrameAnnotation, output_shape: tuple[int, int]) -> list[npt.NDArray[np.float64]]:
    """Return one polygon per bubble, scaled to ``output_shape`` and clipped to the image.

    Bubbles whose polygon collapses to fewer than three points or to zero area are dropped.
    """
    out_height, out_width = output_shape
    if frame.annotation_shape is None:
        scale_x = scale_y = 1.0
    else:
        annotation_height, annotation_width = frame.annotation_shape
        if annotation_height <= 0 or annotation_width <= 0:
            raise ValueError(f"Invalid annotation image shape: {frame.annotation_shape}")
        scale_x = out_width / annotation_width
        scale_y = out_height / annotation_height

    polygons: list[npt.NDArray[np.float64]] = []
    for index, ellipse in enumerate(frame.ellipses):
        manual = frame.polygons[index] if index < len(frame.polygons) else None
        points = np.asarray(manual, dtype=float) if manual else ellipse_to_polygon(ellipse)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        # Manual polygons and sampled ellipses are both in processed pixels.
        scaled = points * np.array([scale_x, scale_y])
        scaled[:, 0] = np.clip(scaled[:, 0], 0.0, float(out_width))
        scaled[:, 1] = np.clip(scaled[:, 1], 0.0, float(out_height))
        if _polygon_area(scaled) <= 0.0:
            continue
        polygons.append(scaled)
    return polygons


def render_label_mask(frame: FrameAnnotation) -> npt.NDArray[np.int32]:
    """Rebuild the integer label image of a frame from its ellipses and polygons.

    Batch processing frees ``labelled_mask`` to save memory, so the ``masks_npy`` export draws it
    again from the stored geometry, in the processed-image resolution and with the same convention
    as the live pipeline: background is 1 and bubbles are numbered from 2.

    Raises:
        ValueError: If the frame has no ``annotation_shape`` to draw on.
    """
    if frame.annotation_shape is None:
        raise ValueError(f"Cannot rebuild the mask of {frame.image_path.name}: unknown image shape.")
    height, width = frame.annotation_shape
    labelled = np.ones((height, width), dtype=np.int32)
    label = 2
    for index, ellipse in enumerate(frame.ellipses):
        manual = frame.polygons[index] if index < len(frame.polygons) else None
        points = np.asarray(manual, dtype=float) if manual else ellipse_to_polygon(ellipse)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        cv2.fillPoly(labelled, [np.round(points).astype(np.int32)], label)  # type: ignore[call-overload]
        label += 1
    return labelled


def _polygon_area(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _bounding_box(points: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` of a polygon."""
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of an image, tolerating non-ASCII paths on Windows."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return int(image.shape[0]), int(image.shape[1])


def _copy_image(image_path: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, images_dir / image_path.name)


def _write_yolo(root: Path, frames: list[_PreparedFrame], class_name: str, segmentation: bool) -> int:
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    instances = 0
    for item in frames:
        height, width = item.size
        _copy_image(item.frame.image_path, images_dir)
        lines: list[str] = []
        for polygon in item.polygons:
            if segmentation:
                normalised = polygon / np.array([width, height])
                lines.append("0 " + " ".join(f"{value:.6f}" for value in normalised.ravel()))
            else:
                x_min, y_min, x_max, y_max = _bounding_box(polygon)
                center_x = (x_min + x_max) / 2.0 / width
                center_y = (y_min + y_max) / 2.0 / height
                lines.append(
                    f"0 {center_x:.6f} {center_y:.6f} {(x_max - x_min) / width:.6f} {(y_max - y_min) / height:.6f}"
                )
        (labels_dir / f"{item.frame.image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        instances += len(lines)

    # `path` is left out on purpose: Ultralytics then resolves train/val from this file's folder.
    (root / "data.yaml").write_text(
        "# Written by Bubble Analyser. No train/val split was made: split the images before training.\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        f"  0: {json.dumps(class_name)}\n",
        encoding="utf-8",
    )
    return instances


def _write_coco(root: Path, frames: list[_PreparedFrame], class_name: str) -> int:
    images_dir = root / "images"
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for image_id, item in enumerate(frames, start=1):
        height, width = item.size
        _copy_image(item.frame.image_path, images_dir)
        images.append({"id": image_id, "file_name": item.frame.image_path.name, "width": width, "height": height})
        for polygon in item.polygons:
            x_min, y_min, x_max, y_max = _bounding_box(polygon)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [[round(float(value), 2) for value in polygon.ravel()]],
                    "bbox": [round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)],
                    "area": round(_polygon_area(polygon), 2),
                    "iscrowd": 0,
                }
            )
    document = {
        "info": {"description": "Bubble Analyser export"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": class_name, "supercategory": class_name}],
    }
    (root / "annotations.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    return len(annotations)


def _write_voc(root: Path, frames: list[_PreparedFrame], class_name: str) -> int:
    images_dir = root / "JPEGImages"
    annotations_dir = root / "Annotations"
    sets_dir = root / "ImageSets" / "Main"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)
    instances = 0
    for item in frames:
        height, width = item.size
        _copy_image(item.frame.image_path, images_dir)
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "JPEGImages"
        ET.SubElement(annotation, "filename").text = item.frame.image_path.name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "segmented").text = "0"
        for polygon in item.polygons:
            x_min, y_min, x_max, y_max = _bounding_box(polygon)
            # VOC pixel indices are 1-based and inclusive.
            box_x_min = int(np.clip(math.floor(x_min), 0, width - 1)) + 1
            box_y_min = int(np.clip(math.floor(y_min), 0, height - 1)) + 1
            box_x_max = int(np.clip(math.ceil(x_max), box_x_min, width))
            box_y_max = int(np.clip(math.ceil(y_max), box_y_min, height))
            touches_border = box_x_min == 1 or box_y_min == 1 or box_x_max == width or box_y_max == height
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "1" if touches_border else "0"
            ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(box_x_min)
            ET.SubElement(box, "ymin").text = str(box_y_min)
            ET.SubElement(box, "xmax").text = str(box_x_max)
            ET.SubElement(box, "ymax").text = str(box_y_max)
            instances += 1
        ET.indent(annotation)
        ET.ElementTree(annotation).write(
            annotations_dir / f"{item.frame.image_path.stem}.xml", encoding="utf-8", xml_declaration=True
        )
    (sets_dir / "trainval.txt").write_text(
        "".join(f"{item.frame.image_path.stem}\n" for item in frames), encoding="utf-8"
    )
    return instances


def _write_masks(root: Path, frames: Sequence[FrameAnnotation]) -> int:
    images_dir = root / "images"
    masks_dir = root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    instances = 0
    for frame in frames:
        stored = frame.labelled_mask
        mask = stored if stored is not None and stored.size > 0 else render_label_mask(frame)
        _copy_image(frame.image_path, images_dir)
        np.save(masks_dir / f"{frame.image_path.stem}_mask.npy", mask)
        instances += len(frame.ellipses)
    return instances


def export_dataset(
    output_root: Path,
    frames: Sequence[FrameAnnotation],
    format_key: str = DEFAULT_FORMAT_KEY,
    class_name: str = "bubble",
    progress: ProgressCallback | None = None,
) -> ExportSummary:
    """Write ``frames`` to ``output_root`` in the format registered under ``format_key``.

    Frames without bubbles are skipped (and counted) rather than written as empty
    annotations, because an empty label is indistinguishable from a failed segmentation.

    Raises:
        ValueError: If the format is unknown or no frame has anything to export.
    """
    get_format(format_key)  # fail early on an unknown key
    output_root = Path(output_root)
    summary = ExportSummary(format_key=format_key, output_root=output_root)
    total = len(frames)

    if format_key == "masks_npy":
        usable = [
            f
            for f in frames
            if (f.labelled_mask is not None and f.labelled_mask.size > 0)
            or (len(f.ellipses) > 0 and f.annotation_shape is not None)
        ]
        summary.frames_without_bubbles = total - len(usable)
        if not usable:
            raise ValueError("No processed frame has a mask to export.")
        summary.instances_exported = _write_masks(output_root, usable)
        summary.frames_exported = len(usable)
        return summary

    prepared: list[_PreparedFrame] = []
    for index, frame in enumerate(frames):
        if progress is not None:
            progress(index, total)
        if len(frame.ellipses) == 0:
            summary.frames_without_bubbles += 1
            continue
        try:
            size = _read_image_size(Path(frame.image_path))
            polygons = bubble_polygons(frame, size)
        except (OSError, ValueError) as error:
            summary.failed.append((Path(frame.image_path).name, str(error)))
            continue
        if not polygons:
            summary.frames_without_bubbles += 1
            continue
        prepared.append(_PreparedFrame(frame, size, polygons))

    if not prepared:
        raise ValueError("No frame has bubbles to export.")

    if format_key == "yolo_seg":
        summary.instances_exported = _write_yolo(output_root, prepared, class_name, segmentation=True)
    elif format_key == "yolo_det":
        summary.instances_exported = _write_yolo(output_root, prepared, class_name, segmentation=False)
    elif format_key == "coco":
        summary.instances_exported = _write_coco(output_root, prepared, class_name)
    else:
        summary.instances_exported = _write_voc(output_root, prepared, class_name)
    summary.frames_exported = len(prepared)
    if progress is not None:
        progress(total, total)
    return summary
