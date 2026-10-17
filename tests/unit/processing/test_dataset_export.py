import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pytest

from bubble_analyser.processing.dataset_export import (
    DATASET_FORMATS,
    FrameAnnotation,
    bubble_polygons,
    ellipse_to_polygon,
    export_dataset,
    get_format,
)

# The original image is 400x200 (width x height); the geometry was drawn on a 200x100 copy, so x2 everywhere.
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 200, 400
ANNOTATION_SHAPE = (100, 200)
ELLIPSE = ((50.0, 50.0), (20.0, 10.0), 30.0)
SQUARE = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), 128, dtype=np.uint8)
    # imencode + tofile, because cv2.imwrite cannot write to non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(str(path))
    return path


def _frame(path: Path, **overrides) -> FrameAnnotation:
    values: dict[str, Any] = {
        "image_path": path,
        "ellipses": [ELLIPSE, ((20.0, 20.0), (20.0, 20.0), 0.0)],
        "polygons": [None, SQUARE],
        "annotation_shape": ANNOTATION_SHAPE,
        "labelled_mask": np.arange(12, dtype=np.int32).reshape(3, 4),
    }
    values.update(overrides)
    return FrameAnnotation(**values)


@pytest.fixture
def frame(tmp_path: Path) -> FrameAnnotation:
    return _frame(_write_image(tmp_path / "src" / "frame_01.png"))


def test_ellipse_polygon_matches_opencv_rasterisation():
    ellipse = ((60.0, 50.0), (80.0, 40.0), 35.0)
    polygon = np.round(ellipse_to_polygon(ellipse)).astype(np.int32)

    from_polygon = np.zeros((120, 140), dtype=np.uint8)
    cv2.fillPoly(from_polygon, [polygon], 255)
    from_opencv = np.zeros((120, 140), dtype=np.uint8)
    cv2.ellipse(from_opencv, ellipse, 255, -1)

    intersection = np.logical_and(from_polygon > 0, from_opencv > 0).sum()
    union = np.logical_or(from_polygon > 0, from_opencv > 0).sum()
    assert intersection / union > 0.95


def test_manual_polygon_wins_over_ellipse_and_is_scaled(frame):
    polygons = bubble_polygons(frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

    assert len(polygons) == 2
    assert np.allclose(polygons[1], np.array(SQUARE) * 2)
    assert np.allclose(polygons[0].mean(axis=0), (100.0, 100.0), atol=1.0)


def test_geometry_is_clipped_to_the_image(tmp_path):
    frame = _frame(
        _write_image(tmp_path / "f.png"),
        ellipses=[((2.0, 2.0), (30.0, 30.0), 0.0)],
        polygons=[None],
    )

    (polygon,) = bubble_polygons(frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

    assert polygon[:, 0].min() >= 0.0 and polygon[:, 1].min() >= 0.0
    assert polygon[:, 0].max() <= ORIGINAL_WIDTH and polygon[:, 1].max() <= ORIGINAL_HEIGHT


def test_without_annotation_shape_geometry_is_taken_as_original_pixels(tmp_path):
    frame = _frame(_write_image(tmp_path / "f.png"), annotation_shape=None, ellipses=[ELLIPSE], polygons=[SQUARE])

    (polygon,) = bubble_polygons(frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

    assert np.allclose(polygon, SQUARE)


def test_yolo_segmentation_layout(frame, tmp_path):
    out = tmp_path / "out"

    summary = export_dataset(out, [frame], "yolo_seg")

    assert (summary.frames_exported, summary.instances_exported) == (1, 2)
    assert (out / "images" / "frame_01.png").is_file()
    lines = (out / "labels" / "frame_01.txt").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        values = line.split()
        assert values[0] == "0"
        coordinates = [float(value) for value in values[1:]]
        assert len(coordinates) % 2 == 0 and all(0.0 <= value <= 1.0 for value in coordinates)
    first_point = [float(value) for value in lines[1].split()[1:3]]
    assert first_point == pytest.approx([20 / ORIGINAL_WIDTH, 20 / ORIGINAL_HEIGHT])
    yaml_text = (out / "data.yaml").read_text()
    assert "train: images" in yaml_text and '0: "bubble"' in yaml_text
    assert "path:" not in yaml_text


def test_yolo_detection_uses_normalised_boxes(frame, tmp_path):
    out = tmp_path / "out"

    export_dataset(out, [frame], "yolo_det")

    lines = (out / "labels" / "frame_01.txt").read_text().strip().splitlines()
    assert all(len(line.split()) == 5 for line in lines)
    # The square spans (20, 20)-(60, 60) in a 400x200 image.
    assert [float(value) for value in lines[1].split()[1:]] == pytest.approx([0.1, 0.2, 0.1, 0.2])


def test_coco_document(frame, tmp_path):
    out = tmp_path / "out"

    export_dataset(out, [frame], "coco")

    document = json.loads((out / "annotations.json").read_text())
    assert document["images"] == [{"id": 1, "file_name": "frame_01.png", "width": 400, "height": 200}]
    assert document["categories"][0]["name"] == "bubble"
    ellipse_annotation, square_annotation = document["annotations"]
    assert square_annotation["bbox"] == [20.0, 20.0, 40.0, 40.0]
    assert square_annotation["area"] == pytest.approx(1600.0)
    assert len(square_annotation["segmentation"]) == 1 and len(square_annotation["segmentation"][0]) == 8
    assert ellipse_annotation["area"] == pytest.approx(np.pi * 20 * 10, rel=0.03)
    assert {item["image_id"] for item in document["annotations"]} == {1}
    assert [item["id"] for item in document["annotations"]] == [1, 2]


def test_voc_annotation(frame, tmp_path):
    out = tmp_path / "out"

    export_dataset(out, [frame], "voc")

    assert (out / "JPEGImages" / "frame_01.png").is_file()
    root = ET.parse(out / "Annotations" / "frame_01.xml").getroot()
    assert root.findtext("filename") == "frame_01.png"
    assert (root.findtext("size/width"), root.findtext("size/height")) == ("400", "200")
    objects = root.findall("object")
    assert len(objects) == 2
    square = objects[1].find("bndbox")
    assert [square.findtext(tag) for tag in ("xmin", "ymin", "xmax", "ymax")] == ["21", "21", "60", "60"]
    assert objects[1].findtext("truncated") == "0"
    assert (out / "ImageSets" / "Main" / "trainval.txt").read_text().split() == ["frame_01"]


def test_masks_keep_the_previous_behaviour(frame, tmp_path):
    out = tmp_path / "out"

    summary = export_dataset(out, [frame], "masks_npy")

    assert summary.frames_exported == 1
    assert (out / "images" / "frame_01.png").is_file()
    assert np.array_equal(np.load(out / "masks" / "frame_01_mask.npy"), frame.labelled_mask)


def test_skipped_and_failed_frames_are_reported(frame, tmp_path):
    empty = _frame(_write_image(tmp_path / "src" / "empty.png"), ellipses=[], polygons=[])
    missing = _frame(tmp_path / "src" / "missing.png")

    summary = export_dataset(tmp_path / "out", [frame, empty, missing], "yolo_seg")

    assert summary.frames_exported == 1
    assert summary.frames_without_bubbles == 1
    assert [name for name, _ in summary.failed] == ["missing.png"]
    assert not (tmp_path / "out" / "labels" / "empty.txt").exists()


def test_non_ascii_paths_are_readable(tmp_path):
    frame = _frame(_write_image(tmp_path / "ñandú" / "frame_01.png"))

    summary = export_dataset(tmp_path / "out", [frame], "coco")

    assert summary.frames_exported == 1


def test_nothing_to_export_raises_and_writes_nothing(tmp_path):
    empty = _frame(_write_image(tmp_path / "src" / "empty.png"), ellipses=[], polygons=[], labelled_mask=None)
    out = tmp_path / "out"

    with pytest.raises(ValueError):
        export_dataset(out, [empty], "voc")
    with pytest.raises(ValueError):
        export_dataset(out, [empty], "masks_npy")

    assert not out.exists()


def test_unknown_format_is_rejected(frame, tmp_path):
    with pytest.raises(ValueError, match="Unknown export format"):
        export_dataset(tmp_path / "out", [frame], "tfrecord")


def test_every_registered_format_can_be_exported(frame, tmp_path):
    for dataset_format in DATASET_FORMATS:
        assert get_format(dataset_format.key) is dataset_format
        summary = export_dataset(tmp_path / dataset_format.key, [frame], dataset_format.key)
        assert summary.frames_exported == 1
