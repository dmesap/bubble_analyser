from pathlib import Path

import numpy as np
import pytest

from bubble_analyser.methods.sam_method import SamSegmentation, masks_to_labels

SAM_WEIGHTS = Path(__file__).resolve().parents[3] / "bubble_analyser" / "weights" / "sam2.1_hiera_large.pt"


@pytest.fixture
def dummy_images():
    img_grey = np.zeros((100, 100), dtype=np.uint8)
    img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    return img_grey, img_rgb


@pytest.fixture
def sam_params():
    return {
        "weights_path": "mock_weights.pt",
        "points_per_side": 16,
        "pred_iou_thresh": 0.5,
        "stability_score_thresh": 0.9,
        "min_mask_region_area": 0,
        "resample": 1.0,
    }


def test_sam_segmentation_init(sam_params):
    method = SamSegmentation(sam_params)
    assert method.name == "SAM 2.1 (Segment Anything)"
    assert method.points_per_side == 16
    assert method.pred_iou_thresh == 0.5
    assert method.stability_score_thresh == 0.9


def test_sam_segmentation_update_params(sam_params):
    method = SamSegmentation(sam_params)
    method.update_params({"points_per_side": 64, "pred_iou_thresh": 0.7})
    assert method.points_per_side == 64
    assert method.pred_iou_thresh == 0.7


def test_sam_get_needed_params_roundtrip(sam_params):
    method = SamSegmentation(sam_params)
    params = method.get_needed_params()
    for key in ("weights_path", "points_per_side", "pred_iou_thresh", "stability_score_thresh", "resample"):
        assert key in params


def test_masks_to_labels_labels_bubbles_from_two():
    masks = np.zeros((100, 100, 3), dtype=bool)
    masks[10:20, 10:20, 0] = True
    masks[40:50, 40:50, 1] = True
    masks[70:80, 70:80, 2] = True

    labels = masks_to_labels(masks, (100, 100))

    assert labels.shape == (100, 100)
    assert set(np.unique(labels)) == {0, 2, 3, 4}
    assert labels[15, 15] == 2
    assert labels[45, 45] == 3
    assert labels[75, 75] == 4


def test_masks_to_labels_empty_stack():
    masks = np.zeros((100, 100, 0), dtype=bool)
    labels = masks_to_labels(masks, (100, 100))
    assert labels.sum() == 0


def test_masks_to_labels_resizes_to_target_shape():
    masks = np.zeros((50, 50, 1), dtype=bool)
    masks[10:20, 10:20, 0] = True

    labels = masks_to_labels(masks, (100, 100))

    assert labels.shape == (100, 100)
    assert 2 in np.unique(labels)


def test_sam_missing_weights_raises(dummy_images, sam_params):
    img_grey, img_rgb = dummy_images
    method = SamSegmentation(sam_params)
    method.initialize_processing(sam_params, img_grey, img_rgb, False, px2mm=1.0)
    with pytest.raises(FileNotFoundError):
        method.detect_bubbles()


def test_sam_detection_confidence_empty():
    method = SamSegmentation({"resample": 1.0})
    assert method.get_detection_confidence() == []


@pytest.mark.skipif(
    not SAM_WEIGHTS.exists(), reason="SAM 2.1 checkpoint not downloaded and torch not verified"
)
def test_sam_end_to_end(dummy_images):
    img_grey, img_rgb = dummy_images
    method = SamSegmentation({"resample": 1.0})
    method.initialize_processing({"resample": 1.0}, img_grey, img_rgb, False, px2mm=1.0)
    labels_on_img, labels_watershed, _ = method.get_results_img()

    assert labels_on_img.shape == img_rgb.shape
    assert labels_watershed.shape == img_grey.shape
