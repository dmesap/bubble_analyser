import numpy as np

from bubble_analyser.methods.yolo_method import YOLOSegmentation, letterbox_content_window, normalize_yolo_device


def test_yolo_device_labels_are_normalized_for_ultralytics() -> None:
    assert normalize_yolo_device("CPU") == "cpu"
    assert normalize_yolo_device("cpu") == "cpu"
    assert normalize_yolo_device("GPU") == "0"
    assert normalize_yolo_device("cuda") == "cuda"
    assert normalize_yolo_device("0") == "0"


def test_yolo_masks_are_converted_to_bubble_analyser_labels() -> None:
    method = YOLOSegmentation({})
    method.img_grey = np.zeros((8, 8), dtype=np.uint8)
    method.confidence_threshold = 0.25
    method.max_bubbles = 2

    masks = np.zeros((3, 8, 8), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 4:7, 4:7] = 1.0
    masks[2, 0:2, 6:8] = 1.0
    scores = np.array([0.90, 0.10, 0.80], dtype=np.float32)

    labels = method._convert_masks_to_labels(masks, scores)

    assert labels.dtype == np.int32
    assert labels[1, 1] == 2
    assert labels[0, 6] == 3
    assert labels[4, 4] == 0
    assert set(np.unique(labels)) == {0, 2, 3}


def test_letterbox_window_matches_ultralytics_padding() -> None:
    # A 1280x720 frame is letterboxed by Ultralytics into a 640x384 input: 12 rows of padding top and bottom.
    assert letterbox_content_window((384, 640), (720, 1280)) == (12, 0, 360, 640)
    assert letterbox_content_window((8, 8), (8, 8)) == (0, 0, 8, 8)


def test_letterboxed_masks_are_unpadded_before_resizing() -> None:
    method = YOLOSegmentation({})
    method.img_grey = np.zeros((8, 16), dtype=np.uint8)  # wide image
    method.confidence_threshold = 0.25

    # The 8x16 image sits in rows 4-11 of a 16x16 letterboxed mask.
    masks = np.zeros((1, 16, 16), dtype=np.float32)
    masks[0, 4:8, 2:6] = 1.0  # top-left corner of the image content
    labels = method._convert_masks_to_labels(masks, np.array([0.9], dtype=np.float32))

    assert labels.shape == (8, 16)
    assert np.argwhere(labels == 2).min(axis=0).tolist() == [0, 2]
    assert np.argwhere(labels == 2).max(axis=0).tolist() == [3, 5]
