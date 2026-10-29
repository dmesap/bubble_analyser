import numpy as np

from bubble_analyser.methods.yolo_method import YOLOSegmentation, normalize_yolo_device


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
