# Bubble Analyser 2.0

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blue)](https://python-poetry.org/)

## Overview

Bubble Analyser 2.0 is a robust Python application for detecting, measuring, and analyzing bubbles in images. It provides advanced image processing capabilities using watershed segmentation algorithms and Deep Learning to accurately identify and measure bubbles of various sizes and distributions. It also offers manual adjustment tools for fine-tuning segmentation results when needed, and high-fidelity ML data export to help researchers train their own custom models.

This project is significantly improved based on the first version of [Bubble Analyser (Mesa et al., 2022)](https://doi.org/10.1016/j.mineng.2022.107497).

## Key Features

- **Image Processing**: Preprocess images with various transformations.
- **Bubble Detection Methods**:
  - **Normal Watershed**: Standard watershed algorithm with triple threshold.
  - **Iterative Watershed**: Advanced algorithm that iteratively applies thresholds to detect objects at different intensity levels.
  - **Peak-Based Watershed**: Feature-based segmentation using local maxima.
  - **BubMask (Deep Learning)**: Integrated Mask R-CNN model based on [BubMask (Kim & Park, 2021)](https://github.com/ywflow/BubMask) for superior accuracy in complex lighting or overlapping conditions.
  - **YOLO Segmentation (optional)**: Any [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO segmentation model (`.pt`). It needs `pip install ultralytics` (AGPL-3.0) and a weights file set in `weights_path`; the device can be CPU or GPU (CUDA).
- **Interactive Ellipse Adjuster**: Manual adjustment interface for pixel-perfect refinement of detected bubbles.
- **Measurement & Calibration**: Calculate sizes, shape distributions, and convert pixel measurements to real-world units (mm) using reference images.
- **Streamlined Workflow**: Linear processing workflow (Batch -> Finalise -> Export) with no mid-process interruptions.
- **Versatile Data Exports**:
  - **Excel + Graph**: Comprehensive descriptive size data alongside PDF/CDF distribution plots.
  - **Annotated Images**: High-quality visual outputs with drawn ellipses for document reporting.
  - **ML Training Data**: Export raw images alongside perfect integer ID masks (`.npy`) allowing users to easily build new machine learning datasets from their manual corrections.

## Installation

### 1. Windows Installation (Recommended)

A fully featured Windows Installation Wizard is now available.

1. Download `BubbleAnalyser_Setup_v2.exe` from the [Latest Release](https://github.com/ImperialCollegeLondon/bubble_analyser/releases).
2. Run the installer.
3. **Important**: During installation, ensure the **"Download Deep Learning weights (~250MB)"** option is checked if you plan to use the BubMask (Deep Learning) method.
4. Launch "Bubble Analyser 2.0" from your Start Menu or Desktop.

### 2. MacOS Executable

A standalone executable for MacOS is also available in the releases section.
*(Note: If using the Deep Learning method on MacOS, the application will prompt you to download the required weights upon your first analysis).*

### 3. From Source (Developers)

To run from source or contribute to the project:

1. [Download and install Poetry](https://python-poetry.org/docs/#installation) following the instructions for your OS.
2. Clone this repository and navigate into the directory.
3. Install dependencies and activate the virtual environment:

   ```bash
   poetry install
   poetry shell
   ```

4. Install the pre-commit hooks (for code quality):

   ```bash
   pre-commit install
   ```

5. Run the application:

   ```bash
   python -m bubble_analyser
   ```

> [!NOTE]
> **Using CNN Detection from Source**: If you attempt to use the BubMask method from source without the weights, the app will display a prompt offering to download the `mask_rcnn_bubble.h5` file automatically into the `bubble_analyser/weights/` directory.

## Exporting Results

Once your batch analysis is finalised, navigate to the **Results** tab to extract your data:

1. **Save Results (Excel + Graph)**: Creates an Excel file containing all bubble metrics (major/minor axis, equivalent diameter, area, eccentricity) and a `.png` of the generated histogram.
2. **Export Annotated Images**: Select a folder to dump all the visual previews (images overlaid with the detected ellipses) for inclusion in reports.
3. **Export Dataset (choose format)**: Pick a format, then a root folder. Every format uses your manual edits, writes bubbles at the original image resolution (manual polygon when there is one, fitted ellipse otherwise), and copies the images next to the annotations. Images with no bubbles are skipped.

   | Format | Output | Typical use |
   | --- | --- | --- |
   | YOLO segmentation | `images/`, `labels/*.txt` (`class x1 y1 x2 y2 ...`, normalised), `data.yaml` | Training Ultralytics `-seg` models |
   | YOLO detection | `images/`, `labels/*.txt` (`class x_center y_center width height`, normalised), `data.yaml` | Training box detectors |
   | COCO instance segmentation | `images/`, `annotations.json` (pixel polygons, boxes, areas) | Detectron2, MMDetection, pycocotools |
   | Pascal VOC | `JPEGImages/`, `Annotations/*.xml`, `ImageSets/Main/trainval.txt` (boxes only) | Box-based VOC tooling |
   | Masks (`.npy`) | `images/`, `masks/*_mask.npy` (integer label image at the processed resolution) | The export of earlier versions |

   The YOLO `data.yaml` points `train` and `val` at the same folder: no train/validation split is made, so split the images before training.

## Dependencies

- Python 3.12
- PySide6 (Qt for Python)
- TensorFlow & Keras
- NumPy, SciPy, Pandas
- scikit-image, OpenCV
- Matplotlib

## Reference

Kim, Y., Park, H. Deep learning-based automated and universal bubble detection and mask extraction in complex two-phase flows. Sci Rep 11, 8940 (2021). <https://doi.org/10.1038/s41598-021-88334-0>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. If you would like to add more image processing algorithms, please see the guidance under the directory [Methods](bubble_analyser/methods).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- Dr Diego Mesa (Main Author) - Department of Earth Science and Engineering, Imperial College London, United Kingdom
- Mr Yiyang Guan - Department of Earth Science and Engineering, Imperial College London, United Kingdom

### Past Collaborators

- Dr Diego Alonso Álvarez - Imperial College London RSE Team, Imperial College London, United Kingdom
- Dr Paulina Quintanilla - Department of Chemical Engineering, Brunel University, United Kingdom
- Dr Francisco Reyes - IntelliSense.io, Queensland, Australia
- Dr Luis Vinnett - Universidad Tecnica Federico Santa Maria, Chile
