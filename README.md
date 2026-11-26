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
  - **SAM 2.1 (Zero-shot Deep Learning)**: Meta's [Segment Anything Model 2.1](https://github.com/facebookresearch/sam2) with automatic mask generation — detects bubbles without any bubble-specific training (Apache 2.0). Runs on GPU when available.
  - **BubMask (Deep Learning)**: Integrated Mask R-CNN model based on [BubMask (Kim & Park, 2021)](https://github.com/ywflow/BubMask) for superior accuracy in complex lighting or overlapping conditions.
  - **YOLO Segmentation (optional)**: Any [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO segmentation model (`.pt`). It needs `ultralytics` (AGPL-3.0; `poetry install` adds it through the `yolo` dependency group, otherwise `pip install ultralytics`) and a weights file set in `weights_path`; the device can be CPU or GPU (CUDA).
- **Interactive Ellipse Adjuster**: Manual adjustment interface for pixel-perfect refinement of detected bubbles.
- **Measurement & Calibration**: Calculate sizes, shape distributions, and convert pixel measurements to real-world units (mm) using reference images.
- **Streamlined Workflow**: Linear processing workflow (Batch -> Finalise -> Export) with no mid-process interruptions.
- **Versatile Data Exports**:
  - **Excel + Graph**: Comprehensive descriptive size data alongside PDF/CDF distribution plots.
  - **Annotated Images**: High-quality visual outputs with drawn ellipses for document reporting.
  - **ML Training Data**: Export raw images alongside perfect integer ID masks (`.npy`) allowing users to easily build new machine learning datasets from their manual corrections.

## Installation

> [!IMPORTANT]
> **This branch (`sammethod-gyy`) adds the SAM 2.1 segmentation method, and it installs differently from the normal version.**
> The Windows installer and the MacOS executable (sections 1 and 2) are the normal version and do not include SAM 2.1.
> To use SAM 2.1, install from source with the steps in [4. SAM 2.1 Version (this branch)](#4-sam-21-version-this-branch).
> Follow section 4 rather than section 3 on this branch: here `poetry install` also installs PyTorch and SAM 2.1.

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

### 4. SAM 2.1 Version (this branch)

The SAM 2.1 version is the normal version plus PyTorch, SAM 2.1 and Ultralytics. It can only be installed from source, and it takes a few more steps than the normal version.

#### How it differs from the normal version

| | Normal version | SAM 2.1 version (this branch) |
| --- | --- | --- |
| How to install | Windows installer, MacOS executable or from source | From source only: the installers do not include SAM 2.1 |
| Operating systems | Windows, MacOS, Linux | Windows and Linux (64-bit). Not MacOS: the lock file pins the CUDA 12.8 build of PyTorch, which has no MacOS version |
| Extra packages | None | PyTorch 2.10 and torchvision 0.25 (CUDA 12.8), SAM-2 (installed from GitHub), Ultralytics |
| Disk space | About 3 GB for the Python environment | About 8 GB for the Python environment, plus 0.86 GB for the SAM 2.1 checkpoint |
| Hardware | Any CPU | NVIDIA GPU recommended, with a driver that supports CUDA 12.8; 6 GB of GPU memory is enough at the default settings. Without a GPU, SAM 2.1 runs on the CPU, about 9 times slower |
| Install commands | `poetry install` | Install PyTorch with pip first, then `poetry install` (steps below) |
| Model weights | BubMask weights, offered on first use | BubMask weights, plus the SAM 2.1 checkpoint offered the first time you select SAM 2.1 |

#### Steps

1. Install Python 3.11 or 3.12 and [Poetry](https://python-poetry.org/docs/#installation). To run SAM 2.1 on the GPU, install an NVIDIA driver that supports CUDA 12.8 (`nvidia-smi` shows the highest CUDA version the driver supports).
2. Clone the repository and switch to this branch:

   ```bash
   git clone https://github.com/dmesap/bubble_analyser.git
   cd bubble_analyser
   git checkout sammethod-gyy
   ```

3. Tell Poetry which Python to use, then install PyTorch on its own. The PyTorch package is larger than 2 GB and `poetry install` can stop with a `MemoryError` while installing it, so install it with pip first:

   ```bash
   poetry env use 3.12   # or the full path to a Python 3.11 or 3.12 executable
   poetry run pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
   ```

4. Install everything else. Poetry finds the PyTorch you just installed and skips it:

   ```bash
   poetry install
   ```

   Add `--without yolo` to leave out Ultralytics if you do not need YOLO Segmentation.

5. Check that PyTorch can use the GPU. `True` means SAM 2.1 will run on it; `False` means it will run on the CPU:

   ```bash
   poetry run python -c "import torch; print(torch.cuda.is_available())"
   ```

6. Start the application:

   ```bash
   poetry run python -m bubble_analyser
   ```

7. In the **Bubble detection and filtering** tab, choose **SAM 2.1 (Segment Anything)**. The first time, the app offers to download `sam2.1_hiera_large.pt` (0.86 GB) into `bubble_analyser/weights/`. The smaller checkpoints (`sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_tiny.pt`) also work: download one from the [SAM 2 repository](https://github.com/facebookresearch/sam2) and set `weights_path` to it.

> [!NOTE]
>
> - `poetry.toml` on this branch keeps the virtual environment out of the project folder (Poetry creates it in its own cache folder). This avoids syncing thousands of files when the project sits in a cloud folder such as OneDrive.
> - The first `poetry install` also downloads a temporary copy of PyTorch to build SAM-2, so allow extra time for that download.
> - On Windows, SAM 2.1's optional CUDA extension is not compiled. If you set `min_mask_region_area` above 0, SAM 2 prints a warning about it; this is expected, and the holes and specks are still cleaned up.

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
- PyTorch & SAM 2.1 (zero-shot segmentation)
- Ultralytics YOLO (optional `yolo` dependency group)
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
- Mr Yiyang Guan - Curtin University, Australia
- Mr Bastian Valdivia-Muñoz - Universidad de Talca, Chile

### Past Collaborators

- Dr Diego Alonso Álvarez - Imperial College London RSE Team, Imperial College London, United Kingdom
- Dr Paulina Quintanilla - Department of Chemical Engineering, Brunel University, United Kingdom
- Dr Francisco Reyes - IntelliSense.io, Queensland, Australia
- Dr Luis Vinnett - Universidad Tecnica Federico Santa Maria, Chile
