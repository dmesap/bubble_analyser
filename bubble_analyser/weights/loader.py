from pathlib import Path

import requests
from tqdm import tqdm

# Direct download URLs for the deep learning weights used by the app.
WEIGHTS_URLS = {
    # BubMask (Mask R-CNN) weights, hosted on the project's v0.3.0 GitHub release.
    "mask_rcnn_bubble.h5": "https://github.com/ImperialCollegeLondon/bubble_analyser/releases/download/v0.3.0/mask_rcnn_bubble.h5",
    # SAM 2.1 checkpoints, hosted on Meta's CDN (Apache 2.0 licensed).
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}

# Any other weights file (e.g. the YOLO model yolo_bubble_seg.pt) is published on
# the project's v2.0.0 GitHub release.
DEFAULT_RELEASE_URL = "https://github.com/ImperialCollegeLondon/bubble_analyser/releases/download/v2.0.0"


def get_weights_url(filename: str) -> str:
    """Return the download URL for a weights file.

    Args:
        filename: Name of the weights file, e.g. ``sam2.1_hiera_large.pt``.

    Returns:
        The known URL for the filename (see WEIGHTS_URLS), or the file on the
        project's v2.0.0 GitHub release for any other filename.
    """
    return WEIGHTS_URLS.get(filename, f"{DEFAULT_RELEASE_URL}/{filename}")


def get_weights_path(
    filename: str = "mask_rcnn_bubble.h5",
    download_if_missing: bool = True,
    url: str | None = None,
) -> tuple[str | None, str | None]:
    """Returns the local path to the weights.

    If missing and download_if_missing is True, fetches via terminal (tqdm).
    If False, returns (None, url) so the GUI can handle the download thread.

    Args:
        filename: Name of the weights file inside the weights directory.
        download_if_missing: Whether to download the file if it is missing.
        url: Optional override download URL. Defaults to get_weights_url(filename):
            the known URL for the filename (see WEIGHTS_URLS), falling back to the
            project's v2.0.0 GitHub release for other filenames.
    """
    weights_dir = Path(__file__).parent
    local_path = weights_dir / filename

    # Resolve the download URL: explicit override, else the known URL for the filename.
    if url is None:
        url = get_weights_url(filename)

    if local_path.exists():
        return str(local_path), None

    if not download_if_missing:
        # Return the URL so the PySide6 GUI can handle download/file prompt
        return None, url

    # Fallback for terminal/script usage
    print(f"Weights not found. Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading Weights") as bar,
        ):
            for data in response.iter_content(chunk_size=16384):  # Larger chunks for efficiency
                f.write(data)
                bar.update(len(data))
        return str(local_path), None
    except Exception as e:
        print(f"Critical error downloading weights: {e}")
        return None, url
