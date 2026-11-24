"""Download and unpack the GTSRB training set into ``data/``.

GTSRB (German Traffic Sign Recognition Benchmark, Stallkamp et al., 2011)
is the public benchmark used in this project. Only the official *training*
archive is needed: it contains ~39,000 images of 43 classes together with
one CSV annotation file per class (file name, image size, sign ROI, class id).
Our own reproducible train/test split is created later in ``src/dataset.py``.

Usage:
    python download_data.py
"""

import http.client
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Official mirror of the GTSRB archives (University of Copenhagen, ERDA).
URL = ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
       "GTSRB_Final_Training_Images.zip")

# All paths are relative to this file so the script works from any directory.
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "GTSRB_Final_Training_Images.zip"
# Incomplete downloads live under a separate name, so a broken file is never
# mistaken for a finished archive.
PART_PATH = ZIP_PATH.with_suffix(".zip.part")
# Folder that exists after a successful extraction; used to skip re-downloads.
IMAGES_DIR = DATA_DIR / "GTSRB" / "Final_Training" / "Images"

# The server sometimes drops long connections, so the download is resumed
# a few times instead of failing on the first interruption.
MAX_ATTEMPTS = 10
CHUNK_SIZE = 1024 * 1024


def download_with_resume() -> None:
    """Download ``URL`` to ``ZIP_PATH``, resuming after connection drops.

    Each attempt asks the server only for the bytes that are still missing
    (HTTP ``Range`` header) and appends them to the ``.part`` file.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        already_have = PART_PATH.stat().st_size if PART_PATH.exists() else 0
        request = urllib.request.Request(
            URL, headers={"Range": f"bytes={already_have}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # 206 = server honours the range; 200 = it sends the whole
                # file again, in which case the old partial data is dropped.
                if response.status == 206:
                    mode = "ab"
                    # Header looks like "bytes 1000-9999/10000"; the number
                    # after "/" is the size of the complete file.
                    content_range = response.headers["Content-Range"]
                    total = int(content_range.split("/")[1])
                else:
                    mode = "wb"
                    total = int(response.headers["Content-Length"])
                with open(PART_PATH, mode) as out_file:
                    while chunk := response.read(CHUNK_SIZE):
                        out_file.write(chunk)
            # A stream can also end early without raising an error, so the
            # archive is only accepted when it has exactly the expected size.
            if PART_PATH.stat().st_size == total:
                PART_PATH.rename(ZIP_PATH)
                return
            print(f"  attempt {attempt} ended early, resuming ...")
        except urllib.error.HTTPError as error:
            # 416 = requested range starts at the end, i.e. the file is
            # already complete.
            if error.code == 416:
                PART_PATH.rename(ZIP_PATH)
                return
            raise
        except (urllib.error.URLError, http.client.HTTPException,
                TimeoutError, ConnectionError) as error:
            size_mb = (PART_PATH.stat().st_size / 1e6
                       if PART_PATH.exists() else 0)
            print(f"  attempt {attempt} interrupted at {size_mb:.0f} MB "
                  f"({error}), resuming ...")
    raise RuntimeError(f"Download failed after {MAX_ATTEMPTS} attempts.")


def main() -> None:
    """Download the archive (if missing) and extract it into ``data/``."""
    if IMAGES_DIR.is_dir():
        print(f"Dataset already present in {IMAGES_DIR}")
        return

    DATA_DIR.mkdir(exist_ok=True)

    # The archive is ~276 MB, so it is kept after download: if extraction
    # fails, re-running the script does not need to download it again.
    if not ZIP_PATH.is_file():
        print(f"Downloading {URL} ...")
        download_with_resume()

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    print(f"Done. Images are in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
