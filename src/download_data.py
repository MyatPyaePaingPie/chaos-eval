"""Download datasets for chaos-eval experiments."""

import os
import zipfile
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in [".gz", ".tar", ".tgz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        print(f"  Unknown archive format: {archive_path.suffix}")


def download_funsd():
    """
    Download FUNSD dataset.

    FUNSD: Form Understanding in Noisy Scanned Documents
    - 199 real, fully annotated, scanned forms
    - 149 training + 50 testing
    - Annotations: question, answer, header, other

    Source: https://guillaumejaume.github.io/FUNSD/
    """
    print("\n=== Downloading FUNSD ===")

    funsd_dir = DATA_DIR / "funsd"

    if (funsd_dir / "dataset").exists():
        print("  FUNSD already downloaded.")
        return

    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    archive = DATA_DIR / "funsd.zip"

    download_file(url, archive, "FUNSD")
    print("  Extracting...")
    extract_archive(archive, funsd_dir)

    # Clean up
    archive.unlink()
    print("  Done!")


def download_sroie():
    """
    Download SROIE dataset.

    SROIE: Scanned Receipts OCR and Information Extraction
    - 1000 receipt images
    - Ground truth: company, address, date, total

    Note: SROIE requires registration at https://rrc.cvc.uab.es/?ch=13
    This function provides instructions for manual download.
    """
    print("\n=== SROIE Dataset ===")

    sroie_dir = DATA_DIR / "sroie"

    if sroie_dir.exists() and any(sroie_dir.iterdir()):
        print("  SROIE directory exists.")
        return

    sroie_dir.mkdir(parents=True, exist_ok=True)

    print("""
  SROIE requires manual download due to registration requirement.

  Steps:
  1. Go to https://rrc.cvc.uab.es/?ch=13&com=downloads
  2. Register/login
  3. Download Task 1 & 2 training data
  4. Extract to: data/sroie/

  Expected structure:
    data/sroie/
    ├── train/
    │   ├── img/
    │   └── entities/
    └── test/
        ├── img/
        └── entities/

  Alternative: Use HuggingFace datasets
    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("mychen76/invoices-and-receipts_ocr_v1")
""")


def download_nod():
    """
    Download NOD (Noisy OCR Dataset).

    NOD: Documents at multiple noise levels for testing OCR robustness
    - 18,504 images (14,168 English, 4,400 Arabic)
    - 44 different noise conditions

    Source: https://zenodo.org/records/5068735
    """
    print("\n=== NOD Dataset ===")

    nod_dir = DATA_DIR / "nod"

    if nod_dir.exists() and any(nod_dir.iterdir()):
        print("  NOD directory exists.")
        return

    nod_dir.mkdir(parents=True, exist_ok=True)

    # NOD is ~2GB, provide instructions
    print("""
  NOD dataset is large (~2GB). Download options:

  Option 1: Direct download (English subset)
    URL: https://zenodo.org/records/5068735/files/English.zip

  Option 2: Use zenodo_get
    pip install zenodo_get
    zenodo_get 5068735 -o data/nod/

  Option 3: Manual download
    1. Go to https://zenodo.org/records/5068735
    2. Download desired files
    3. Extract to data/nod/

  For quick testing, we'll download a sample...
""")

    # Download just the README for now
    try:
        readme_url = "https://zenodo.org/records/5068735/files/README.md"
        download_file(readme_url, nod_dir / "README.md", "NOD README")
    except Exception:
        pass


def download_rvlcdip_sample():
    """
    Download a sample from RVL-CDIP for document classification testing.

    RVL-CDIP: 400K images, 16 document classes
    Full dataset is 40GB, so we use HuggingFace streaming.
    """
    print("\n=== RVL-CDIP Sample ===")
    print("""
  RVL-CDIP is 40GB. For experiments, use HuggingFace streaming:

    from datasets import load_dataset
    ds = load_dataset("rvl_cdip", split="test", streaming=True)

    # Get samples
    samples = list(ds.take(100))
""")


def create_sample_docs():
    """Create sample documents for quick testing."""
    print("\n=== Creating Sample Documents ===")

    samples_dir = DATA_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple test image with PIL
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a "messy form" sample
        img = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(img)

        # Add some "form-like" content
        draw.text((50, 30), "SAMPLE FORM - FOR TESTING", fill="black")
        draw.line([(50, 60), (750, 60)], fill="black", width=2)

        draw.text((50, 80), "Name: _____________________", fill="black")
        draw.text((50, 120), "Date: _____________________", fill="black")
        draw.text((50, 160), "ID Number: ________________", fill="black")

        draw.text((50, 220), "Comments:", fill="black")
        draw.rectangle([(50, 250), (750, 400)], outline="black")

        # Add some "noise" - gray speckles
        import random
        for _ in range(500):
            x = random.randint(0, 799)
            y = random.randint(0, 599)
            gray = random.randint(180, 220)
            draw.point((x, y), fill=(gray, gray, gray))

        img.save(samples_dir / "sample_form.png")
        print(f"  Created: {samples_dir / 'sample_form.png'}")

    except ImportError:
        print("  PIL not available, skipping sample creation")

    # Create a README
    readme = samples_dir / "README.md"
    readme.write_text("""# Sample Documents

Place your own test documents here for quick experimentation.

Supported formats: PNG, JPG, JPEG, TIFF, BMP

The sample_form.png is a synthetic test document for verifying the pipeline works.
""")
    print(f"  Created: {readme}")


def main():
    """Download all datasets."""
    print("=" * 50)
    print("chaos-eval Dataset Downloader")
    print("=" * 50)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_funsd()
    download_sroie()
    download_nod()
    create_sample_docs()

    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)

    datasets = {
        "FUNSD": DATA_DIR / "funsd" / "dataset",
        "SROIE": DATA_DIR / "sroie",
        "NOD": DATA_DIR / "nod",
        "Samples": DATA_DIR / "samples",
    }

    for name, path in datasets.items():
        status = "✓ Ready" if path.exists() and any(path.iterdir()) else "✗ Not ready"
        print(f"  {name}: {status}")

    print("\nRun experiments with: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()
