"""Dataset loaders for FUNSD, SROIE, and NOD."""

import json
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
import base64

from PIL import Image


@dataclass
class Document:
    """A document with image and optional ground truth."""
    id: str
    image_path: Path
    image: Image.Image | None = None
    ground_truth: dict | None = None
    metadata: dict | None = None
    dataset: str = ""

    def load_image(self) -> Image.Image:
        """Load the image if not already loaded."""
        if self.image is None:
            self.image = Image.open(self.image_path)
        return self.image

    def to_base64(self) -> str:
        """Convert image to base64 for API calls."""
        import io
        img = self.load_image()
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def __repr__(self):
        return f"Document(id={self.id}, dataset={self.dataset}, path={self.image_path.name})"


class DataLoader:
    """Load documents from various datasets."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)

    def load_funsd(self, split: str = "test") -> Iterator[Document]:
        """
        Load FUNSD dataset.

        FUNSD has 199 forms split into train (149) and test (50).
        Each form has annotations for: question, answer, header, other.
        """
        funsd_dir = self.data_dir / "funsd" / "dataset" / f"testing_data" if split == "test" else self.data_dir / "funsd" / "dataset" / "training_data"

        if not funsd_dir.exists():
            raise FileNotFoundError(
                f"FUNSD not found at {funsd_dir}. Run `python src/download_data.py` first."
            )

        images_dir = funsd_dir / "images"
        annotations_dir = funsd_dir / "annotations"

        for img_path in sorted(images_dir.glob("*.png")):
            doc_id = img_path.stem
            ann_path = annotations_dir / f"{doc_id}.json"

            ground_truth = None
            if ann_path.exists():
                with open(ann_path, encoding="utf-8") as f:
                    ann_data = json.load(f)
                    # Parse FUNSD format into our format
                    ground_truth = self._parse_funsd_annotation(ann_data)

            yield Document(
                id=doc_id,
                image_path=img_path,
                ground_truth=ground_truth,
                dataset="funsd",
                metadata={"split": split}
            )

    def _parse_funsd_annotation(self, ann_data: dict) -> dict:
        """Parse FUNSD annotation format."""
        result = {
            "header": [],
            "question": [],
            "answer": [],
            "other": [],
            "key_value_pairs": []
        }

        form = ann_data.get("form", [])
        id_to_text = {}
        id_to_label = {}

        for item in form:
            text = item.get("text", "")
            label = item.get("label", "other")
            item_id = item.get("id")

            if text.strip():
                result[label].append(text)
                id_to_text[item_id] = text
                id_to_label[item_id] = label

        # Build key-value pairs from linking
        for item in form:
            if item.get("label") == "question":
                for linked_id in item.get("linking", []):
                    if isinstance(linked_id, list):
                        linked_id = linked_id[1] if len(linked_id) > 1 else linked_id[0]
                    if linked_id in id_to_text and id_to_label.get(linked_id) == "answer":
                        result["key_value_pairs"].append({
                            "key": item.get("text", ""),
                            "value": id_to_text[linked_id]
                        })

        return result

    def load_sroie(self, split: str = "test") -> Iterator[Document]:
        """
        Load SROIE dataset.

        SROIE has receipt images with ground truth for:
        company, address, date, total
        """
        sroie_dir = self.data_dir / "sroie" / split

        if not sroie_dir.exists():
            raise FileNotFoundError(
                f"SROIE not found at {sroie_dir}. Run `python src/download_data.py` first."
            )

        img_dir = sroie_dir / "img"

        for img_path in sorted(img_dir.glob("*.jpg")):
            doc_id = img_path.stem

            # Ground truth files
            entities_path = sroie_dir / "entities" / f"{doc_id}.txt"

            ground_truth = None
            if entities_path.exists():
                ground_truth = self._parse_sroie_entities(entities_path)

            yield Document(
                id=doc_id,
                image_path=img_path,
                ground_truth=ground_truth,
                dataset="sroie",
                metadata={"split": split}
            )

    def _parse_sroie_entities(self, entities_path: Path) -> dict:
        """Parse SROIE entities file (company, address, date, total)."""
        result = {
            "company": None,
            "address": None,
            "date": None,
            "total": None
        }

        with open(entities_path, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        if len(lines) >= 4:
            result["company"] = lines[0] if lines[0] else None
            result["address"] = lines[1] if lines[1] else None
            result["date"] = lines[2] if lines[2] else None
            result["total"] = lines[3] if lines[3] else None

        return result

    def load_nod(self, noise_level: str = "original") -> Iterator[Document]:
        """
        Load NOD (Noisy OCR Dataset).

        NOD has the same documents at different noise levels,
        perfect for testing chaos tolerance.

        noise_level options:
        - "original": Clean original
        - "noise_1" through "noise_5": Increasing noise
        - "blur_1" through "blur_5": Increasing blur
        - etc.
        """
        nod_dir = self.data_dir / "nod"

        if not nod_dir.exists():
            raise FileNotFoundError(
                f"NOD not found at {nod_dir}. Run `python src/download_data.py` first."
            )

        # NOD structure varies - adapt to actual downloaded structure
        target_dir = nod_dir / noise_level if (nod_dir / noise_level).exists() else nod_dir

        for img_path in sorted(target_dir.glob("*.png")) or sorted(target_dir.glob("*.jpg")):
            doc_id = img_path.stem

            # NOD includes ground truth text files
            gt_path = img_path.with_suffix(".txt")
            ground_truth = None
            if gt_path.exists():
                with open(gt_path, encoding="utf-8") as f:
                    ground_truth = {"text": f.read().strip()}

            yield Document(
                id=doc_id,
                image_path=img_path,
                ground_truth=ground_truth,
                dataset="nod",
                metadata={"noise_level": noise_level}
            )

    def load_samples(self) -> Iterator[Document]:
        """Load sample documents for quick testing."""
        samples_dir = self.data_dir / "samples"

        if not samples_dir.exists():
            samples_dir.mkdir(parents=True, exist_ok=True)
            return

        for img_path in sorted(samples_dir.glob("*.*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                yield Document(
                    id=img_path.stem,
                    image_path=img_path,
                    dataset="samples"
                )

    def get_dataset(self, name: str, **kwargs) -> Iterator[Document]:
        """Get documents from a named dataset."""
        loaders = {
            "funsd": self.load_funsd,
            "sroie": self.load_sroie,
            "nod": self.load_nod,
            "samples": self.load_samples,
        }

        if name.lower() not in loaders:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

        return loaders[name.lower()](**kwargs)


def list_available_datasets(data_dir: str | Path = "data") -> dict[str, bool]:
    """Check which datasets are available."""
    data_dir = Path(data_dir)

    return {
        "funsd": (data_dir / "funsd" / "dataset").exists(),
        "sroie": (data_dir / "sroie").exists(),
        "nod": (data_dir / "nod").exists(),
        "samples": (data_dir / "samples").exists(),
    }
