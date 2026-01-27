"""Utility functions for chaos-eval."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


console = Console()


def save_results(results: list[dict], filepath: str | Path, append: bool = False):
    """Save results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if append and filepath.exists():
        with open(filepath) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            results = existing + results

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: str | Path) -> list[dict]:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def generate_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_metrics_table(metrics: list[dict], title: str = "Results"):
    """Pretty print metrics as a table."""
    if not metrics:
        console.print("[yellow]No metrics to display[/yellow]")
        return

    table = Table(title=title)

    # Add columns from first result
    columns = list(metrics[0].keys())
    for col in columns:
        table.add_column(col)

    # Add rows
    for m in metrics:
        row = [str(m.get(col, "")) for col in columns]
        table.add_row(*row)

    console.print(table)


def format_percentage(value: float) -> str:
    """Format a decimal as percentage."""
    return f"{value * 100:.1f}%"


def truncate_string(s: str, max_length: int = 50) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


class ExperimentLogger:
    """Simple logger for experiment tracking."""

    def __init__(self, experiment_id: str | None = None, log_dir: str | Path = "results"):
        self.experiment_id = experiment_id or generate_experiment_id()
        self.log_dir = Path(log_dir) / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "experiment.log"
        self.results: list[dict] = []

        self.log(f"Experiment started: {self.experiment_id}")

    def log(self, message: str):
        """Log a message."""
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] {message}\n"

        with open(self.log_file, "a") as f:
            f.write(line)

        console.print(f"[dim]{timestamp}[/dim] {message}")

    def add_result(self, result: dict):
        """Add a result to the experiment."""
        self.results.append(result)

    def save(self):
        """Save all results."""
        save_results(self.results, self.log_dir / "results.json")
        self.log(f"Saved {len(self.results)} results")

    def summary(self) -> dict:
        """Get experiment summary."""
        return {
            "experiment_id": self.experiment_id,
            "num_results": len(self.results),
            "log_dir": str(self.log_dir),
        }


def add_noise_to_image(image, noise_type: str = "gaussian", intensity: float = 0.1):
    """
    Add noise to an image for chaos testing.

    noise_type: "gaussian", "salt_pepper", "blur", "rotation"
    intensity: 0.0 to 1.0
    """
    import numpy as np
    from PIL import Image, ImageFilter

    img_array = np.array(image)

    if noise_type == "gaussian":
        noise = np.random.normal(0, intensity * 50, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    elif noise_type == "salt_pepper":
        noisy = img_array.copy()
        # Salt
        salt_mask = np.random.random(img_array.shape[:2]) < intensity / 2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(img_array.shape[:2]) < intensity / 2
        noisy[pepper_mask] = 0
        return Image.fromarray(noisy)

    elif noise_type == "blur":
        radius = int(intensity * 10) + 1
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    elif noise_type == "rotation":
        angle = intensity * 15  # Max 15 degrees
        return image.rotate(angle, fillcolor="white")

    else:
        return image


def create_chaos_gradient(image, num_levels: int = 5) -> list:
    """
    Create multiple versions of an image at increasing chaos levels.

    Returns list of (level, image) tuples.
    """
    from PIL import Image

    results = [(0, image.copy())]  # Level 0 = original

    for level in range(1, num_levels + 1):
        intensity = level / num_levels

        # Combine multiple noise types
        noisy = image.copy()
        noisy = add_noise_to_image(noisy, "gaussian", intensity * 0.5)
        noisy = add_noise_to_image(noisy, "blur", intensity * 0.3)

        if intensity > 0.3:
            noisy = add_noise_to_image(noisy, "salt_pepper", intensity * 0.1)

        if intensity > 0.5:
            noisy = add_noise_to_image(noisy, "rotation", intensity * 0.5)

        results.append((level, noisy))

    return results
