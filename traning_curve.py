from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
	r"Epoch \[(?P<epoch>\d+)/\d+\], Train Loss: (?P<train_loss>[0-9.]+), Val Loss: (?P<val_loss>[0-9.]+)"
)


def parse_training_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
	epochs: list[int] = []
	train_losses: list[float] = []
	val_losses: list[float] = []

	for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
		match = EPOCH_PATTERN.search(line)
		if match is None:
			continue

		epochs.append(int(match.group("epoch")))
		train_losses.append(float(match.group("train_loss")))
		val_losses.append(float(match.group("val_loss")))

	if not epochs:
		raise ValueError(f"No epoch/loss lines were found in {log_path}")

	return epochs, train_losses, val_losses


def plot_training_curve(log_path: Path, output_path: Path) -> None:
	epochs, train_losses, val_losses = parse_training_log(log_path)

	plt.figure(figsize=(10, 6))
	plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
	plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training Curve")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot training and validation loss curves from a log file.")
	parser.add_argument("--log", default="traning_log.txt", help="Path to the training log file")
	parser.add_argument("--output", default="training_curve.png", help="Path to save the plot image")
	args = parser.parse_args()

	log_path = Path(args.log)
	output_path = Path(args.output)
	plot_training_curve(log_path, output_path)
	print(f"Saved training curve to {output_path.resolve()}")


if __name__ == "__main__":
	main()
