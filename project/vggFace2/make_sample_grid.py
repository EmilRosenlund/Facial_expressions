import argparse
import os
from PIL import Image


def find_latest_samples_dir(samples_root):
    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"Samples root not found: {samples_root}")

    candidates = []
    for name in os.listdir(samples_root):
        path = os.path.join(samples_root, name)
        if os.path.isdir(path) and name.startswith("samples_"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No sample folders found in: {samples_root}")

    return max(candidates, key=os.path.getmtime)


def get_sample_images(samples_dir, limit):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(
        f
        for f in os.listdir(samples_dir)
        if f.lower().endswith(image_exts) and f.lower().startswith("sample_")
    )

    if len(files) < limit:
        raise RuntimeError(
            f"Need at least {limit} sample images in {samples_dir}, found {len(files)}"
        )

    return [os.path.join(samples_dir, f) for f in files[:limit]]


def build_grid(image_paths, output_path, rows=5, cols=4, margin=8, bg_color=(245, 245, 245)):
    images = [Image.open(path).convert("RGB") for path in image_paths]

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    canvas_w = cols * cell_w + (cols + 1) * margin
    canvas_h = rows * cell_h + (rows + 1) * margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        x0 = margin + col * (cell_w + margin)
        y0 = margin + row * (cell_h + margin)

        x = x0 + (cell_w - img.width) // 2
        y = y0 + (cell_h - img.height) // 2
        canvas.paste(img, (x, y))

    canvas.save(output_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    parser = argparse.ArgumentParser(
        description="Create one grid image from 20 sample prediction images (4 columns x 5 rows)."
    )
    parser.add_argument(
        "--samples-root",
        default=os.path.join(root_dir, "random_test_samples"),
        help="Root folder containing sample run folders (default: random_test_samples).",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="Specific samples folder to use. If omitted, the latest samples_* folder is used.",
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of rows (default: 5).")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns (default: 4).")
    parser.add_argument("--margin", type=int, default=8, help="Margin in pixels (default: 8).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: <samples_dir>/grid_4x5.jpg",
    )

    args = parser.parse_args()

    samples_dir = args.samples_dir or find_latest_samples_dir(args.samples_root)
    total_needed = args.rows * args.cols
    image_paths = get_sample_images(samples_dir, total_needed)

    output_path = args.output or os.path.join(samples_dir, f"grid_{args.cols}x{args.rows}.jpg")
    build_grid(
        image_paths=image_paths,
        output_path=output_path,
        rows=args.rows,
        cols=args.cols,
        margin=max(0, args.margin),
    )

    print(f"Samples folder: {samples_dir}")
    print(f"Saved grid image: {output_path}")
