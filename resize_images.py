#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image


def resize_image(input_path: Path, output_dir: Path, width: int, height: int):
    with Image.open(input_path) as image:
        resized = image.resize((width, height), Image.Resampling.LANCZOS)

        output_path = output_dir / input_path.name
        resized.save(output_path, quality=95)

        print(f"{input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize multiple JPG images."
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input JPG files"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width (default: 512)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height (default: 512)"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("resized"),
        help="Output directory (default: ./resized)"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in args.files:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        try:
            resize_image(
                file_path,
                args.output_dir,
                args.width,
                args.height
            )
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")


if __name__ == "__main__":
    main()
