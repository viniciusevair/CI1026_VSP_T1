#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
GAUSSIAN_KERNEL = (5, 5)
SIZE_256 = (256, 256)
SIZE_128 = (128, 128)


def save_image(image, output_path: Path) -> None:
    """Save an image, creating its parent directory when necessary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError("OpenCV não conseguiu salvar o arquivo")


def convert_to_grayscale(
    input_path: Path, output_path: Path
):
    """Convert one image to grayscale and save it at output_path."""
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("arquivo não pôde ser lido pelo OpenCV")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(grayscale, output_path)
    return grayscale


def resize_with_gaussian(image, size: tuple[int, int]):
    """Apply Gaussian smoothing and resize an image to the requested size."""
    blurred = cv2.GaussianBlur(image, GAUSSIAN_KERNEL, 0)
    return cv2.resize(blurred, size, interpolation=cv2.INTER_AREA)


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    """Return supported image files in input_dir."""
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Converte imagens para grayscale e gera versões 256x256 e 128x128."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="diretório que contém as imagens de entrada",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="diretório de saída (padrão: ./output)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="também procura imagens em subdiretórios",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"diretório de entrada não existe: {args.input_dir}")

    images = find_images(args.input_dir, args.recursive)
    if not images:
        print(f"Nenhuma imagem encontrada em {args.input_dir}")
        return 0

    converted = 0
    for input_path in images:
        relative_path = input_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        output_256_path = args.output_dir / "256x256" / relative_path
        output_128_path = args.output_dir / "128x128" / relative_path

        try:
            grayscale = convert_to_grayscale(input_path, output_path)
            image_256 = resize_with_gaussian(grayscale, SIZE_256)
            save_image(image_256, output_256_path)
            image_128 = resize_with_gaussian(image_256, SIZE_128)
            save_image(image_128, output_128_path)
        except (OSError, ValueError, cv2.error) as exc:
            print(f"Falha ao processar {input_path}: {exc}")
            continue

        print(
            f"{input_path} -> {output_path}, "
            f"{output_256_path}, {output_128_path}"
        )
        converted += 1

    print(f"{converted}/{len(images)} imagem(ns) convertida(s).")
    return 0 if converted == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
