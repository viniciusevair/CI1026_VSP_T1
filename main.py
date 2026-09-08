#!/usr/bin/env python3

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np


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
ImageFilter = Callable[[np.ndarray], np.ndarray]


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


def build_window_feature_vectors(
    image: np.ndarray,
    filters: Sequence[ImageFilter],
    window_size: int = 32,
) -> np.ndarray:
    """Build one 3*N feature vector for every image window.

    Each filter is applied to the original image and to two reduced scales
    (half and one quarter of the original dimensions). The window is reduced
    proportionally at each scale, so the default sizes are 32x32, 16x16 and
    8x8. For filter ``i`` and scale ``j``, the value is stored at position
    ``3 * i + j``.

    Filters must receive one image and return an image with the same height
    and width. The returned array has one row per complete window.
    """
    if image is None or image.ndim < 2:
        raise ValueError("image deve ser um array com pelo menos duas dimensões")
    if not filters:
        raise ValueError("filters deve conter pelo menos um filtro")
    if window_size <= 0:
        raise ValueError("window_size deve ser maior que zero")

    height, width = image.shape[:2]
    scales = [
        image,
        cv2.resize(
            image,
            (max(1, width // 2), max(1, height // 2)),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(
            image,
            (max(1, width // 4), max(1, height // 4)),
            interpolation=cv2.INTER_AREA,
        ),
    ]

    filtered_scales = []
    for scaled_image in scales:
        filtered_images = []
        for filter_function in filters:
            filtered_image = filter_function(scaled_image)
            if filtered_image is None or filtered_image.ndim < 2:
                raise ValueError("cada filtro deve retornar uma imagem válida")
            if filtered_image.shape[:2] != scaled_image.shape[:2]:
                raise ValueError(
                    "cada filtro deve preservar as dimensões da imagem"
                )
            filtered_images.append(filtered_image)
        filtered_scales.append(filtered_images)

    rows = height // window_size
    columns = width // window_size
    vectors = []

    for row in range(rows):
        for column in range(columns):
            vector = []
            for filter_index in range(len(filters)):
                for scale_index, filtered_images in enumerate(filtered_scales):
                    scaled_height, scaled_width = scales[scale_index].shape[:2]
                    scaled_window_height = max(
                        1, round(window_size * scaled_height / height)
                    )
                    scaled_window_width = max(
                        1, round(window_size * scaled_width / width)
                    )
                    y = row * scaled_window_height
                    x = column * scaled_window_width
                    window = filtered_images[filter_index][
                        y : y + scaled_window_height,
                        x : x + scaled_window_width,
                    ]
                    vector.append(float(np.mean(window)))
            vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32)


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
