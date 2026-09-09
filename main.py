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
KMEANS_CLUSTERS = 4
ImageFilter = Callable[[np.ndarray], np.ndarray]


def make_kernel_filter(kernel: np.ndarray) -> ImageFilter:
    """Create a filter function from a convolution kernel."""
    def apply_kernel(image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, cv2.CV_32F, kernel)

    return apply_kernel


FILTERS = [
    make_kernel_filter(
        np.array(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32
        )
    ),  # F1 horizontal edge
    make_kernel_filter(
        np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
        )
    ),  # F2 vertical edge
    make_kernel_filter(
        np.array(
            [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32
        )
    ),  # F3 45-degree edge
    make_kernel_filter(
        np.array(
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32
        )
    ),  # F4 135-degree edge
    make_kernel_filter(
        np.array(
            [
                [0, -1, -1, -1, 0],
                [-1, -1, -2, -1, -1],
                [-1, -2, 16, -2, -1],
                [-1, -1, -2, -1, -1],
                [0, -1, -1, -1, 0],
            ],
            dtype=np.float32,
        )
    ),  # F5 circular / spot
    make_kernel_filter(
        np.array(
            [[1, -1, 1], [-1, -4, -1], [1, -1, 1]], dtype=np.float32
        )
    ),  # F6 corner / junction
    make_kernel_filter(
        cv2.getGaborKernel(
            (9, 9), 2.0, np.pi / 4, 5.0, 0.5, 0, cv2.CV_32F
        )
    ),  # F7 wave / ripple
    make_kernel_filter(
        np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
        )
    ),  # F8 Laplacian / high-frequency detail
]


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

    Each filter is applied to a three-level Gaussian pyramid: the original
    image, then a Gaussian-smoothed version reduced by half in each dimension,
    followed by another Gaussian smoothing and reduction by half. The window
    is reduced proportionally at each scale, so the default sizes are 32x32,
    16x16 and 8x8. For filter ``i`` and scale ``j``, the value is stored at
    position ``3 * i + j``. Each value is the mean absolute filter response
    inside that window, which measures texture energy without cancellation
    between positive and negative filter responses.

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
    scales = [image]
    for _ in range(2):
        previous_scale = scales[-1]
        previous_height, previous_width = previous_scale.shape[:2]
        smoothed = cv2.GaussianBlur(previous_scale, GAUSSIAN_KERNEL, 0)
        scales.append(
            cv2.resize(
                smoothed,
                (max(1, previous_width // 2), max(1, previous_height // 2)),
                interpolation=cv2.INTER_AREA,
            )
        )

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
                    vector.append(float(np.mean(np.abs(window))))
            vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32)


def segment_image_by_windows(
    image: np.ndarray,
    filters: Sequence[ImageFilter],
    k: int = KMEANS_CLUSTERS,
    window_size: int = 32,
) -> np.ndarray:
    """Segment an image by clustering its window feature vectors."""
    feature_vectors = build_window_feature_vectors(
        image, filters, window_size
    )
    if len(feature_vectors) < k:
        raise ValueError("a imagem não possui janelas suficientes para o K-means")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )
    _, labels, _ = cv2.kmeans(
        feature_vectors,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    height, width = image.shape[:2]
    rows = height // window_size
    columns = width // window_size
    labels = labels.reshape(rows, columns)
    cluster_values = np.round(np.linspace(0, 255, k)).astype(np.uint8)
    segmented = np.zeros((height, width), dtype=np.uint8)

    for row in range(rows):
        for column in range(columns):
            y = row * window_size
            x = column * window_size
            segmented[y : y + window_size, x : x + window_size] = (
                cluster_values[labels[row, column]]
            )

    return segmented


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    """Return supported image files in input_dir."""
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image_directory(
    input_dir: Path,
    output_dir: Path,
    filters: Sequence[ImageFilter] = FILTERS,
    window_size: int = 32,
    k: int = KMEANS_CLUSTERS,
    recursive: bool = False,
) -> tuple[int, int]:
    """Process every image and return ``(processed, total)``.

    The function is reusable by other scripts: callers can provide their own
    filter functions and window size. Each image produces grayscale, resized
    and K-means segmented outputs under ``output_dir``.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"diretório de entrada não existe: {input_dir}")
    if not filters:
        raise ValueError("filters deve conter pelo menos um filtro")

    images = find_images(input_dir, recursive)
    if not images:
        print(f"Nenhuma imagem encontrada em {input_dir}")
        return 0, 0

    processed = 0
    for input_path in images:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_256_path = output_dir / "256x256" / relative_path
        output_128_path = output_dir / "128x128" / relative_path
        segmented_path = output_dir / "segmented" / relative_path

        try:
            grayscale = convert_to_grayscale(input_path, output_path)
            image_256 = resize_with_gaussian(grayscale, SIZE_256)
            save_image(image_256, output_256_path)
            image_128 = resize_with_gaussian(image_256, SIZE_128)
            save_image(image_128, output_128_path)
            segmented = segment_image_by_windows(
                grayscale, filters, k, window_size
            )
            save_image(segmented, segmented_path)
        except (OSError, ValueError, cv2.error) as exc:
            print(f"Falha ao processar {input_path}: {exc}")
            continue

        print(
            f"{input_path} -> {output_path}, "
            f"{output_256_path}, {output_128_path}, {segmented_path}"
        )
        processed += 1

    print(f"{processed}/{len(images)} imagem(ns) convertida(s).")
    return processed, len(images)


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
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="tamanho da janela base (padrão: 32x32)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"diretório de entrada não existe: {args.input_dir}")

    processed, total = process_image_directory(
        args.input_dir,
        args.output_dir,
        filters=FILTERS,
        window_size=args.window_size,
        k=KMEANS_CLUSTERS,
        recursive=args.recursive,
    )
    return 0 if processed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
