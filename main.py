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


def convert_to_grayscale(
    input_path: Path, output_path: Path
) -> None:
    """Convert one image to grayscale and save it at output_path."""
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("arquivo não pôde ser lido pelo OpenCV")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), grayscale):
        raise OSError("OpenCV não conseguiu salvar o arquivo")


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
        description="Converte todas as imagens de um diretório para grayscale."
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
        help="diretório de saída (padrão: ./grayscale)",
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

        try:
            convert_to_grayscale(input_path, output_path)
        except (OSError, ValueError) as exc:
            print(f"Falha ao processar {input_path}: {exc}")
            continue

        print(f"{input_path} -> {output_path}")
        converted += 1

    print(f"{converted}/{len(images)} imagem(ns) convertida(s).")
    return 0 if converted == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
