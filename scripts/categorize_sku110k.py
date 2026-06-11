"""

Heurística: "desordem" = densidade de arestas (Laplacian-like via PIL FIND_EDGES)
combinada com desvio-padrão de cor. Imagens com prateleiras muito cheias e
caóticas tendem a ter scores altos; prateleiras arrumadas e uniformes, scores baixos.

Esta categorização automática é um PONTO DE PARTIDA - o aluno deve fazer uma
revisão manual rápida antes da entrega final (mover algumas imagens entre pastas
se a heurística errar obviamente).

Uso:
    python scripts/categorize_sku110k.py
"""
import os
import shutil

import numpy as np
from PIL import Image, ImageFilter

SRC_DIR = "data/images/raw_sku110k"
OUT_BASE = "data/images"

N_NORMAL = 150
N_DIRTY = 80
N_AMBIGUOUS = 70


def clutter_score(path):
    img = Image.open(path).convert("L").resize((256, 256))
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges, dtype=np.float32)
    edge_density = edge_arr.std()

    color_img = Image.open(path).convert("RGB").resize((256, 256))
    color_arr = np.asarray(color_img, dtype=np.float32)
    color_std = color_arr.std()

    return 0.7 * edge_density + 0.3 * color_std


def main():
    files = sorted(
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"A calcular scores para {len(files)} imagens...")

    scored = []
    for i, f in enumerate(files):
        path = os.path.join(SRC_DIR, f)
        try:
            score = clutter_score(path)
        except Exception as e:
            print(f"  erro em {f}: {e}")
            continue
        scored.append((f, score))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}")

    scored.sort(key=lambda x: x[1])  # ascendente de menos caotico para mais mais caotico

    n = len(scored)
    normal = scored[:N_NORMAL]
    dirty = scored[-N_DIRTY:]

    # ambiguous
    mid_start = (n - N_AMBIGUOUS) // 2
    remaining_pool = scored[N_NORMAL: n - N_DIRTY]
    mid = len(remaining_pool) // 2
    half = N_AMBIGUOUS // 2
    ambiguous = remaining_pool[max(0, mid - half): mid - half + N_AMBIGUOUS]

    used_names = {f for f, _ in normal} | {f for f, _ in dirty} | {f for f, _ in ambiguous}
    leftover = [item for item in scored if item[0] not in used_names]

    buckets = {
        "normal": normal,
        "dirty": dirty,
        "ambiguous": ambiguous,
        "leftover_for_synthetic": leftover,
    }

    for bucket_name, items in buckets.items():
        out_dir = os.path.join(OUT_BASE, bucket_name)
        os.makedirs(out_dir, exist_ok=True)
        for f, score in items:
            shutil.copy(os.path.join(SRC_DIR, f), os.path.join(out_dir, f))
        print(f"{bucket_name}: {len(items)} imagens - {out_dir}")

    print("\nIntervalos de score (min - max) por bucket:")
    for bucket_name, items in buckets.items():
        if items:
            scores = [s for _, s in items]
            print(f"  {bucket_name}: {min(scores):.2f} - {max(scores):.2f}")


if __name__ == "__main__":
    main()
