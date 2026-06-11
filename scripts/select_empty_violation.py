import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageFilter

POOL_DIRS = ["data/images/pool_candidates", "data/images/pool_candidates2"]
EMPTY_DIR = "data/images/empty"
VIOLATION_DIR = "data/images/planogram_violation"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-empty", type=int, default=100)
    parser.add_argument("--target-violation", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(EMPTY_DIR, exist_ok=True)
    os.makedirs(VIOLATION_DIR, exist_ok=True)

    existing_empty = {f for f in os.listdir(EMPTY_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    existing_violation = {f for f in os.listdir(VIOLATION_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    print(f"Ja existentes: empty={len(existing_empty)}, planogram_violation={len(existing_violation)}")

    files = []
    for pool_dir in POOL_DIRS:
        if not os.path.isdir(pool_dir):
            continue
        for f in sorted(os.listdir(pool_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append((pool_dir, f))

    print(f"A calcular scores de desordem para {len(files)} imagens do pool...")

    scored = []
    for i, (pool_dir, f) in enumerate(files):
        path = os.path.join(pool_dir, f)
        try:
            score = clutter_score(path)
        except Exception as e:
            print(f"  erro em {f}: {e}")
            continue
        scored.append((pool_dir, f, score))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(files)}")

    scored.sort(key=lambda x: x[2])  # ascendente: mais uniforme -> mais caotico

    need_empty = args.target_empty - len(existing_empty)
    need_violation = args.target_violation - len(existing_violation)

    # candidatos "vazia": scores mais baixos, ainda nao usados
    empty_added = 0
    for pool_dir, f, score in scored:
        if empty_added >= need_empty:
            break
        if f in existing_empty or f in existing_violation:
            continue
        shutil.copy(os.path.join(pool_dir, f), os.path.join(EMPTY_DIR, f))
        existing_empty.add(f)
        empty_added += 1

    # candidatos "violacao": scores mais altos, ainda nao usados
    violation_added = 0
    for pool_dir, f, score in reversed(scored):
        if violation_added >= need_violation:
            break
        if f in existing_empty or f in existing_violation:
            continue
        shutil.copy(os.path.join(pool_dir, f), os.path.join(VIOLATION_DIR, f))
        existing_violation.add(f)
        violation_added += 1

    print("\nRESUMO")
    print(f"empty: +{empty_added} - total {len(existing_empty)}/{args.target_empty}")
    print(f"planogram_violation: +{violation_added} - total {len(existing_violation)}/{args.target_violation}")


if __name__ == "__main__":
    main()
