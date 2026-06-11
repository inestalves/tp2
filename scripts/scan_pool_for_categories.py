"""
Procura, no pool de candidatos (data/images/pool_candidates/, imagens reais
do SKU-110K ainda nao usadas), imagens que correspondam as categorias
"prateleira vazia" e "violacao de planograma", usando:

1. Heuristica de "desordem visual" (a mesma de categorize_sku110k.py) para
   pre-ordenar o pool: imagens muito uniformes -> candidatas a "vazia";
   imagens muito caoticas -> candidatas a "violacao".
2. shelf_inspector.py (Gemini, estrategia B) para confirmar via analise real:
   - "vazia": issue do tipo empty_shelf com affected_area_pct >= 0.08
   - "violacao": issue do tipo damaged|misaligned|wrong_product|label_missing

As imagens confirmadas sao copiadas para data/images/empty/ e
data/images/planogram_violation/ (ate atingir os alvos), e os inspection
records sao guardados em data/inspections/_scan/ para referencia.

Uso:
    python scripts/scan_pool_for_categories.py --target-empty 100 --target-violation 100 --max-scan 250
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.shelf_inspector import analyze_image, QuotaManager  # noqa: E402
from google import genai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

POOL_DIR = "data/images/pool_candidates"
EMPTY_DIR = "data/images/empty"
VIOLATION_DIR = "data/images/planogram_violation"
SCAN_RECORDS_DIR = "data/inspections/_scan"

VIOLATION_TYPES = {"damaged", "misaligned", "wrong_product", "label_missing"}


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
    parser.add_argument("--max-scan", type=int, default=250, help="max imagens a enviar a LLM por categoria")
    args = parser.parse_args()

    os.makedirs(EMPTY_DIR, exist_ok=True)
    os.makedirs(VIOLATION_DIR, exist_ok=True)
    os.makedirs(SCAN_RECORDS_DIR, exist_ok=True)

    existing_empty = len([f for f in os.listdir(EMPTY_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    existing_violation = len([f for f in os.listdir(VIOLATION_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Ja existentes: empty={existing_empty}, planogram_violation={existing_violation}")

    files = sorted(
        f for f in os.listdir(POOL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"A calcular scores de desordem para {len(files)} imagens do pool...")

    scored = []
    for i, f in enumerate(files):
        path = os.path.join(POOL_DIR, f)
        try:
            score = clutter_score(path)
        except Exception as e:
            print(f"  erro em {f}: {e}")
            continue
        scored.append((f, score))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(files)}")

    scored.sort(key=lambda x: x[1])  # ascendente

    n = len(scored)
    empty_pool = [f for f, _ in scored[: int(n * 0.4)]]   # mais uniformes
    violation_pool = [f for f, _ in scored[int(n * 0.6):]]  # mais caoticas

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None
    quota = QuotaManager()

    found_empty = existing_empty
    found_violation = existing_violation

    def scan(pool, target, found_count, out_dir, match_fn, label):
        scanned = 0
        for f in pool:
            if found_count >= target or scanned >= args.max_scan:
                break
            if quota.daily_exhausted():
                print("  Quota diaria esgotada, a parar scan.")
                break

            path = os.path.join(POOL_DIR, f)
            record = analyze_image(path, zone_id="Z_SCAN", strategy="B", client=client, quota=quota)
            scanned += 1

            if record.get("_fallback"):
                print(f"  [{label}] {f}: fallback ({record.get('_error')})")
                continue

            if match_fn(record):
                shutil.copy(path, os.path.join(out_dir, f))
                with open(os.path.join(SCAN_RECORDS_DIR, f"{record['inspection_id']}.json"), "w", encoding="utf-8") as out:
                    json.dump(record, out, ensure_ascii=False, indent=2)
                found_count += 1
                print(f"  [{label}] MATCH ({found_count}/{target}): {f} -> {record['overall_status']}")
            elif scanned % 10 == 0:
                print(f"  [{label}] scanned={scanned}, found={found_count}/{target}")

        return found_count, scanned

    def is_empty_match(record):
        for issue in record.get("issues", []) or []:
            if issue.get("type") == "empty_shelf" and (issue.get("affected_area_pct") or 0) >= 0.08:
                return True
        return False

    def is_violation_match(record):
        for issue in record.get("issues", []) or []:
            if issue.get("type") in VIOLATION_TYPES:
                return True
        return False

    print(f"\nA procurar 'vazia' (alvo={args.target_empty}, ja temos {found_empty})...")
    found_empty, scanned_empty = scan(empty_pool, args.target_empty, found_empty, EMPTY_DIR, is_empty_match, "vazia")

    print(f"\nA procurar 'violacao' (alvo={args.target_violation}, ja temos {found_violation})...")
    found_violation, scanned_violation = scan(violation_pool, args.target_violation, found_violation, VIOLATION_DIR, is_violation_match, "violacao")

    print("\n=== RESUMO ===")
    print(f"vazia: {found_empty}/{args.target_empty} (scanned {scanned_empty})")
    print(f"violacao: {found_violation}/{args.target_violation} (scanned {scanned_violation})")
    print(f"Quota diaria restante: {quota.daily_remaining()}")


if __name__ == "__main__":
    main()
