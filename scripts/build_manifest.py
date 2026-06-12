import csv
import os

BASE = "data/images"

CATEGORIES = {
    "normal": {
        "label": "Prateleira normal",
        "source": "SKU-110K (Goldman et al., 2019)",
        "license": "Academico / nao-comercial (ver SKU110K_CVPR19 license)",
        "synthetic": False,
        "selection_method": "heuristica de desordem visual (score baixo)",
    },
    "dirty": {
        "label": "Prateleira suja / desordenada",
        "source": "SKU-110K (Goldman et al., 2019)",
        "license": "Academico / nao-comercial (ver SKU110K_CVPR19 license)",
        "synthetic": False,
        "selection_method": "heuristica de desordem visual (score alto)",
    },
    "ambiguous": {
        "label": "Caso ambiguo",
        "source": "SKU-110K (Goldman et al., 2019)",
        "license": "Academico / nao-comercial (ver SKU110K_CVPR19 license)",
        "synthetic": False,
        "selection_method": "heuristica de desordem visual (score intermedio)",
    },
    "empty": {
        "label": "Prateleira vazia (total ou parcial)",
        "source": "SKU-110K (Goldman et al., 2019)",
        "license": "Academico / nao-comercial (ver SKU110K_CVPR19 license)",
        "synthetic": False,
        "selection_method": "heuristica de desordem visual (score muito baixo)",
    },
    "planogram_violation": {
        "label": "Violacao de planograma",
        "source": "SKU-110K (Goldman et al., 2019)",
        "license": "Academico / nao-comercial (ver SKU110K_CVPR19 license)",
        "synthetic": False,
        "selection_method": "heuristica de desordem visual (score muito alto)",
    },
}

# Imagens validadas individualmente pelo shelf_inspector (Gemini)
LLM_VALIDATED = {
    "empty": {"train_1853.jpg", "train_4098.jpg", "train_5414.jpg", "train_6437.jpg"},
    "planogram_violation": {"train_3691.jpg", "train_4435.jpg"},
}


def main():
    rows = []
    for category, meta in CATEGORIES.items():
        cat_dir = os.path.join(BASE, category)
        if not os.path.isdir(cat_dir):
            continue
        files = sorted(
            f for f in os.listdir(cat_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for f in files:
            selection_method = meta["selection_method"]
            if f in LLM_VALIDATED.get(category, set()):
                selection_method = "validada por LLM (shelf_inspector, estrategia B)"
            rows.append({
                "image_path": os.path.join(cat_dir, f).replace("\\", "/"),
                "category": category,
                "category_label": meta["label"],
                "source": meta["source"],
                "license": meta["license"],
                "synthetic": meta["synthetic"],
                "selection_method": selection_method,
            })

    out_path = os.path.join(BASE, "manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "category", "category_label", "source", "license", "synthetic", "selection_method"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest gerado: {out_path} ({len(rows)} imagens)")

    print("\nDistribuicao por categoria:")
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    print(f"  TOTAL: {len(rows)}")


if __name__ == "__main__":
    main()
