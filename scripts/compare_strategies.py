"""
Compara as 3 estrategias de prompting do shelf_inspector (A = zero-shot,
B = chain-of-thought, C = few-shot) sobre as 15 imagens de
data/ground_truth.json, calculando as metricas da Seccao 9 do enunciado:

- JSON Parse Rate: % de respostas que resultaram em JSON valido (sem fallback
  por erro de parsing).
- Issue Detection Rate (recall): de entre os tipos de issue presentes no
  ground truth de uma imagem, quantos foram tambem reportados pelo modelo.
- False Positive Rate: de entre os tipos de issue reportados pelo modelo,
  quantos NAO estao no ground truth dessa imagem.
- Severity Accuracy: para issues cujo tipo coincide entre ground truth e
  previsao, em quantos casos a severidade tambem coincide.
- Hallucination Rate: fracao de imagens "ok" (sem issues no ground truth) em
  que o modelo reportou pelo menos um issue (falso alarme).

Usa o modelo gemini-2.5-flash-lite (env GEMINI_MODEL), com quota diaria
separada da do gemini-2.5-flash ja usada no scan do dataset.

Uso:
    python scripts/compare_strategies.py
"""
import json
import os
import sys

os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash-lite")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.shelf_inspector import analyze_image, QuotaManager  # noqa: E402
from google import genai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

GROUND_TRUTH_PATH = "data/ground_truth.json"
OUT_DIR = "data/inspections/_compare"
RESULTS_PATH = "data/evaluation/strategy_comparison.json"

STRATEGIES = ["A", "B", "C"]


def predicted_issue_types(record):
    return [(issue.get("type"), issue.get("severity")) for issue in (record.get("issues") or [])]


def evaluate_strategy(strategy, ground_truth_images, client, quota):
    os.makedirs(OUT_DIR, exist_ok=True)

    total = len(ground_truth_images)
    api_ok = 0
    parse_ok = 0

    gt_issue_count = 0
    detected_count = 0

    pred_issue_count = 0
    false_positive_count = 0

    matched_type_count = 0
    severity_correct_count = 0

    ok_images = 0
    hallucinated_ok = 0

    per_image = []

    for gt in ground_truth_images:
        path = gt["image_path"]
        record = analyze_image(path, zone_id="Z_CMP", strategy=strategy, client=client, quota=quota)

        is_api_error = record.get("_fallback") and "JSON" not in (record.get("_error") or "") and record.get("_error")
        is_parse_error = record.get("_fallback") and "JSON" in (record.get("_error") or "")
        if not is_api_error:
            api_ok += 1
        if not is_parse_error and not is_api_error:
            parse_ok += 1

        gt_issues = [(i["type"], i["severity"]) for i in gt.get("issues", [])]
        pred_issues = predicted_issue_types(record)

        gt_types = {t for t, _ in gt_issues}
        pred_types = {t for t, _ in pred_issues}

        gt_issue_count += len(gt_types)
        detected_count += len(gt_types & pred_types)

        pred_issue_count += len(pred_types)
        false_positive_count += len(pred_types - gt_types)

        for t, sev in gt_issues:
            for pt, psev in pred_issues:
                if pt == t:
                    matched_type_count += 1
                    if psev == sev:
                        severity_correct_count += 1
                    break

        if not gt_issues:
            ok_images += 1
            if pred_issues:
                hallucinated_ok += 1

        out_path = os.path.join(OUT_DIR, f"{record['inspection_id']}_{strategy}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        per_image.append({
            "image": path,
            "gt_status": gt["overall_status"],
            "pred_status": record.get("overall_status"),
            "gt_issues": gt_issues,
            "pred_issues": pred_issues,
            "fallback": record.get("_fallback", False),
            "from_cache": record.get("_from_cache", False),
        })

        print(f"  [{strategy}] {os.path.basename(path)}: gt={gt['overall_status']}/{gt_types or '-'} "
              f"-> pred={record.get('overall_status')}/{pred_types or '-'}"
              f"{' [FALLBACK]' if record.get('_fallback') else ''}")

    metrics = {
        "api_success_rate": api_ok / total,
        "json_parse_rate": (parse_ok / api_ok) if api_ok else None,
        "issue_detection_rate": (detected_count / gt_issue_count) if gt_issue_count else None,
        "false_positive_rate": (false_positive_count / pred_issue_count) if pred_issue_count else 0.0,
        "severity_accuracy": (severity_correct_count / matched_type_count) if matched_type_count else None,
        "hallucination_rate": (hallucinated_ok / ok_images) if ok_images else None,
        "n_images": total,
        "n_ok_images": ok_images,
        "n_gt_issue_types": gt_issue_count,
        "n_pred_issue_types": pred_issue_count,
    }

    return metrics, per_image


def main():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    images = gt_data["images"]

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None
    quota = QuotaManager()

    results = {}
    for strategy in STRATEGIES:
        print(f"\n=== Estrategia {strategy} ===")
        metrics, per_image = evaluate_strategy(strategy, images, client, quota)
        results[strategy] = {"metrics": metrics, "per_image": per_image}
        print(f"  Metricas: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMO COMPARATIVO ===")
    print(f"{'Metrica':<25}" + "".join(f"{s:>12}" for s in STRATEGIES))
    metric_keys = ["api_success_rate", "json_parse_rate", "issue_detection_rate", "false_positive_rate",
                    "severity_accuracy", "hallucination_rate"]
    for mk in metric_keys:
        row = f"{mk:<25}"
        for s in STRATEGIES:
            v = results[s]["metrics"][mk]
            row += f"{(f'{v:.2f}' if v is not None else 'N/A'):>12}"
        print(row)

    print(f"\nQuota diaria restante: {quota.daily_remaining()}")
    print(f"Resultados guardados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
