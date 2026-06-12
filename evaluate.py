"""
Harness de avaliacao do Retail Vision Intelligence System (Seccao 9 do
enunciado).

Agrega as metricas dos varios componentes num unico relatorio JSON:

1. shelf_inspector (Componente 1): metricas de qualidade da analise visual
   (Issue Detection Rate, False Positive Rate, Severity Accuracy, JSON Parse
   Rate, Hallucination Rate), calculadas:
   - se --images-dir for indicado, sobre as imagens desse diretorio,
     comparando com data/ground_truth.json (por nome de ficheiro);
   - caso contrario, a partir de data/evaluation/strategy_comparison.json
     (ja calculado por scripts/compare_strategies.py para as estrategias
     A/B/C sobre as 15 imagens de ground truth).
2. rule_engine (Componente 2): estatisticas de execucao das regras ativas
   sobre o historico de inspecoes, mais Rule Parse Rate, Rule Correctness e
   Ambiguity Detection sobre TODAS as regras em data/rules/ (validas e
   invalidas).
3. rag_memory (Componente 3): Recall@3 por estrategia de chunking
   (data/evaluation/rag_recall.json, calculado por
   src/rag_memory.py --evaluate), mais Faithfulness e Answer Relevance sobre
   as 4 perguntas obrigatorias da Seccao 6.4 (rag_memory.OBLIGATORY_QUERIES).
4. LLM-as-judge (opcional, --llm-judge): usa um modelo Gemini para avaliar
   (1-5) a qualidade da saida do shelf_inspector face ao ground truth, numa
   pequena amostra de imagens (--judge-n), reutilizando os records ja em
   cache (sem reanalisar imagens).

Uso (invocacao obrigatoria da Seccao 9.1):
    python evaluate.py --images-dir test_images/ --output evaluation_report.json

Outras invocacoes:
    python evaluate.py
    python evaluate.py --llm-judge --judge-n 3 --judge-strategy A
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rag_memory import load_inspection_records, answer_query, OBLIGATORY_QUERIES  # noqa: E402
from rule_engine import load_rules, evaluate_rules  # noqa: E402
from shelf_inspector import analyze_image, QuotaManager, IMAGE_EXTENSIONS  # noqa: E402

PROMPTS_DIR = ROOT_DIR / "prompts"
EVALUATION_DIR = ROOT_DIR / "data" / "evaluation"
GROUND_TRUTH_PATH = ROOT_DIR / "data" / "ground_truth.json"
COMPARE_DIR = ROOT_DIR / "data" / "dev_artifacts" / "_compare"

STRATEGY_COMPARISON_PATH = EVALUATION_DIR / "strategy_comparison.json"
RAG_RECALL_PATH = EVALUATION_DIR / "rag_recall.json"

JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3-flash-preview")


# --------------------------------------------------------------------------
# Componente 1: shelf_inspector - analise visual
# --------------------------------------------------------------------------

def load_shelf_inspector_metrics() -> dict | None:
    """Metricas pre-calculadas (estrategias A/B/C) - usado quando nao e dado
    --images-dir."""
    if not STRATEGY_COMPARISON_PATH.exists():
        return None
    with open(STRATEGY_COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {strategy: data[strategy]["metrics"] for strategy in data if strategy in ("A", "B", "C")}


def evaluate_images_dir(images_dir: str, strategy: str = "B") -> dict:
    """Analisa todas as imagens de `images_dir` com `strategy` e calcula as
    metricas de analise visual da Seccao 9.2, comparando com
    data/ground_truth.json sempre que o nome do ficheiro coincida com uma
    imagem de ground truth (a maioria das imagens de teste vem do conjunto
    de 15 imagens anotadas)."""
    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise FileNotFoundError(f"Diretorio de imagens nao encontrado: {images_dir}")

    images = sorted(
        p for p in images_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    gt_by_name: dict[str, dict] = {}
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        for gt in gt_data["images"]:
            gt_by_name[Path(gt["image_path"]).name] = gt

    api_key = os.environ.get("GEMINI_API_KEY")
    from google import genai
    client = genai.Client(api_key=api_key) if api_key else None
    quota = QuotaManager()

    total = len(images)
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
    n_with_gt = 0

    per_image = []

    for image_path in images:
        record = analyze_image(str(image_path), zone_id="Z_CMP", strategy=strategy, client=client, quota=quota)

        is_api_error = record.get("_fallback") and "JSON" not in (record.get("_error") or "") and record.get("_error")
        is_parse_error = record.get("_fallback") and "JSON" in (record.get("_error") or "")
        if not is_api_error:
            api_ok += 1
        if not is_parse_error and not is_api_error:
            parse_ok += 1

        pred_issues = [(i.get("type"), i.get("severity")) for i in (record.get("issues") or [])]
        pred_types = {t for t, _ in pred_issues}

        gt = gt_by_name.get(image_path.name)
        if gt is not None:
            n_with_gt += 1
            gt_issues = [(i["type"], i["severity"]) for i in gt.get("issues", [])]
            gt_types = {t for t, _ in gt_issues}

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

        per_image.append({
            "image": str(image_path),
            "gt_status": gt.get("overall_status") if gt else None,
            "pred_status": record.get("overall_status"),
            "gt_issues": [i["type"] for i in gt.get("issues", [])] if gt else None,
            "pred_issues": sorted(pred_types),
            "fallback": record.get("_fallback", False),
            "from_cache": record.get("_from_cache", False),
        })

    return {
        "strategy": strategy,
        "n_images": total,
        "n_images_with_ground_truth": n_with_gt,
        "api_success_rate": (api_ok / total) if total else None,
        "json_parse_rate": (parse_ok / api_ok) if api_ok else None,
        "issue_detection_rate": (detected_count / gt_issue_count) if gt_issue_count else None,
        "false_positive_rate": (false_positive_count / pred_issue_count) if pred_issue_count else (0.0 if pred_issue_count == 0 and n_with_gt else None),
        "severity_accuracy": (severity_correct_count / matched_type_count) if matched_type_count else None,
        "hallucination_rate": (hallucinated_ok / ok_images) if ok_images else None,
        "per_image": per_image,
    }


# --------------------------------------------------------------------------
# Componente 2: rule_engine
# --------------------------------------------------------------------------

def evaluate_rule_engine() -> dict:
    active_rules = load_rules(active_only=True)
    all_rules = load_rules(active_only=False)
    records = load_inspection_records()

    n_records = len(records)
    n_with_match = 0
    matches_per_rule = {
        rule["rule_id"]: {"description": rule.get("description"), "n_matches": 0}
        for rule in active_rules
    }

    for record in records:
        evaluations = evaluate_rules(record, active_rules)
        matched_any = False
        for ev in evaluations:
            if ev["matched"]:
                matched_any = True
                matches_per_rule[ev["rule_id"]]["n_matches"] += 1
        if matched_any:
            n_with_match += 1

    # Rule Parse Rate: fracao de regras em data/rules/ com a estrutura
    # completa do schema da Seccao 5.3 (conditions/action/validation).
    required_keys = {"natural_language", "description", "conditions", "action", "validation"}
    parsed_ok = sum(1 for r in all_rules if required_keys.issubset(r.keys()))
    rule_parse_rate = (parsed_ok / len(all_rules)) if all_rules else None

    # Rule Correctness: das regras consideradas validas pelo conversor
    # (validation.is_valid=true), fracao cujas condicoes sao acionaveis
    # (pelo menos uma condicao de disparo nao-nula).
    valid_rules = [r for r in all_rules if r.get("validation", {}).get("is_valid")]
    actionable_keys = ("issue_types", "severity_threshold", "fill_rate_threshold", "location_filter")
    actionable = sum(
        1 for r in valid_rules
        if any(r.get("conditions", {}).get(k) for k in actionable_keys)
    )
    rule_correctness = (actionable / len(valid_rules)) if valid_rules else None

    # Ambiguity Detection: das regras marcadas como invalidas, fracao que
    # tem pelo menos uma ambiguidade documentada (justificacao do porque a
    # conversao nao foi aceite).
    invalid_rules = [r for r in all_rules if not r.get("validation", {}).get("is_valid")]
    ambiguity_documented = sum(1 for r in invalid_rules if r.get("validation", {}).get("ambiguities"))
    ambiguity_detection_rate = (ambiguity_documented / len(invalid_rules)) if invalid_rules else None

    return {
        "n_active_rules": len(active_rules),
        "n_total_rules": len(all_rules),
        "n_records_evaluated": n_records,
        "n_records_with_match": n_with_match,
        "match_rate": (n_with_match / n_records) if n_records else None,
        "matches_per_rule": matches_per_rule,
        "rule_parse_rate": rule_parse_rate,
        "rule_correctness": rule_correctness,
        "n_invalid_rules": len(invalid_rules),
        "ambiguity_detection_rate": ambiguity_detection_rate,
    }


# --------------------------------------------------------------------------
# Componente 3: rag_memory (Recall@3, Faithfulness, Answer Relevance)
# --------------------------------------------------------------------------

def load_rag_recall_metrics() -> dict | None:
    if not RAG_RECALL_PATH.exists():
        return None
    with open(RAG_RECALL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        strategy: data[strategy]["avg_recall_at_k"]
        for strategy in data if strategy != "_meta"
    } | {"_meta": data.get("_meta")}


_INSPECTION_ID_RE = re.compile(r"INS_[A-Za-z0-9_]+")

_NO_INFO_MARKERS = (
    "nao foram encontrados registos relevantes",
    "informacao disponivel nao",
    "nao e possivel responder",
)


def evaluate_rag_answers(client=None, queries: list[str] | None = None) -> dict:
    """Faithfulness e Answer Relevance (Seccao 9.2) sobre as 4 perguntas
    obrigatorias da Seccao 6.4, usando rag_memory.answer_query.

    - Faithfulness: fracao de perguntas em que TODOS os inspection_id
      mencionados explicitamente na resposta correspondem a registos
      efetivamente recuperados (sources) - deteta "alucinacao" de
      inspection_ids inexistentes.
    - Answer Relevance: fracao de perguntas para as quais foi devolvida uma
      resposta nao-vazia que nao seja a mensagem generica de "sem registos".
    """
    queries = queries or OBLIGATORY_QUERIES

    n = 0
    faithful = 0
    relevant = 0
    per_query = []

    for q in queries:
        try:
            result = answer_query(q, k=3, strategy="hybrid", client=client)
        except Exception as exc:  # noqa: BLE001
            per_query.append({"query": q, "error": str(exc)})
            continue

        n += 1
        answer = result.get("answer", "")
        source_ids = {s["inspection_id"] for s in result.get("sources", [])}
        cited_ids = set(_INSPECTION_ID_RE.findall(answer))

        is_faithful = cited_ids.issubset(source_ids)
        if is_faithful:
            faithful += 1

        is_relevant = bool(answer.strip()) and not any(m in answer.lower() for m in _NO_INFO_MARKERS)
        if is_relevant:
            relevant += 1

        per_query.append({
            "query": q,
            "answer": answer,
            "n_sources": len(result.get("sources", [])),
            "cited_ids_not_in_sources": sorted(cited_ids - source_ids),
            "faithful": is_faithful,
            "relevant": is_relevant,
        })

    return {
        "n_queries": n,
        "faithfulness": (faithful / n) if n else None,
        "answer_relevance": (relevant / n) if n else None,
        "per_query": per_query,
    }


# --------------------------------------------------------------------------
# LLM-as-judge (opcional)
# --------------------------------------------------------------------------

def build_judge_prompt(ground_truth: dict, system_output: dict) -> str:
    template = (PROMPTS_DIR / "llm_judge.txt").read_text(encoding="utf-8")
    gt_text = json.dumps({
        "overall_status": ground_truth.get("overall_status"),
        "issues": ground_truth.get("issues"),
        "shelf_fill_rate": ground_truth.get("shelf_fill_rate"),
        "notes": ground_truth.get("notes"),
    }, ensure_ascii=False, indent=2)
    sys_text = json.dumps({
        "overall_status": system_output.get("overall_status"),
        "issues": system_output.get("issues"),
        "shelf_fill_rate": system_output.get("shelf_fill_rate"),
        "model_reasoning": (system_output.get("model_reasoning") or "")[:1500],
    }, ensure_ascii=False, indent=2)
    return template.replace("{ground_truth}", gt_text).replace("{system_output}", sys_text)


def extract_json(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def find_cached_record(image_path: str, strategy: str) -> dict | None:
    """Procura um inspection record ja existente (em data/inspections/_compare)
    para a imagem e estrategia indicadas, sem reanalisar a imagem."""
    if not COMPARE_DIR.exists():
        return None
    for path in COMPARE_DIR.glob(f"*_{strategy}.json"):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        if record.get("image_path") == image_path and not record.get("_fallback"):
            return record
    return None


def run_llm_judge(judge_n: int, judge_strategy: str) -> dict:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY nao definido em .env"}
    client = genai.Client(api_key=api_key)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_images = gt_data["images"][:judge_n]

    results = []
    for gt in gt_images:
        record = find_cached_record(gt["image_path"], judge_strategy)
        if record is None:
            results.append({"image": gt["image_path"], "error": "sem record em cache para esta estrategia"})
            continue

        prompt = build_judge_prompt(gt, record)
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
            judgment = extract_json(response.text)
        except Exception as e:  # noqa: BLE001
            results.append({"image": gt["image_path"], "error": str(e)})
            continue

        results.append({
            "image": gt["image_path"],
            "score": judgment.get("score"),
            "justification": judgment.get("justification"),
        })

    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    return {
        "model": JUDGE_MODEL,
        "n_judged": len(scores),
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "per_image": results,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harness de avaliacao do Retail Vision Intelligence System")
    parser.add_argument("--images-dir", type=str, default=None,
                         help="diretorio de imagens a analisar para as metricas do Componente 1 (ex: test_images/)")
    parser.add_argument("--strategy", type=str, default="B", choices=["A", "B", "C"],
                         help="estrategia de prompting usada com --images-dir")
    parser.add_argument("--output", type=str, default=str(EVALUATION_DIR / "evaluation_report.json"))
    parser.add_argument("--llm-judge", action="store_true", help="ativa avaliacao LLM-as-judge (chamadas adicionais ao Gemini)")
    parser.add_argument("--judge-n", type=int, default=3, help="numero de imagens a avaliar com LLM-as-judge")
    parser.add_argument("--judge-strategy", type=str, default="A", choices=["A", "B", "C"],
                         help="estrategia (records em cache) usada para o LLM-as-judge")
    args = parser.parse_args()

    report: dict = {}

    print("=== Componente 1: shelf_inspector (analise visual) ===")
    if args.images_dir:
        images_metrics = evaluate_images_dir(args.images_dir, strategy=args.strategy)
        report["shelf_inspector"] = images_metrics
        print(f"  [{images_metrics['strategy']}] n_images={images_metrics['n_images']} "
              f"(com ground truth: {images_metrics['n_images_with_ground_truth']})")
        print(f"  api_success_rate={images_metrics['api_success_rate']} "
              f"json_parse_rate={images_metrics['json_parse_rate']} "
              f"issue_detection_rate={images_metrics['issue_detection_rate']} "
              f"false_positive_rate={images_metrics['false_positive_rate']} "
              f"severity_accuracy={images_metrics['severity_accuracy']} "
              f"hallucination_rate={images_metrics['hallucination_rate']}")
    else:
        shelf_inspector_metrics = load_shelf_inspector_metrics()
        report["shelf_inspector"] = shelf_inspector_metrics
        if shelf_inspector_metrics is not None:
            for strategy, metrics in shelf_inspector_metrics.items():
                print(f"  [{strategy}] api_success_rate={metrics['api_success_rate']:.2f} "
                      f"json_parse_rate={metrics['json_parse_rate']:.2f} "
                      f"issue_detection_rate={metrics['issue_detection_rate']}")
        else:
            print("  (sem data/evaluation/strategy_comparison.json e sem --images-dir)")

    print("\n=== Componente 2: rule_engine ===")
    rule_engine_metrics = evaluate_rule_engine()
    report["rule_engine"] = rule_engine_metrics
    print(f"  Regras ativas: {rule_engine_metrics['n_active_rules']} / total: {rule_engine_metrics['n_total_rules']}")
    print(f"  Inspecoes avaliadas: {rule_engine_metrics['n_records_evaluated']}")
    if rule_engine_metrics["match_rate"] is not None:
        print(f"  Inspecoes com pelo menos 1 regra acionada: {rule_engine_metrics['n_records_with_match']} "
              f"({rule_engine_metrics['match_rate']:.1%})")
    for rule_id, info in rule_engine_metrics["matches_per_rule"].items():
        print(f"    - {rule_id}: {info['n_matches']} acionamentos")
    print(f"  Rule Parse Rate: {rule_engine_metrics['rule_parse_rate']}")
    print(f"  Rule Correctness: {rule_engine_metrics['rule_correctness']}")
    print(f"  Ambiguity Detection Rate: {rule_engine_metrics['ambiguity_detection_rate']} "
          f"(regras invalidas: {rule_engine_metrics['n_invalid_rules']})")

    print("\n=== Componente 3: rag_memory ===")
    rag_recall_metrics = load_rag_recall_metrics()
    if rag_recall_metrics is not None:
        for strategy in ("document", "issue", "hybrid"):
            if strategy in rag_recall_metrics:
                print(f"  [{strategy}] recall@3={rag_recall_metrics[strategy]:.2f}")
    else:
        print("  (sem data/evaluation/rag_recall.json - corre src/rag_memory.py --evaluate)")

    api_key = os.environ.get("GEMINI_API_KEY")
    rag_client = None
    if api_key:
        from google import genai
        quota = QuotaManager()
        if not quota.daily_exhausted():
            rag_client = genai.Client(api_key=api_key)
    rag_answer_metrics = evaluate_rag_answers(client=rag_client)
    print(f"  Faithfulness: {rag_answer_metrics['faithfulness']}")
    print(f"  Answer Relevance: {rag_answer_metrics['answer_relevance']}")

    report["rag_memory"] = {
        "recall_at_3": rag_recall_metrics,
        "faithfulness": rag_answer_metrics["faithfulness"],
        "answer_relevance": rag_answer_metrics["answer_relevance"],
        "n_queries_evaluated": rag_answer_metrics["n_queries"],
        "per_query": rag_answer_metrics["per_query"],
    }

    if args.llm_judge:
        print("\n=== LLM-as-judge ===")
        judge_results = run_llm_judge(args.judge_n, args.judge_strategy)
        report["llm_judge"] = judge_results
        if "error" in judge_results:
            print(f"  Erro: {judge_results['error']}")
        else:
            print(f"  Modelo: {judge_results['model']} | n_judged={judge_results['n_judged']} "
                  f"| avg_score={judge_results['avg_score']}")
            for r in judge_results["per_image"]:
                if "error" in r:
                    print(f"    {r['image']}: erro - {r['error']}")
                else:
                    print(f"    {r['image']}: score={r['score']} - {r['justification']}")
    else:
        report["llm_judge"] = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRelatorio de avaliacao guardado em {out_path}")


if __name__ == "__main__":
    main()
