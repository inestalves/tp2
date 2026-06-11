"""
Componente 4: Report Generator

Gera relatorios de inspecao em Markdown para uma SESSAO (conjunto de
inspection records - ex: todas as inspecoes de hoje, ou todas as inspecoes
de uma zona num periodo), combinando:
- as regras de deteccao acionadas (rule_engine.py);
- contexto historico semelhante recuperado do RAG memory (rag_memory.py),
  com referencia explicita a inspection_id e data;
- recomendacoes geradas por template a partir dos problemas/regras
  detetados, ordenadas por urgencia e limitadas a 5 (sem chamadas
  adicionais ao Gemini, dado o limite de quota documentado).

Estrutura do relatorio (Seccao 7 do enunciado):
1. Sumario Executivo
2. Problemas por Zona
3. Regras Disparadas
4. Contexto Historico Relevante
5. Recomendacoes (max. 5, por ordem de urgencia)
6. Integracao com Projeto 1 (opcional - nao implementado)

Uso:
    python src/report_generator.py --inspection data/inspections/INS_xxx.json
    python src/report_generator.py --inspections-dir data/inspections --output data/reports/sessao.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rule_engine import evaluate_rules, load_rules
from rag_memory import query_memory, build_record_text

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "data" / "reports"

SEVERITY_LABELS = {"low": "baixa", "medium": "media", "high": "alta"}

ISSUE_LABELS = {
    "empty_shelf": "Prateleira vazia",
    "wrong_product": "Produto fora de posicao",
    "damaged": "Produto danificado",
    "misaligned": "Produtos desalinhados",
    "label_missing": "Etiqueta em falta",
    "other": "Outro problema",
}

STATUS_LABELS = {
    "ok": "OK",
    "warning": "Atencao",
    "critical": "Critico",
    "unknown": "Desconhecido",
}

# Ordem de urgencia (menor = mais urgente) para issues e niveis de alerta.
ISSUE_URGENCY = {"empty_shelf": 0, "damaged": 1, "wrong_product": 2, "misaligned": 3, "label_missing": 4, "other": 5}
ALERT_URGENCY = {"critical": 0, "warning": 1, "info": 2}

RECOMMENDATION_TEMPLATES = {
    "empty_shelf": "Reabastecer com urgencia a zona {zone} (prateleira vazia detetada).",
    "wrong_product": "Reorganizar os produtos fora de posicao na zona {zone} de acordo com o planograma.",
    "damaged": "Remover e substituir os produtos danificados na zona {zone}.",
    "misaligned": "Reajustar a arrumacao dos produtos na zona {zone} para melhorar o alinhamento.",
    "label_missing": "Repor as etiquetas de preco/identificacao em falta na zona {zone}.",
    "other": "Verificar manualmente o problema assinalado na zona {zone}.",
}


def _format_pct(value: float | None) -> str:
    """Formata uma percentagem (preenchimento ou area afetada). Alguns
    records antigos tem valores 0-100 em vez de 0.0-1.0; este helper trata
    ambos os formatos."""
    if value is None:
        return "N/D"
    if value > 1:
        return f"{value:.0f}%"
    return f"{value:.0%}"


# --------------------------------------------------------------------------
# Seccoes
# --------------------------------------------------------------------------

def _section_executive_summary(records: list[dict]) -> str:
    n_zones = len({r.get("zone_id", "?") for r in records})
    n_critical = sum(1 for r in records if r.get("overall_status") == "critical")
    n_warning = sum(1 for r in records if r.get("overall_status") == "warning")
    n_ok = sum(1 for r in records if r.get("overall_status") == "ok")
    n_issues = sum(len(r.get("issues") or []) for r in records)

    text = (
        f"Nesta sessao foram analisadas {len(records)} inspecao(oes) abrangendo "
        f"{n_zones} zona(s). Estado geral: {n_critical} em estado critico, "
        f"{n_warning} em atencao e {n_ok} sem problemas. "
        f"No total foram detetados {n_issues} problema(s) entre todas as inspecoes."
    )
    return "## 1. Sumario Executivo\n\n" + text


def _section_zones(records: list[dict]) -> str:
    lines = ["## 2. Problemas por Zona", ""]

    by_zone: dict[str, list[dict]] = {}
    for r in records:
        by_zone.setdefault(r.get("zone_id", "?"), []).append(r)

    for zone, recs in sorted(by_zone.items()):
        lines.append(f"### Zona {zone}")
        for r in sorted(recs, key=lambda x: x.get("timestamp", "")):
            status = STATUS_LABELS.get(r.get("overall_status"), r.get("overall_status"))
            fill = _format_pct(r.get("shelf_fill_rate"))
            date = (r.get("timestamp") or "")[:10] or "N/D"
            lines.append(
                f"- Inspecao `{r.get('inspection_id', 'N/D')}` ({date}): "
                f"estado **{status}**, preenchimento {fill}"
            )
            issues = r.get("issues") or []
            if issues:
                for issue in issues:
                    label = ISSUE_LABELS.get(issue.get("type"), issue.get("type", "other"))
                    sev = SEVERITY_LABELS.get(issue.get("severity"), issue.get("severity", "N/D"))
                    loc = issue.get("location", "N/D")
                    area = _format_pct(issue.get("affected_area_pct"))
                    lines.append(f"  - {label} (severidade {sev}, local: {loc}, area afetada: {area})")
            else:
                lines.append("  - Sem problemas detetados.")
        lines.append("")

    return "\n".join(lines)


def _section_rules(records: list[dict], rules: list[dict]) -> tuple[str, list[dict]]:
    lines = ["## 3. Regras Disparadas", ""]
    matched: list[dict] = []

    if not rules:
        lines.append("Nao existem regras ativas (validation.is_valid=true) configuradas.")
        return "\n".join(lines), matched

    for r in records:
        evaluations = evaluate_rules(r, rules)
        for ev in evaluations:
            if ev["matched"]:
                matched.append({**ev, "zone_id": r.get("zone_id"),
                                 "inspection_id": r.get("inspection_id"),
                                 "date": (r.get("timestamp") or "")[:10]})

    if not matched:
        lines.append(f"Nenhuma das {len(rules)} regra(s) ativa(s) foi acionada nesta sessao.")
    else:
        for ev in matched:
            lines.append(
                f"- `{ev['rule_id']}` [{ev['alert_level']}] em `{ev['inspection_id']}` "
                f"(zona {ev['zone_id']}, {ev['date']}): {ev['notification_message']}"
            )

    return "\n".join(lines), matched


def _section_history(records: list[dict]) -> str:
    lines = ["## 4. Contexto Historico Relevante", ""]
    current_ids = {r.get("inspection_id") for r in records}
    any_history = False

    for r in records:
        if not (r.get("issues") or r.get("overall_status") in ("warning", "critical")):
            continue
        try:
            results = query_memory(build_record_text(r), n_results=4, strategy="hybrid")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"(Nao foi possivel consultar a memoria historica: {exc})")
            return "\n".join(lines)

        results = [x for x in results if x["inspection_id"] not in current_ids][:2]
        if not results:
            continue

        any_history = True
        date = (r.get("timestamp") or "")[:10] or "N/D"
        lines.append(f"**Zona {r.get('zone_id')}** (inspecao `{r.get('inspection_id')}`, {date}):")
        for x in results:
            xdate = (x.get("timestamp") or "")[:10] or "N/D"
            xstatus = STATUS_LABELS.get(x["overall_status"], x["overall_status"])
            lines.append(
                f"  - Inspecao semelhante `{x['inspection_id']}` ({xdate}, zona {x['zone_id']}, "
                f"estado {xstatus})"
            )

    if not any_history:
        lines.append("Sem historico relevante para os problemas detetados nesta sessao.")

    return "\n".join(lines)


def _section_recommendations(records: list[dict], matched_evals: list[dict]) -> str:
    candidates: list[tuple[int, str]] = []

    for r in records:
        zone = r.get("zone_id", "?")
        for issue in r.get("issues") or []:
            issue_type = issue.get("type", "other")
            urgency = ISSUE_URGENCY.get(issue_type, 5)
            template = RECOMMENDATION_TEMPLATES.get(issue_type, RECOMMENDATION_TEMPLATES["other"])
            candidates.append((urgency, template.format(zone=zone)))

        fill = r.get("shelf_fill_rate")
        if fill is not None and fill < 0.7:
            candidates.append((0, f"Planear reposicao de stock na zona {zone} (preenchimento {fill:.0%})."))

    for ev in matched_evals:
        urgency = ALERT_URGENCY.get(ev.get("alert_level"), 3)
        candidates.append((urgency, f"[{ev['rule_id']}] {ev['notification_message']}"))

    seen: set[str] = set()
    ordered: list[str] = []
    for _, text in sorted(candidates, key=lambda c: c[0]):
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= 5:
            break

    if not ordered:
        ordered = ["Nenhuma acao adicional necessaria. Manter rotina de inspecao habitual."]

    lines = ["## 5. Recomendacoes", ""]
    for rec in ordered:
        lines.append(f"- {rec}")
    return "\n".join(lines)


def _section_integration() -> str:
    return (
        "## 6. Integracao com Projeto 1\n\n"
        "Seccao opcional do enunciado - nao implementada nesta entrega. "
        "A integracao consistiria em cruzar os inspection records com a "
        "trajetoria de movimento de clientes do Projeto 1 (ex: tempo de "
        "permanencia por zona vs. problemas detetados nessa zona)."
    )


# --------------------------------------------------------------------------
# Geracao do relatorio completo
# --------------------------------------------------------------------------

def generate_session_report(records: list[dict], title: str = "Relatorio de Inspecao") -> str:
    """Gera o relatorio Markdown de sessao (6 seccoes) para um conjunto de
    inspection records."""
    rules = load_rules(active_only=True)
    rules_section, matched_evals = _section_rules(records, rules)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    inspection_ids = ", ".join(r.get("inspection_id", "N/D") for r in records) if records else "(nenhuma)"

    sections = [
        f"# {title}",
        f"*Gerado em: {generated_at}*",
        f"*Inspecoes incluidas: {inspection_ids}*",
        "",
        _section_executive_summary(records) if records else "## 1. Sumario Executivo\n\nNenhuma inspecao encontrada para os criterios indicados.",
        "",
        _section_zones(records),
        "",
        rules_section,
        "",
        _section_history(records),
        "",
        _section_recommendations(records, matched_evals),
        "",
        _section_integration(),
    ]
    return "\n".join(sections) + "\n"


def generate_report_for_records(records: list[dict], output_path: str | None,
                                  title: str = "Relatorio de Inspecao") -> str:
    report = generate_session_report(records, title=title)

    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"session_{stamp}.md")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path


def generate_report_for_path(inspection_path: str, output_path: str | None = None) -> str:
    """Compatibilidade: gera um relatorio de sessao com um unico inspection
    record."""
    with open(inspection_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(REPORTS_DIR / f"{record.get('inspection_id', Path(inspection_path).stem)}.md")

    return generate_report_for_records([record], output_path,
                                        title=f"Relatorio de Inspecao - {record.get('inspection_id', 'N/D')}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Report Generator - relatorios de inspecao em Markdown")
    parser.add_argument("--inspection", type=str, help="caminho para um unico inspection record (.json)")
    parser.add_argument("--inspections-dir", type=str, help="diretorio com varios inspection records (.json) a incluir na sessao")
    parser.add_argument("--output", type=str, help="caminho do ficheiro Markdown de saida")
    args = parser.parse_args()

    if args.inspection:
        output_path = generate_report_for_path(args.inspection, args.output)
    elif args.inspections_dir:
        records = []
        for path in sorted(Path(args.inspections_dir).glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        output_path = generate_report_for_records(records, args.output, title="Relatorio de Sessao")
    else:
        parser.error("indica --inspection <path> ou --inspections-dir <dir>")
        return

    print(f"Relatorio gerado em: {output_path}")


if __name__ == "__main__":
    main()
