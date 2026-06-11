"""
Componente 5: Interface Conversacional

CLI unificada para o gestor de loja, que liga todos os componentes do
sistema (Seccao 8 do enunciado):
- shelf_inspector.py (analise visual de imagens)
- rule_engine.py (regras de deteccao em linguagem natural)
- rag_memory.py (memoria/historico semantico, com sintese LLM)
- report_generator.py (relatorios de sessao em Markdown)

Mantem um estado de sessao (data/session_state.json) com as regras
carregadas e o historico de inspecoes feitas na sessao.

Comandos:
    python src/interface.py inspect Z_S3 --image shelf_photo.jpg
    python src/interface.py inspect all --images-dir ./today_photos/
    python src/interface.py add rule "Alertar sempre que o shelf_fill_rate for inferior a 70%"
    python src/interface.py list rules
    python src/interface.py delete rule RULE_003
    python src/interface.py test rule RULE_001 --image shelf_photo.jpg
    python src/interface.py history "Quando foi a ultima vez que a zona Z_S1 teve problemas de prateleira vazia?"
    python src/interface.py compare Z_S1 Z_S3 --period "last 7 days"
    python src/interface.py report --session today
    python src/interface.py report --zone Z_S3 --period "last 14 days"

Erros nunca sao apresentados como stack traces - apenas como mensagens
amigaveis prefixadas por "Erro:".
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import shelf_inspector
import rule_engine
import rag_memory
import report_generator

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
INSPECTIONS_DIR = ROOT_DIR / "data" / "inspections"
SESSION_PATH = ROOT_DIR / "data" / "session_state.json"

STATUS_LABELS = report_generator.STATUS_LABELS

IMAGE_EXTENSIONS = shelf_inspector.IMAGE_EXTENSIONS

# Zonas usadas para atribuir um zone_id as imagens em `inspect all
# --images-dir`, quando o nome do ficheiro nao identifica a zona (atribuicao
# round-robin, documentada no relatorio).
ZONE_POOL = ["Z_S1", "Z_S2", "Z_S3", "Z_S4", "Z_S5"]


class CLIError(Exception):
    """Erro amigavel apresentado ao utilizador sem stack trace."""


# --------------------------------------------------------------------------
# Estado de sessao
# --------------------------------------------------------------------------

def load_session() -> dict:
    if SESSION_PATH.exists():
        with open(SESSION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"loaded_rules": [], "inspection_history": []}


def save_session(state: dict) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def refresh_loaded_rules(session: dict) -> dict:
    rules = rule_engine.load_rules(active_only=True)
    session["loaded_rules"] = [r["rule_id"] for r in rules]
    return session


def record_inspection_in_session(session: dict, record: dict, path: Path) -> dict:
    session.setdefault("inspection_history", [])
    session["inspection_history"].append({
        "inspection_id": record.get("inspection_id"),
        "zone_id": record.get("zone_id"),
        "timestamp": record.get("timestamp"),
        "path": str(path),
    })
    session["last_inspection_id"] = record.get("inspection_id")
    session["last_inspection_path"] = str(path)
    return session


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_client() -> genai.Client | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    quota = shelf_inspector.QuotaManager()
    if quota.daily_exhausted():
        return None
    return genai.Client(api_key=api_key)


def parse_period(period: str | None) -> tuple[datetime, datetime] | None:
    """Converte 'today', 'last N days' ou 'last N weeks' num intervalo
    (inicio, fim) em UTC. Devolve None se period for None."""
    if not period:
        return None
    p = period.strip().lower()
    now = datetime.now(timezone.utc)

    if p == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now

    m = re.match(r"last (\d+) days?$", p)
    if m:
        return now - timedelta(days=int(m.group(1))), now

    m = re.match(r"last (\d+) weeks?$", p)
    if m:
        return now - timedelta(weeks=int(m.group(1))), now

    raise CLIError(
        f"Periodo nao reconhecido: '{period}'. "
        "Usa 'today', 'last N days' ou 'last N weeks'."
    )


def _record_datetime(record: dict) -> datetime | None:
    ts = record.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def filter_records_by_period(records: list[dict], period: str | None) -> list[dict]:
    bounds = parse_period(period)
    if bounds is None:
        return records
    start, end = bounds
    out = []
    for r in records:
        dt = _record_datetime(r)
        if dt is not None and start <= dt <= end:
            out.append(r)
    return out


def load_all_records() -> list[dict]:
    return rag_memory.load_inspection_records()


# --------------------------------------------------------------------------
# Comandos: inspect
# --------------------------------------------------------------------------

def _save_record(record: dict) -> Path:
    INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INSPECTIONS_DIR / f"{record['inspection_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return out_path


def _print_inspection_result(record: dict, out_path: Path) -> None:
    status = record.get("overall_status", "?")
    n_issues = len(record.get("issues") or [])
    fonte = "cache" if record.get("_from_cache") else "API"
    print(f"Inspecao '{record['inspection_id']}' (zona {record.get('zone_id')}) concluida (fonte={fonte}).")
    print(f"  Estado: {STATUS_LABELS.get(status, status)} | "
          f"Preenchimento: {record.get('shelf_fill_rate')} | Problemas: {n_issues}")
    print(f"  Guardado em: {out_path}")


def _alert_for_record(record: dict, rules: list[dict]) -> None:
    evaluations = rule_engine.evaluate_rules(record, rules)
    rule_engine.log_execution(record, evaluations)
    for ev in evaluations:
        if ev["matched"]:
            print(f"  [ALERTA {ev['alert_level']}] ({ev['rule_id']}) {ev['notification_message']}")


def cmd_inspect(args: argparse.Namespace) -> None:
    quota = shelf_inspector.QuotaManager()
    client = get_client()
    rules = rule_engine.load_rules(active_only=True)

    session = load_session()
    refresh_loaded_rules(session)

    if args.zone == "all":
        if not args.images_dir:
            raise CLIError("'inspect all' requer --images-dir <diretorio>")
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise CLIError(f"Diretorio nao encontrado: {images_dir}")

        images = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise CLIError(f"Nenhuma imagem encontrada em {images_dir}")

        for i, image_path in enumerate(images):
            zone_id = ZONE_POOL[i % len(ZONE_POOL)]
            record = shelf_inspector.analyze_image(
                str(image_path), zone_id=zone_id, strategy=args.strategy,
                force=args.force, client=client, quota=quota,
            )
            out_path = _save_record(record)
            _print_inspection_result(record, out_path)
            _alert_for_record(record, rules)
            session = record_inspection_in_session(session, record, out_path)
        save_session(session)
        return

    if not args.image:
        raise CLIError("'inspect <ZONA>' requer --image <path>")

    record = shelf_inspector.analyze_image(
        args.image, zone_id=args.zone, strategy=args.strategy,
        force=args.force, client=client, quota=quota,
    )
    out_path = _save_record(record)
    _print_inspection_result(record, out_path)
    _alert_for_record(record, rules)

    session = record_inspection_in_session(session, record, out_path)
    save_session(session)


# --------------------------------------------------------------------------
# Comandos: rules
# --------------------------------------------------------------------------

def cmd_add_rule(args: argparse.Namespace) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise CLIError("GEMINI_API_KEY nao definido em .env")
    client = genai.Client(api_key=api_key)

    rule = rule_engine.create_rule(args.description, client=client)
    valid = rule["validation"].get("is_valid")

    print(f"Regra '{rule['rule_id']}' criada.")
    print(f"  Descricao: {rule['description']}")
    print(f"  Alerta: [{rule['action'].get('alert_level')}] {rule['action'].get('notification_message')}")
    if valid:
        print("  Validacao: sem ambiguidades detetadas.")
    else:
        print("  Validacao: AMBIGUIDADES DETETADAS (regra nao sera aplicada ate ser corrigida):")
        for amb in rule["validation"].get("ambiguities", []):
            print(f"    - {amb}")
    assumptions = rule["validation"].get("assumptions") or []
    if assumptions:
        print("  Suposicoes assumidas:")
        for a in assumptions:
            print(f"    - {a}")

    session = load_session()
    refresh_loaded_rules(session)
    save_session(session)


def cmd_list_rules(args: argparse.Namespace) -> None:
    rules = rule_engine.load_rules(active_only=False)
    if not rules:
        print("Nenhuma regra encontrada em data/rules/.")
        return
    for r in rules:
        valid = r.get("validation", {}).get("is_valid")
        tag = "valida" if valid else "com ambiguidades"
        cond = r.get("conditions", {})
        action = r.get("action", {})
        print(f"- {r['rule_id']} [{tag}] [{action.get('alert_level')}]: {r.get('description')}")
        print(f"    condicoes: {json.dumps(cond, ensure_ascii=False)}")


def cmd_delete_rule(args: argparse.Namespace) -> None:
    if rule_engine.delete_rule(args.rule_id):
        print(f"Regra '{args.rule_id}' removida.")
        session = load_session()
        refresh_loaded_rules(session)
        save_session(session)
    else:
        raise CLIError(f"Regra '{args.rule_id}' nao encontrada.")


def cmd_test_rule(args: argparse.Namespace) -> None:
    rule = rule_engine.load_rule(args.rule_id)
    if rule is None:
        raise CLIError(f"Regra '{args.rule_id}' nao encontrada.")

    quota = shelf_inspector.QuotaManager()
    client = get_client()
    record = shelf_inspector.analyze_image(
        args.image, zone_id="Z_TEST", strategy="B",
        force=False, client=client, quota=quota,
    )

    evaluation = rule_engine.evaluate_rule(rule, record)
    print(f"Regra '{rule['rule_id']}': {rule.get('description')}")
    print(f"Imagem: {args.image}")
    print(f"Resultado da inspecao: estado={record.get('overall_status')}, "
          f"preenchimento={record.get('shelf_fill_rate')}, "
          f"issues={[i.get('type') for i in (record.get('issues') or [])]}")
    if evaluation["matched"]:
        print(f"-> REGRA ACIONADA [{evaluation['alert_level']}]: {evaluation['notification_message']}")
    else:
        print("-> Regra nao acionada para esta imagem.")


# --------------------------------------------------------------------------
# Comandos: history (RAG)
# --------------------------------------------------------------------------

def cmd_history(args: argparse.Namespace) -> None:
    client = get_client()
    result = rag_memory.answer_query(args.query, k=args.n, strategy=args.strategy, client=client)

    print(f"Pergunta: {result['query']}\n")
    print(f"Resposta: {result['answer']}\n")
    if result["sources"]:
        print("Fontes:")
        for s in result["sources"]:
            print(f"  - {s['inspection_id']} ({s['date']}, zona {s['zone_id']})")
    else:
        print("(Sem fontes - corre 'python src/rag_memory.py --index' se a memoria estiver vazia.)")


# --------------------------------------------------------------------------
# Comandos: compare
# --------------------------------------------------------------------------

def _zone_stats(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "avg_fill_rate": None, "status_counts": {}, "issue_counts": {}}

    fills = [r["shelf_fill_rate"] for r in records if r.get("shelf_fill_rate") is not None]
    status_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for r in records:
        status = r.get("overall_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for issue in r.get("issues") or []:
            t = issue.get("type", "other")
            issue_counts[t] = issue_counts.get(t, 0) + 1

    return {
        "n": len(records),
        "avg_fill_rate": (sum(fills) / len(fills)) if fills else None,
        "status_counts": status_counts,
        "issue_counts": issue_counts,
    }


def cmd_compare(args: argparse.Namespace) -> None:
    all_records = load_all_records()
    period_records = filter_records_by_period(all_records, args.period)

    recs1 = [r for r in period_records if r.get("zone_id") == args.zone1]
    recs2 = [r for r in period_records if r.get("zone_id") == args.zone2]

    stats1 = _zone_stats(recs1)
    stats2 = _zone_stats(recs2)

    period_label = args.period or "todo o historico"
    print(f"Comparacao entre '{args.zone1}' e '{args.zone2}' ({period_label})\n")
    print(f"{'':<28}{args.zone1:<20}{args.zone2:<20}")
    print(f"{'Numero de inspecoes':<28}{stats1['n']:<20}{stats2['n']:<20}")

    f1 = f"{stats1['avg_fill_rate']:.0%}" if stats1["avg_fill_rate"] is not None else "N/D"
    f2 = f"{stats2['avg_fill_rate']:.0%}" if stats2["avg_fill_rate"] is not None else "N/D"
    print(f"{'Preenchimento medio':<28}{f1:<20}{f2:<20}")

    all_statuses = sorted(set(stats1["status_counts"]) | set(stats2["status_counts"]))
    for status in all_statuses:
        label = f"Inspecoes '{STATUS_LABELS.get(status, status)}'"
        print(f"{label:<28}{stats1['status_counts'].get(status, 0):<20}{stats2['status_counts'].get(status, 0):<20}")

    all_issue_types = sorted(set(stats1["issue_counts"]) | set(stats2["issue_counts"]))
    if all_issue_types:
        print("\nProblemas detetados por tipo:")
        for t in all_issue_types:
            label = report_generator.ISSUE_LABELS.get(t, t)
            print(f"{label:<28}{stats1['issue_counts'].get(t, 0):<20}{stats2['issue_counts'].get(t, 0):<20}")

    if stats1["n"] == 0 and stats2["n"] == 0:
        print("\n(Sem inspecoes para nenhuma das duas zonas no periodo indicado.)")


# --------------------------------------------------------------------------
# Comandos: report
# --------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    if not args.session and not args.zone:
        raise CLIError("indica --session <today> ou --zone <ZONE> [--period <periodo>]")

    all_records = load_all_records()

    if args.session:
        if args.session.lower() != "today":
            raise CLIError("'--session' so suporta o valor 'today' nesta versao.")
        records = filter_records_by_period(all_records, "today")
        title = "Relatorio de Sessao - Hoje"
    else:
        period = args.period or "last 7 days"
        records = filter_records_by_period(all_records, period)
        records = [r for r in records if r.get("zone_id") == args.zone]
        title = f"Relatorio - Zona {args.zone} ({period})"

    output_path = report_generator.generate_report_for_records(records, args.output, title=title)
    print(f"Relatorio gerado em: {output_path} ({len(records)} inspecao(oes) incluidas)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail Vision Intelligence System - interface CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_inspect = subparsers.add_parser("inspect", help="analisa uma imagem (ou diretorio de imagens) de prateleira")
    p_inspect.add_argument("zone", help="zone_id (ex: Z_S3) ou 'all' para inspecionar um diretorio de imagens")
    p_inspect.add_argument("--image", type=str, help="caminho da imagem (obrigatorio exceto com 'all')")
    p_inspect.add_argument("--images-dir", type=str, help="diretorio de imagens (obrigatorio com 'all')")
    p_inspect.add_argument("--strategy", type=str, default="B", choices=list(shelf_inspector.VALID_STRATEGIES))
    p_inspect.add_argument("--force", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    p_add = subparsers.add_parser("add", help="cria uma nova regra de deteccao")
    p_add.add_argument("entity", choices=["rule"])
    p_add.add_argument("description", help="descricao da regra em linguagem natural")
    p_add.set_defaults(func=cmd_add_rule)

    p_list = subparsers.add_parser("list", help="lista as regras configuradas")
    p_list.add_argument("entity", choices=["rules"])
    p_list.set_defaults(func=cmd_list_rules)

    p_delete = subparsers.add_parser("delete", help="remove uma regra")
    p_delete.add_argument("entity", choices=["rule"])
    p_delete.add_argument("rule_id")
    p_delete.set_defaults(func=cmd_delete_rule)

    p_test = subparsers.add_parser("test", help="testa uma regra contra uma imagem")
    p_test.add_argument("entity", choices=["rule"])
    p_test.add_argument("rule_id")
    p_test.add_argument("--image", type=str, required=True)
    p_test.set_defaults(func=cmd_test_rule)

    p_history = subparsers.add_parser("history", help="pergunta em linguagem natural sobre o historico de inspecoes (RAG)")
    p_history.add_argument("query", help="pergunta em linguagem natural")
    p_history.add_argument("--n", type=int, default=3)
    p_history.add_argument("--strategy", type=str, default="hybrid", choices=rag_memory.VALID_STRATEGIES)
    p_history.set_defaults(func=cmd_history)

    p_compare = subparsers.add_parser("compare", help="compara duas zonas num periodo")
    p_compare.add_argument("zone1")
    p_compare.add_argument("zone2")
    p_compare.add_argument("--period", type=str, default=None, help="ex: 'last 7 days', 'last 2 weeks', 'today'")
    p_compare.set_defaults(func=cmd_compare)

    p_report = subparsers.add_parser("report", help="gera um relatorio Markdown de sessao/zona")
    p_report.add_argument("--session", type=str, help="ex: 'today'")
    p_report.add_argument("--zone", type=str, help="zone_id (usar com --period)")
    p_report.add_argument("--period", type=str, default=None, help="ex: 'last 14 days'")
    p_report.add_argument("--output", type=str, help="caminho do ficheiro Markdown de saida")
    p_report.set_defaults(func=cmd_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return

    try:
        args.func(args)
    except CLIError as exc:
        print(f"Erro: {exc}")
    except FileNotFoundError as exc:
        print(f"Erro: ficheiro nao encontrado ({exc.filename}).")
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}")


if __name__ == "__main__":
    main()
