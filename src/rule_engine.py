from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
RULES_DIR = ROOT_DIR / "data" / "rules"
EXECUTION_LOG_PATH = RULES_DIR / "execution_log.jsonl"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

VALID_SEVERITIES = ["low", "medium", "high"]
VALID_ALERT_LEVELS = ["info", "warning", "critical"]
VALID_LOCATION_FILTERS = ["bottom", "middle", "top", "any"]

LOCATION_KEYWORDS = {
    "bottom": ["inferior", "baixo", "fundo", "base"],
    "middle": ["meio", "central", "media", "intermedi"],
    "top": ["superior", "cima", "topo", "alto"],
}


# Conversao NL -> JSON

def build_rule_prompt(nl_description: str) -> str:
    template = (PROMPTS_DIR / "rule_conversion.txt").read_text(encoding="utf-8")
    schema = (PROMPTS_DIR / "rule_schema.json").read_text(encoding="utf-8")
    return template.replace("{nl_description}", nl_description).replace("{schema}", schema)


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


def next_rule_id() -> str:
    max_n = 0
    if RULES_DIR.exists():
        for path in RULES_DIR.glob("RULE_*.json"):
            m = re.match(r"RULE_(\d+)$", path.stem)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"RULE_{max_n + 1:03d}"


def call_model(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text


def convert_rule_nl(nl_description: str, client: genai.Client) -> dict:
    """Chama o modelo para converter uma descricao NL numa regra JSON
    seguindo o schema da Seccao 5.3 do enunciado."""
    prompt = build_rule_prompt(nl_description)
    raw_text = call_model(client, prompt)
    return extract_json(raw_text)


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------

def rule_path(rule_id: str) -> Path:
    return RULES_DIR / f"{rule_id}.json"


def save_rule(rule: dict) -> Path:
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    path = rule_path(rule["rule_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
    return path


def delete_rule(rule_id: str) -> bool:
    path = rule_path(rule_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def load_rules(active_only: bool = True) -> list[dict]:
    """Carrega as regras de data/rules/. Se active_only=True, devolve apenas
    regras com validation.is_valid=True (regras sem ambiguidades por
    resolver)."""
    if not RULES_DIR.exists():
        return []
    rules = []
    for path in sorted(RULES_DIR.glob("RULE_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rule = json.load(f)
        if active_only and not rule.get("validation", {}).get("is_valid", False):
            continue
        rules.append(rule)
    return rules


def load_rule(rule_id: str) -> dict | None:
    path = rule_path(rule_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Criacao de regras
# --------------------------------------------------------------------------

def create_rule(nl_description: str, client: genai.Client | None = None) -> dict:
    """Cria uma regra a partir de uma descricao em linguagem natural,
    seguindo o schema da Seccao 5.3. A regra e sempre guardada; se
    `validation.is_valid` for False, fica registada com as ambiguidades
    detetadas em `validation.ambiguities` (a regra nao e considerada
    "ativa" por `load_rules(active_only=True)` ate ser corrigida)."""
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY nao definido em .env")
        client = genai.Client(api_key=api_key)

    result = convert_rule_nl(nl_description, client)
    result["rule_id"] = next_rule_id()
    result["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # garante que os campos de topo seguem a ordem do schema
    ordered = {
        "rule_id": result["rule_id"],
        "created_at": result["created_at"],
        "natural_language": result.get("natural_language", nl_description),
        "description": result.get("description", ""),
        "conditions": result.get("conditions", {}),
        "action": result.get("action", {}),
        "validation": result.get("validation", {"is_valid": False, "ambiguities": [], "assumptions": []}),
    }
    save_rule(ordered)
    return ordered


# --------------------------------------------------------------------------
# Execucao de regras
# --------------------------------------------------------------------------

def _severity_ge(severity: str | None, threshold: str | None) -> bool:
    if not threshold:
        return True
    if severity not in VALID_SEVERITIES or threshold not in VALID_SEVERITIES:
        return False
    return VALID_SEVERITIES.index(severity) >= VALID_SEVERITIES.index(threshold)


def _location_matches(location_filter: str | None, issue_location: str | None) -> bool:
    if not location_filter or location_filter == "any":
        return True
    if not issue_location:
        return False
    loc = issue_location.lower()
    return any(kw in loc for kw in LOCATION_KEYWORDS.get(location_filter, []))


def _record_hour(record: dict) -> int | None:
    ts = record.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").hour
    except ValueError:
        return None


def _hour_in_range(hour: int, hours_start: int, hours_end: int) -> bool:
    if hours_start <= hours_end:
        return hours_start <= hour <= hours_end
    return hour >= hours_start or hour <= hours_end  # intervalo noturno (ex: 22-6)


def _render_message(rule: dict, record: dict, matched_issues: list[dict]) -> str:
    template = rule.get("action", {}).get("notification_message", "")
    issue = matched_issues[0] if matched_issues else {}
    fill = record.get("shelf_fill_rate")
    fill_str = f"{fill:.0%}" if fill is not None else "N/D"
    try:
        return template.format(
            zone_id=record.get("zone_id", "?"),
            issue_type=issue.get("type", ""),
            severity=issue.get("severity", ""),
            shelf_fill_rate=fill_str,
        )
    except (KeyError, IndexError, ValueError):
        return template


def evaluate_rule(rule: dict, record: dict) -> dict:
    """Avalia uma regra contra um inspection record.

    Devolve {"rule_id", "description", "matched", "in_scope",
    "matched_issue_ids", "alert_level", "notification_message"}.
    """
    rule_id = rule.get("rule_id")
    cond = rule.get("conditions") or {}

    zone_filter = cond.get("zone_filter")
    if zone_filter and record.get("zone_id") not in zone_filter:
        return {
            "rule_id": rule_id, "description": rule.get("description"),
            "matched": False, "in_scope": False,
            "matched_issue_ids": [], "alert_level": None, "notification_message": None,
        }

    time_filter = cond.get("time_filter")
    if time_filter:
        hour = _record_hour(record)
        if hour is None or not _hour_in_range(hour, time_filter["hours_start"], time_filter["hours_end"]):
            return {
                "rule_id": rule_id, "description": rule.get("description"),
                "matched": False, "in_scope": True,
                "matched_issue_ids": [], "alert_level": None, "notification_message": None,
            }

    checks: list[bool] = []
    matched_issues: list[dict] = []

    issue_types = cond.get("issue_types")
    if issue_types:
        severity_threshold = cond.get("severity_threshold")
        location_filter = cond.get("location_filter")
        for issue in record.get("issues") or []:
            if issue.get("type") not in issue_types:
                continue
            if not _severity_ge(issue.get("severity"), severity_threshold):
                continue
            if not _location_matches(location_filter, issue.get("location")):
                continue
            matched_issues.append(issue)
        checks.append(len(matched_issues) > 0)

    fill_rate_threshold = cond.get("fill_rate_threshold")
    if fill_rate_threshold is not None:
        fill = record.get("shelf_fill_rate")
        checks.append(fill is not None and fill < fill_rate_threshold)

    matched = bool(checks) and all(checks)

    return {
        "rule_id": rule_id,
        "description": rule.get("description"),
        "matched": matched,
        "in_scope": True,
        "matched_issue_ids": [i.get("issue_id") for i in matched_issues],
        "alert_level": rule.get("action", {}).get("alert_level") if matched else None,
        "notification_message": _render_message(rule, record, matched_issues) if matched else None,
    }


def evaluate_rules(record: dict, rules: list[dict] | None = None) -> list[dict]:
    if rules is None:
        rules = load_rules(active_only=True)
    return [evaluate_rule(rule, record) for rule in rules]


def log_execution(record: dict, evaluations: list[dict]) -> None:
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inspection_id": record.get("inspection_id"),
        "image_path": record.get("image_path"),
        "zone_id": record.get("zone_id"),
        "evaluations": evaluations,
        "n_matched": sum(1 for e in evaluations if e["matched"]),
    }
    with open(EXECUTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_rules_on_record(record_path: str, rules: list[dict] | None = None) -> dict:
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if rules is None:
        rules = load_rules(active_only=True)
    evaluations = evaluate_rules(record, rules)
    log_execution(record, evaluations)
    return {"record": record, "evaluations": evaluations}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule Engine - regras de deteccao em linguagem natural")
    parser.add_argument("--add", type=str, help="descricao em linguagem natural de uma nova regra")
    parser.add_argument("--list", action="store_true", help="lista as regras existentes")
    parser.add_argument("--delete", type=str, metavar="RULE_ID", help="remove uma regra pelo rule_id")
    parser.add_argument("--run", action="store_true", help="executa as regras ativas sobre um inspection record")
    parser.add_argument("--inspection", type=str, help="caminho para um inspection record (.json)")
    parser.add_argument("--run-all", action="store_true", help="executa as regras sobre todos os records num diretorio")
    parser.add_argument("--inspections-dir", type=str, default=str(ROOT_DIR / "data" / "inspections"))
    args = parser.parse_args()

    if args.add:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            parser.error("GEMINI_API_KEY nao definido em .env")
        client = genai.Client(api_key=api_key)
        rule = create_rule(args.add, client=client)
        valid = rule["validation"].get("is_valid")
        print(f"\nRegra '{rule['rule_id']}' criada (validation.is_valid={valid}):")
        print(json.dumps(rule, ensure_ascii=False, indent=2))
        return

    if args.list:
        rules = load_rules(active_only=False)
        if not rules:
            print("Nenhuma regra encontrada em data/rules/.")
        for r in rules:
            valid = r.get("validation", {}).get("is_valid")
            tag = "valida" if valid else "com ambiguidades"
            print(f"- {r['rule_id']} [{tag}]: {r.get('description')}")
        return

    if args.delete:
        if delete_rule(args.delete):
            print(f"Regra '{args.delete}' removida.")
        else:
            print(f"Regra '{args.delete}' nao encontrada.")
        return

    if args.run:
        if not args.inspection:
            parser.error("--run requer --inspection <path>")
        result = run_rules_on_record(args.inspection)
        for ev in result["evaluations"]:
            status = "MATCH" if ev["matched"] else "no match"
            print(f"- {ev['rule_id']}: {status}")
            if ev["matched"]:
                print(f"    [{ev['alert_level']}] {ev['notification_message']}")
        return

    if args.run_all:
        d = Path(args.inspections_dir)
        for path in sorted(d.glob("*.json")):
            print(f"\n{path.name}:")
            result = run_rules_on_record(str(path))
            for ev in result["evaluations"]:
                if ev["matched"]:
                    print(f"  [MATCH] {ev['rule_id']} [{ev['alert_level']}]: {ev['notification_message']}")
        return

    parser.error("indica --add, --list, --delete, --run ou --run-all")


if __name__ == "__main__":
    main()
