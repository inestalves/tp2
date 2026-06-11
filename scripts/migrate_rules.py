import json
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent.parent / "data" / "rules"

new_rules = [
    {
        "rule_id": "RULE_001",
        "created_at": "2026-06-10T20:09:01Z",
        "natural_language": "Alertar sempre que o shelf_fill_rate de uma prateleira for inferior a 0.7",
        "description": "O sistema deve emitir um alerta sempre que a taxa de preenchimento de prateleira (shelf_fill_rate) de qualquer zona for inferior a 0.7 (70%).",
        "conditions": {
            "zone_filter": None,
            "time_filter": None,
            "issue_types": None,
            "severity_threshold": None,
            "fill_rate_threshold": 0.7,
            "location_filter": None,
        },
        "action": {
            "alert_level": "warning",
            "notification_message": "Alerta: a zona {zone_id} tem uma taxa de preenchimento de {shelf_fill_rate}, abaixo do limiar de 70%."
        },
        "validation": {
            "is_valid": True,
            "ambiguities": [],
            "assumptions": ["Aplica-se a todas as zonas, dado que a descricao nao especifica nenhuma zona em particular."]
        }
    },
    {
        "rule_id": "RULE_002",
        "created_at": "2026-06-10T20:10:35Z",
        "natural_language": "Alertar quando houver muitos produtos fora de posicao numa prateleira",
        "description": "O sistema deve emitir um alerta quando forem detetados produtos desalinhados (misaligned) numa prateleira, em quantidade considerada elevada.",
        "conditions": {
            "zone_filter": None,
            "time_filter": None,
            "issue_types": ["misaligned"],
            "severity_threshold": "medium",
            "fill_rate_threshold": None,
            "location_filter": None,
        },
        "action": {
            "alert_level": "warning",
            "notification_message": "Alerta: foram detetados produtos desalinhados na zona {zone_id} (severidade {severity})."
        },
        "validation": {
            "is_valid": False,
            "ambiguities": [
                "A descricao nao especifica o que significa 'muitos' produtos fora de posicao (ex: percentagem de area afetada ou numero de issues), pelo que nao e possivel definir um limiar fiavel."
            ],
            "assumptions": [
                "Assumiu-se severidade minima 'medium' como proxy provisorio para 'muitos', mas o schema de condicoes nao permite expressar um limiar de area afetada."
            ]
        }
    },
    {
        "rule_id": "RULE_003",
        "created_at": "2026-06-10T20:15:20Z",
        "natural_language": "Alertar quando houver muitos produtos fora de posicao numa prateleira (mais de 20% da area, severidade media ou superior)",
        "description": "O sistema deve emitir um alerta quando for detetado pelo menos um issue do tipo 'produto fora de posicao' (wrong_product) com severidade media ou superior em qualquer zona.",
        "conditions": {
            "zone_filter": None,
            "time_filter": None,
            "issue_types": ["wrong_product"],
            "severity_threshold": "medium",
            "fill_rate_threshold": None,
            "location_filter": None,
        },
        "action": {
            "alert_level": "warning",
            "notification_message": "Alerta: produtos fora de posicao detetados na zona {zone_id} com severidade {severity}."
        },
        "validation": {
            "is_valid": True,
            "ambiguities": [],
            "assumptions": [
                "O limiar original de 'mais de 20% da area afetada' nao e representavel no schema de condicoes definido (que nao inclui affected_area_pct); usa-se apenas o filtro de severidade >= 'medium' como aproximacao."
            ]
        }
    },
    {
        "rule_id": "RULE_004",
        "created_at": "2026-06-10T20:16:20Z",
        "natural_language": "Marcar como critico qualquer prateleira que tenha um problema do tipo prateleira vazia (empty_shelf) com severidade alta",
        "description": "O sistema deve emitir um alerta critico sempre que for detetado um issue do tipo 'prateleira vazia' (empty_shelf) com severidade alta em qualquer zona.",
        "conditions": {
            "zone_filter": None,
            "time_filter": None,
            "issue_types": ["empty_shelf"],
            "severity_threshold": "high",
            "fill_rate_threshold": None,
            "location_filter": None,
        },
        "action": {
            "alert_level": "critical",
            "notification_message": "Alerta CRITICO: prateleira vazia com severidade alta detetada na zona {zone_id}."
        },
        "validation": {
            "is_valid": True,
            "ambiguities": [],
            "assumptions": []
        }
    },
]

for p in RULES_DIR.glob("RULE_2*.json"):
    p.unlink()

for r in new_rules:
    path = RULES_DIR / f"{r['rule_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print("wrote", path)

log = RULES_DIR / "execution_log.jsonl"
if log.exists():
    log.unlink()
    print("removed old execution_log.jsonl")
