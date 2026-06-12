from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
CACHE_DIR = ROOT_DIR / "cache"
INSPECTIONS_DIR = ROOT_DIR / "data" / "inspections"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
RPD_LIMIT = int(os.getenv("GEMINI_RPD_LIMIT", "1500"))

VALID_STRATEGIES = {"A": "strategy_a_zero_shot.txt", "B": "strategy_b_cot.txt", "C": "strategy_c_fewshot.txt"}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

REQUIRED_FIELDS = [
    "inspection_id", "timestamp", "image_path", "zone_id", "overall_status",
    "issues", "shelf_fill_rate", "products_detected", "model_reasoning",
]

# Utilidades de cache

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(image_hash: str, strategy: str, zone_id: str) -> str:
    return f"{image_hash}_{strategy}_{zone_id}"


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def load_from_cache(key: str) -> dict | None:
    path = cache_path(key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(key: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(key), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# rate limiting

class QuotaManager:
    """Controla limites de requests por minuto (RPM) e por dia (RPD)."""

    def __init__(self):
        self.state_path = CACHE_DIR / "quota_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get("date") != today:
            state = {"date": today, "count": 0}

        state.setdefault("recent_request_times", [])
        return state

    def _save_state(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f)

    def daily_remaining(self) -> int:
        return max(0, RPD_LIMIT - self.state["count"])

    def daily_exhausted(self) -> bool:
        return self.daily_remaining() <= 0

    def wait_for_rpm_slot(self) -> None:
        now = time.time()
        recent = [t for t in self.state["recent_request_times"] if now - t < 60]
        if len(recent) >= RPM_LIMIT:
            sleep_time = 60 - (now - recent[0]) + 0.5
            if sleep_time > 0:
                print(f"  [rate limit] a aguardar {sleep_time:.1f}s para respeitar {RPM_LIMIT} req/min...")
                time.sleep(sleep_time)
        self.state["recent_request_times"] = [
            t for t in self.state["recent_request_times"] if time.time() - t < 60
        ]

    def record_request(self) -> None:
        self.state["count"] += 1
        self.state["recent_request_times"].append(time.time())
        self._save_state()


# prompts

def build_prompt(strategy: str) -> str:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Estrategia invalida: {strategy}. Use uma de {list(VALID_STRATEGIES)}")

    prompt_file = PROMPTS_DIR / VALID_STRATEGIES[strategy]
    schema_file = PROMPTS_DIR / "schema.json"

    prompt_template = prompt_file.read_text(encoding="utf-8")
    schema_text = schema_file.read_text(encoding="utf-8")

    return prompt_template.replace("{schema}", schema_text)


# Chamada ao modelo + parsing

def extract_json(text: str) -> dict:
    text = text.strip()

    # Remove blocos de codigo markdown
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Tenta encontrar o primeiro '{' e o ultimo '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


def call_gemini_with_retry(client: genai.Client, prompt: str, image_bytes: bytes,
                            mime_type: str, max_retries: int = 5) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text
        except APIError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "429" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  [429] erro de rate limit, backoff {delay:.1f}s (tentativa {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("Numero maximo de tentativas excedido.")


# Inspecao principal
_inspection_counter = 0

def next_inspection_id() -> str:
    global _inspection_counter
    _inspection_counter += 1
    now = datetime.now(timezone.utc)
    return f"INS_{now.strftime('%Y%m%d_%H%M%S')}_{_inspection_counter:03d}"


def fallback_record(image_path: str, zone_id: str, reason: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "inspection_id": next_inspection_id(),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_path": image_path,
        "zone_id": zone_id,
        "overall_status": "unknown",
        "issues": [],
        "shelf_fill_rate": None,
        "products_detected": [],
        "model_reasoning": f"FALLBACK: nao foi possivel analisar esta imagem. Motivo: {reason}",
        "_fallback": True,
        "_error": reason,
    }


def analyze_image(image_path: str, zone_id: str = "Z_UNKNOWN", strategy: str = "B",
                   force: bool = False, client: genai.Client | None = None,
                   quota: QuotaManager | None = None) -> dict:
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    image_hash = md5_of_file(image_path_obj)
    key = cache_key(image_hash, strategy, zone_id)

    if not force:
        cached = load_from_cache(key)
        if cached is not None:
            cached = dict(cached)
            cached["_from_cache"] = True
            return cached

    if quota is None:
        quota = QuotaManager()

    if quota.daily_exhausted():
        return fallback_record(
            str(image_path), zone_id,
            f"Quota diaria de {RPD_LIMIT} req/dia esgotada. "
            f"Apenas resultados em cache estao disponiveis para esta imagem.",
        )

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return fallback_record(str(image_path), zone_id, "GEMINI_API_KEY nao definido em .env")
        client = genai.Client(api_key=api_key)

    prompt = build_prompt(strategy)

    suffix = image_path_obj.suffix.lower()
    mime_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix.lstrip("."), "image/jpeg"
    )
    image_bytes = image_path_obj.read_bytes()

    quota.wait_for_rpm_slot()

    try:
        raw_text = call_gemini_with_retry(client, prompt, image_bytes, mime_type)
        quota.record_request()
    except Exception as e:  # noqa: BLE001 - queremos fallback gracioso para qualquer erro de API
        return fallback_record(str(image_path), zone_id, f"Erro na chamada a API: {e}")

    try:
        record = extract_json(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        record = fallback_record(str(image_path), zone_id, f"Resposta do modelo nao e JSON valido: {e}")
        record["_raw_response"] = raw_text
        save_to_cache(key, record)
        return record

    # inspection_id e timestamp sao gerados pelo sistema
    record["inspection_id"] = next_inspection_id()
    record["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record["image_path"] = str(image_path)
    record["zone_id"] = zone_id
    record["_strategy"] = strategy

    for field in REQUIRED_FIELDS:
        if field not in record:
            record[field] = [] if field in ("issues", "products_detected") else None

    save_to_cache(key, record)
    return record


# CLI
def main():
    parser = argparse.ArgumentParser(description="Shelf Inspector - analise visual de prateleiras com Gemini")
    parser.add_argument("--image", type=str, help="caminho para uma imagem")
    parser.add_argument("--images-dir", type=str, help="diretorio com varias imagens")
    parser.add_argument("--zone", type=str, default="Z_UNKNOWN", help="identificador da zona")
    parser.add_argument("--strategy", type=str, default="B", choices=list(VALID_STRATEGIES), help="estrategia de prompting (A/B/C)")
    parser.add_argument("--force", action="store_true", help="ignora cache e reanalisa")
    parser.add_argument("--out-dir", type=str, default=str(INSPECTIONS_DIR), help="diretorio onde guardar inspection records")
    args = parser.parse_args()

    if not args.image and not args.images_dir:
        parser.error("e necessario indicar --image ou --images-dir")

    if args.image:
        image_paths = [args.image]
    else:
        d = Path(args.images_dir)
        image_paths = sorted(
            str(p) for p in d.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quota = QuotaManager()
    client = None
    if not quota.daily_exhausted():
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)

    for i, img_path in enumerate(image_paths):
        print(f"[{i + 1}/{len(image_paths)}] {img_path}")
        record = analyze_image(img_path, zone_id=args.zone, strategy=args.strategy,
                                force=args.force, client=client, quota=quota)

        status = record.get("overall_status", "?")
        n_issues = len(record.get("issues", []))
        source = "cache" if record.get("_from_cache") else "API"
        print(f"  - status={status}, issues={n_issues}, fonte={source}")

        out_path = out_dir / f"{record['inspection_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\nQuota diaria restante: {quota.daily_remaining()}/{RPD_LIMIT}")


if __name__ == "__main__":
    main()
