"""
Componente 3: RAG Memory

Indexacao e recuperacao semantica do historico de inspecoes (records gerados
por shelf_inspector.py), usando ChromaDB (vectorstore persistente em
vectorstore/) e sentence-transformers multilingue para os embeddings.

Estrategia de chunking (comparadas com Recall@3):
- "document": um chunk por inspecao, com um resumo textual (estado geral,
  zona, taxa de preenchimento, issues, produtos, excerto do raciocinio do
  modelo).
- "issue": um chunk por issue detetado (ou um chunk "sem problemas" para
  inspecoes sem issues).
- "hybrid": combinacao das duas anteriores (recomendada).

Por omissao, os resumos indexados sao gerados localmente (template) a partir
dos campos estruturados do record e do "model_reasoning" ja existente, o que
e deterministico, gratuito e reprodutivel. Quando `summary_mode="llm"` e
fornecido um cliente Gemini, o campo "summary" indexado para os chunks de
tipo "document" e gerado pela LLM (Seccao 6.2 do enunciado;
prompts/rag_summary.txt) - usado com moderacao dado o limite severo de quota
da API gratuita (20 pedidos/dia/modelo, ver strategy_comparison.json).

As consultas em linguagem natural (`answer_query`) recuperam os k chunks mais
relevantes e, se houver um cliente Gemini disponivel, usam a LLM para
sintetizar uma resposta com referencia explicita a inspection_id e data
(Seccao 6.4); caso contrario, devolvem uma resposta de fallback determinista
construida a partir dos mesmos registos recuperados.

Uso:
    python src/rag_memory.py --index --strategy hybrid
    python src/rag_memory.py --query "prateleiras vazias" --n 3 --strategy hybrid
    python src/rag_memory.py --ask "Quando foi a ultima vez que a zona Z_S1 teve problemas de prateleira vazia?"
    python src/rag_memory.py --evaluate
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
INSPECTIONS_DIR = ROOT_DIR / "data" / "inspections"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
EVALUATION_DIR = ROOT_DIR / "data" / "evaluation"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

VALID_STRATEGIES = ("document", "issue", "hybrid")

# 4 perguntas obrigatórias
OBLIGATORY_QUERIES = [
    "Quando foi a ultima vez que a zona Z_S1 teve problemas de prateleira vazia?",
    "Que zonas tiveram mais issues de planograma nas ultimas 2 semanas?",
    "Existe algum padrao nos problemas detetados as sextas-feiras a tarde?",
    "Que regras foram mais frequentemente disparadas este mes?",
]


# carregamento de inspection records
def load_inspection_records(base_dir: Path | None = None) -> list[dict]:
    base_dir = base_dir or INSPECTIONS_DIR
    records: dict[str, dict] = {}
    for path in sorted(base_dir.glob("**/*.json")):
        if path.name == "execution_log.jsonl":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if record.get("_fallback") or record.get("overall_status") == "unknown":
            continue
        if "inspection_id" not in record:
            continue
        records[record["inspection_id"]] = record
    return list(records.values())


# Geracao de texto (sem chamadas ao Gemini)
def build_record_text(record: dict) -> str:
    status = record.get("overall_status", "unknown")
    zone = record.get("zone_id", "?")
    fill = record.get("shelf_fill_rate")

    parts = [f"Inspecao da zona {zone} com estado geral '{status}'."]
    if fill is not None:
        parts.append(f"Taxa de preenchimento da prateleira: {fill:.0%}.")

    issues = record.get("issues") or []
    if issues:
        issue_descs = [
            f"{iss.get('type')} (severidade {iss.get('severity')}, "
            f"{(iss.get('affected_area_pct') or 0):.0%} da area)"
            for iss in issues
        ]
        parts.append("Problemas detetados: " + "; ".join(issue_descs) + ".")
    else:
        parts.append("Nenhum problema detetado nesta inspecao.")

    products = record.get("products_detected") or []
    if products:
        parts.append("Produtos identificados: " + ", ".join(products[:8]) + ".")

    reasoning = record.get("model_reasoning") or ""
    if reasoning:
        parts.append(reasoning[:800])

    return " ".join(parts)


def build_summary_prompt(record: dict) -> str:
    template = (PROMPTS_DIR / "rag_summary.txt").read_text(encoding="utf-8")
    record_json = json.dumps({
        "inspection_id": record.get("inspection_id"),
        "timestamp": record.get("timestamp"),
        "zone_id": record.get("zone_id"),
        "overall_status": record.get("overall_status"),
        "shelf_fill_rate": record.get("shelf_fill_rate"),
        "issues": record.get("issues"),
        "products_detected": record.get("products_detected"),
        "model_reasoning": (record.get("model_reasoning") or "")[:500],
    }, ensure_ascii=False, indent=2)
    return template.replace("{record_json}", record_json)

# gerar summary
def generate_summary_llm(record: dict, client=None) -> str:
    """Gera o "summary" indexado para um record (Seccao 6.2 do enunciado).

    Se `client` for None, ou a chamada ao Gemini falhar, recorre ao resumo
    local determinista `build_record_text` (fallback documentado, dado o
    limite severo de quota da API gratuita)."""
    if client is None:
        return build_record_text(record)
    try:
        from google.genai import types
        prompt = build_summary_prompt(record)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        text = (response.text or "").strip()
        return text if text else build_record_text(record)
    except Exception:
        return build_record_text(record)


def build_issue_text(record: dict, issue: dict) -> str:
    zone = record.get("zone_id", "?")
    return (
        f"Na zona {zone}, foi detetado um problema do tipo '{issue.get('type')}' "
        f"com severidade '{issue.get('severity')}', afetando "
        f"{(issue.get('affected_area_pct') or 0):.0%} da area da prateleira. "
        f"Estado geral da inspecao: {record.get('overall_status')}."
    )


def _base_metadata(record: dict) -> dict:
    return {
        "inspection_id": record["inspection_id"],
        "image_path": record.get("image_path", ""),
        "zone_id": record.get("zone_id", ""),
        "overall_status": record.get("overall_status", ""),
        "shelf_fill_rate": (
            record.get("shelf_fill_rate")
            if record.get("shelf_fill_rate") is not None
            else -1.0
        ),
        "timestamp": record.get("timestamp", ""),
    }


def get_chunks(record: dict, strategy: str = "hybrid", summary_mode: str = "template",
                client=None) -> list[tuple[str, str, dict]]:
    """Devolve [(chunk_id, texto, metadata), ...] para um record, de acordo
    com a estrategia de chunking ("document", "issue" ou "hybrid").

    `summary_mode="llm"` (com `client` definido) gera o texto do chunk
    "document" via LLM (Seccao 6.2); por omissao ("template") usa
    `build_record_text` (resumo local determinista)."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Estrategia de chunking invalida: {strategy}")

    inspection_id = record["inspection_id"]
    base_meta = _base_metadata(record)
    chunks: list[tuple[str, str, dict]] = []

    if strategy in ("document", "hybrid"):
        meta = dict(base_meta, chunk_type="document")
        if summary_mode == "llm":
            text = generate_summary_llm(record, client=client)
        else:
            text = build_record_text(record)
        chunks.append((f"{inspection_id}_doc", text, meta))

    if strategy in ("issue", "hybrid"):
        issues = record.get("issues") or []
        if issues:
            for i, issue in enumerate(issues):
                meta = dict(
                    base_meta,
                    chunk_type="issue",
                    issue_type=issue.get("type", ""),
                    severity=issue.get("severity", ""),
                )
                chunks.append((f"{inspection_id}_issue_{i}", build_issue_text(record, issue), meta))
        elif strategy == "issue":
            meta = dict(base_meta, chunk_type="issue", issue_type="none", severity="none")
            text = (
                f"Inspecao na zona {record.get('zone_id')} sem problemas detetados, "
                f"estado geral '{record.get('overall_status')}'."
            )
            chunks.append((f"{inspection_id}_issue_none", text, meta))

    return chunks


# ChromaDB
_embedding_function = None

def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_function


def get_client() -> chromadb.ClientAPI:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


def collection_name_for(strategy: str) -> str:
    return f"inspections_{strategy}"


def index_records(strategy: str = "hybrid", records: list[dict] | None = None,
                   reset: bool = True, summary_mode: str = "template",
                   llm_client=None) -> tuple["chromadb.Collection", int, int]:
    """Indexa inspection records em ChromaDB usando a estrategia de chunking
    indicada. Devolve (collection, n_records, n_chunks).
    """
    client = get_client()
    name = collection_name_for(strategy)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name, embedding_function=get_embedding_function())

    if records is None:
        records = load_inspection_records()

    ids, docs, metas = [], [], []
    for record in records:
        for chunk_id, text, meta in get_chunks(record, strategy, summary_mode=summary_mode, client=llm_client):
            ids.append(chunk_id)
            docs.append(text)
            metas.append(meta)

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    return collection, len(records), len(ids)


def query_memory(query_text: str, n_results: int = 3, strategy: str = "hybrid") -> list[dict]:
    """Pesquisa semantica em linguagem natural sobre o historico de
    inspecoes. Devolve ate `n_results` records distintos (deduplicados por
    inspection_id), ordenados por relevancia."""
    client = get_client()
    collection = client.get_or_create_collection(
        collection_name_for(strategy), embedding_function=get_embedding_function()
    )
    if collection.count() == 0:
        return []

    fetch_n = min(collection.count(), n_results * 3)
    res = collection.query(query_texts=[query_text], n_results=fetch_n)

    seen: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        iid = meta["inspection_id"]
        if iid not in seen or dist < seen[iid]["distance"]:
            seen[iid] = {
                "inspection_id": iid,
                "image_path": meta.get("image_path"),
                "zone_id": meta.get("zone_id"),
                "overall_status": meta.get("overall_status"),
                "chunk_type": meta.get("chunk_type"),
                "timestamp": meta.get("timestamp", ""),
                "text": doc,
                "distance": dist,
            }

    return sorted(seen.values(), key=lambda r: r["distance"])[:n_results]


# RAG
def build_answer_prompt(query_text: str, retrieved: list[dict]) -> str:
    template = (PROMPTS_DIR / "rag_answer.txt").read_text(encoding="utf-8")
    context_lines = []
    for r in retrieved:
        date = (r.get("timestamp") or "")[:10] or "data desconhecida"
        context_lines.append(
            f"- inspection_id={r['inspection_id']}, data={date}, zona={r['zone_id']}, "
            f"estado={r['overall_status']}: {r['text']}"
        )
    return template.replace("{query}", query_text).replace("{context}", "\n".join(context_lines))


def _fallback_answer(query_text: str, retrieved: list[dict]) -> str:
    """Resposta determinista (sem LLM), citando inspection_id e data dos
    registos recuperados - usada quando nao ha cliente Gemini disponivel ou
    a chamada falha."""
    lines = [
        f"Foram encontrados {len(retrieved)} registo(s) relevante(s) no historico de "
        f"inspecoes para a pergunta \"{query_text}\":"
    ]
    for r in retrieved:
        date = (r.get("timestamp") or "")[:10] or "data desconhecida"
        lines.append(
            f"- Inspecao {r['inspection_id']} ({date}, zona {r['zone_id']}, "
            f"estado {r['overall_status']}): {r['text'][:220]}"
        )
    return "\n".join(lines)


def answer_query(query_text: str, k: int = 3, strategy: str = "hybrid", client=None) -> dict:
    """Recupera os k chunks mais relevantes e sintetiza uma resposta com
    referencia explicita a inspection_id e data (Seccao 6.4).

    Devolve {"query", "answer", "sources": [{"inspection_id","date","zone_id"}]}.
    Se `client` for None ou a chamada ao Gemini falhar, usa
    `_fallback_answer` (resposta determinista a partir dos mesmos registos
    recuperados)."""
    retrieved = query_memory(query_text, n_results=k, strategy=strategy)

    if not retrieved:
        return {
            "query": query_text,
            "answer": "Nao foram encontrados registos relevantes no historico de inspecoes para esta pergunta.",
            "sources": [],
        }

    answer = None
    if client is not None:
        try:
            from google.genai import types
            prompt = build_answer_prompt(query_text, retrieved)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
            answer = (response.text or "").strip()
        except Exception:
            answer = None

    if not answer:
        answer = _fallback_answer(query_text, retrieved)

    return {
        "query": query_text,
        "answer": answer,
        "sources": [
            {
                "inspection_id": r["inspection_id"],
                "date": (r.get("timestamp") or "")[:10],
                "zone_id": r["zone_id"],
            }
            for r in retrieved
        ],
    }


# Avaliacao Recall@3 das estrategias de chunking

EVAL_QUERIES = [
    {
        "query": "prateleiras vazias ou com falta de stock",
        "match": lambda r: any(i.get("type") == "empty_shelf" for i in (r.get("issues") or [])),
    },
    {
        "query": "produtos trocados ou fora de posicao na prateleira",
        "match": lambda r: any(i.get("type") == "wrong_product" for i in (r.get("issues") or [])),
    },
    {
        "query": "produtos desalinhados ou mal arrumados",
        "match": lambda r: any(i.get("type") == "misaligned" for i in (r.get("issues") or [])),
    },
    {
        "query": "inspecoes em estado critico que precisam de atencao imediata",
        "match": lambda r: r.get("overall_status") == "critical",
    },
    {
        "query": "prateleiras bem organizadas e completamente preenchidas sem problemas",
        "match": lambda r: r.get("overall_status") == "ok",
    },
]


def evaluate_strategies(k: int = 3, strategies: tuple[str, ...] = VALID_STRATEGIES) -> dict:
    records = load_inspection_records()
    results = {}

    for strategy in strategies:
        index_records(strategy=strategy, records=records, reset=True)

        per_query = []
        for q in EVAL_QUERIES:
            relevant = {r["inspection_id"] for r in records if q["match"](r)}
            if not relevant:
                continue
            retrieved = query_memory(q["query"], n_results=k, strategy=strategy)
            retrieved_ids = {r["inspection_id"] for r in retrieved}
            hits = len(retrieved_ids & relevant)
            recall = hits / min(k, len(relevant))
            per_query.append({
                "query": q["query"],
                "n_relevant": len(relevant),
                "hits": hits,
                "recall_at_k": recall,
            })

        avg_recall = sum(p["recall_at_k"] for p in per_query) / len(per_query) if per_query else None
        results[strategy] = {"avg_recall_at_k": avg_recall, "per_query": per_query}

    results["_meta"] = {"k": k, "n_records": len(records)}
    return results


# CLI

def main():
    parser = argparse.ArgumentParser(description="RAG Memory - indexacao e pesquisa semantica de inspecoes")
    parser.add_argument("--index", action="store_true", help="(re)indexa o historico de inspecoes em ChromaDB")
    parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="hybrid",
                         help="estrategia de chunking (default: hybrid)")
    parser.add_argument("--query", type=str, help="consulta em linguagem natural sobre o historico de inspecoes (apenas recuperacao)")
    parser.add_argument("--ask", type=str, help="pergunta em linguagem natural; recupera e sintetiza uma resposta com a LLM (Seccao 6.4)")
    parser.add_argument("--n", type=int, default=3, help="numero de resultados a devolver (default: 3)")
    parser.add_argument("--evaluate", action="store_true",
                         help="compara estrategias de chunking via Recall@3 e guarda em data/evaluation/rag_recall.json")
    args = parser.parse_args()

    if args.index:
        collection, n_records, n_chunks = index_records(strategy=args.strategy)
        print(f"Indexados {n_records} inspection records em {n_chunks} chunks "
              f"(estrategia='{args.strategy}', colecao='{collection.name}').")
        return

    if args.query:
        results = query_memory(args.query, n_results=args.n, strategy=args.strategy)
        if not results:
            print("Sem resultados (a colecao esta vazia? corre --index primeiro).")
            return
        print(f"Top {len(results)} resultados para: \"{args.query}\" (estrategia='{args.strategy}')\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['inspection_id']} | {r['image_path']} | zona={r['zone_id']} "
                  f"| estado={r['overall_status']} | distancia={r['distance']:.4f}")
            print(f"   {r['text'][:200]}...")
        return

    if args.ask:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = None
        if api_key:
            from google import genai
            client = genai.Client(api_key=api_key)
        result = answer_query(args.ask, k=args.n, strategy=args.strategy, client=client)
        print(f"Pergunta: {result['query']}\n")
        print(f"Resposta: {result['answer']}\n")
        if result["sources"]:
            print("Fontes:")
            for s in result["sources"]:
                print(f"  - {s['inspection_id']} ({s['date']}, zona {s['zone_id']})")
        return

    if args.evaluate:
        results = evaluate_strategies(k=3)
        print(f"\nRecall@3 (n_records={results['_meta']['n_records']})\n")
        print(f"{'Estrategia':<12} {'Recall@3 medio':<16}")
        for strategy in VALID_STRATEGIES:
            avg = results[strategy]["avg_recall_at_k"]
            print(f"{strategy:<12} {avg:.2f}" if avg is not None else f"{strategy:<12} N/A")
        for strategy in VALID_STRATEGIES:
            print(f"\n--- {strategy} ---")
            for p in results[strategy]["per_query"]:
                print(f"  '{p['query']}': {p['hits']}/{min(3, p['n_relevant'])} "
                      f"(n_relevant={p['n_relevant']}, recall@3={p['recall_at_k']:.2f})")

        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVALUATION_DIR / "rag_recall.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados em {out_path.relative_to(ROOT_DIR)}")
        return

    parser.error("indica --index, --query ou --evaluate")


if __name__ == "__main__":
    main()
