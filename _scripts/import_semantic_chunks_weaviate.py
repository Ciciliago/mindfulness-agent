import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import weaviate


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.constants import WEAVIATE_DOCS_INDEX_NAME

DATA_ROOT = REPO_ROOT / "data" / "structured data and embeddings  with meta data"
DEFAULT_CONFIG = REPO_ROOT.parent / "config.txt"

SEMANTIC_FILES = [
    DATA_ROOT / "chunks" / "chunks_rag_theory_semantic_enriched.jsonl",
    DATA_ROOT / "chunks" / "chunks_skill_script_semantic_enriched.jsonl",
]
EMBEDDING_FILES = [
    DATA_ROOT
    / "embeddings"
    / "chunks_rag_theory_semantic_enriched.sentence-transformers__all-MiniLM-L6-v2.embeddings.npy",
    DATA_ROOT
    / "embeddings"
    / "chunks_skill_script_semantic_enriched.sentence-transformers__all-MiniLM-L6-v2.embeddings.npy",
]


def parse_config_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    pattern = re.compile(r'^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$')
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            env[match.group(1)] = match.group(2)
    return env


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_class(client: weaviate.Client, class_name: str) -> None:
    if client.schema.exists(class_name):
        return

    schema = {
        "class": class_name,
        "description": "Mindfulness semantic chunks imported from local structured data.",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "doc_id", "dataType": ["text"]},
            {"name": "track", "dataType": ["text"]},
            {"name": "splitter", "dataType": ["text"]},
            {"name": "topic", "dataType": ["text"]},
            {"name": "scenario", "dataType": ["text"]},
            {"name": "chunk_title_zh", "dataType": ["text"]},
            {"name": "chunk_summary_zh", "dataType": ["text"]},
            {"name": "source_url", "dataType": ["text"]},
        ],
    }
    client.schema.create_class(schema)


def get_count(client: weaviate.Client, class_name: str) -> int:
    result = client.query.aggregate(class_name).with_meta_count().do()
    try:
        return int(result["data"]["Aggregate"][class_name][0]["meta"]["count"])
    except Exception:
        return -1


def build_object(row: Dict) -> Dict:
    source = row.get("source_url") or row.get("file_path") or row.get("doc_id") or ""
    title = row.get("chunk_title_zh") or row.get("title") or row.get("doc_id") or ""
    return {
        "text": row.get("text", ""),
        "source": source,
        "title": title,
        "chunk_id": row.get("chunk_id", ""),
        "doc_id": row.get("doc_id", ""),
        "track": row.get("track", ""),
        "splitter": row.get("splitter", ""),
        "topic": row.get("topic", ""),
        "scenario": row.get("scenario", ""),
        "chunk_title_zh": row.get("chunk_title_zh", ""),
        "chunk_summary_zh": row.get("chunk_summary_zh", ""),
        "source_url": row.get("source_url", ""),
    }


def stable_uuid(class_name: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{class_name}:{chunk_id}"))


def batches(rows: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def load_semantic_rows(paths: List[Path]) -> Tuple[List[Dict], Dict[str, int], List[str]]:
    all_rows: List[Dict] = []
    counts: Dict[str, int] = {}
    source_names: List[str] = []
    for path in paths:
        rows = read_jsonl(path)
        semantic_rows = [r for r in rows if r.get("splitter") == "semantic"]
        all_rows.extend(semantic_rows)
        counts[path.name] = len(semantic_rows)
        source_names.extend([path.stem] * len(semantic_rows))
    return all_rows, counts, source_names


def load_semantic_vectors(paths: List[Path]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for path in paths:
        arr = np.load(path)
        if len(arr.shape) != 2:
            raise ValueError(f"invalid embedding shape in {path}: {arr.shape}")
        vectors.extend(arr.tolist())
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import semantic chunks into Weaviate (semantic only, no fixed chunks)."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config.txt that contains $env:WEAVIATE_URL / WEAVIATE_API_KEY / API_SECRET_KEY",
    )
    parser.add_argument(
        "--class-name",
        default=WEAVIATE_DOCS_INDEX_NAME,
        help="Weaviate class name to import into.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    env = parse_config_env(config_path)
    required = ["WEAVIATE_URL", "WEAVIATE_API_KEY"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise ValueError(f"missing required config keys: {missing}")

    for key, value in env.items():
        os.environ[key] = value

    files = SEMANTIC_FILES
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"semantic chunk file not found: {f}")

    rows, per_file_counts, _ = load_semantic_rows(files)
    if not rows:
        raise ValueError("no semantic rows found")
    for emb in EMBEDDING_FILES:
        if not emb.exists():
            raise FileNotFoundError(f"embedding file not found: {emb}")
    vectors = load_semantic_vectors(EMBEDDING_FILES)
    if len(vectors) != len(rows):
        raise ValueError(
            f"row/vector count mismatch: rows={len(rows)} vectors={len(vectors)}"
        )

    print("Semantic input files:")
    for file_name, count in per_file_counts.items():
        print(f"  - {file_name}: {count}")
    print(f"Total semantic rows: {len(rows)}")

    if args.dry_run:
        print("Dry-run only, no data written.")
        return

    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        startup_period=5,
    )

    ensure_class(client, args.class_name)
    before_count = get_count(client, args.class_name)
    print(f"Weaviate class: {args.class_name}")
    print(f"Count before import: {before_count}")

    total = 0

    client.batch.configure(batch_size=args.batch_size, dynamic=False)

    for offset, batch_rows in enumerate(batches(rows, args.batch_size)):
        batch_vectors = vectors[offset * args.batch_size : offset * args.batch_size + len(batch_rows)]
        with client.batch as batch:
            for row, vector in zip(batch_rows, batch_vectors):
                chunk_id = row.get("chunk_id") or f"{row.get('doc_id','unknown')}_{total}"
                obj = build_object(row)
                batch.add_data_object(
                    data_object=obj,
                    class_name=args.class_name,
                    uuid=stable_uuid(args.class_name, chunk_id),
                    vector=vector,
                )
                total += 1

    after_count = get_count(client, args.class_name)
    print(f"Imported/updated semantic chunks: {total}")
    print(f"Count after import: {after_count}")


if __name__ == "__main__":
    main()
