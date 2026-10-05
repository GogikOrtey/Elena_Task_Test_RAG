"""Пробник embedding-моделей Gemini и сборка rag/index.json.

Сначала проверяет, какая модель отвечает на ключе из .env, затем векторизует
куски из chunk.py пачками. Повторный запуск не пересчитывает уже сохранённые id.
Запускается вручную: python rag/index_build.py --probe и python rag/index_build.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chunk
import gemini_http

ROOT = gemini_http.ROOT
INDEX_PATH = ROOT / "rag" / "index.json"
PROBE_TEXT = "понятный язык в ответах на обращения граждан"
CANDIDATES = (
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
)
BATCH_SIZE = 8
PAUSE_SEC = 1.0


def _stdout_utf8() -> None:
    """Чтобы русский текст пробника не ломался в консоли Windows и не копился в буфере."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def probe(api_key: str) -> str:
    """Печатает доступные embed-модели и выбирает первую, которая вернула вектор.

    Предпочитает gemini-embedding-001: она стабильная и умеет пачку строк.
    Если все кандидаты дали 403 или 429, останавливает сборку.
    """
    listed = gemini_http.list_embed_models(api_key)
    print("models:", ", ".join(listed) if listed else "(пусто)")
    blocked: list[str] = []
    chosen = ""
    for model in CANDIDATES:
        task_type = None if gemini_http.is_embedding_v2(model) else "RETRIEVAL_QUERY"
        text = gemini_http.prepare_query(model, PROBE_TEXT)
        try:
            vector = gemini_http.embed_one(
                api_key,
                model,
                text,
                task_type=task_type,
            )
        except gemini_http.ApiError as exc:
            print(f"probe {model}: {exc.status}")
            if exc.status in (403, 429):
                blocked.append(f"{model} HTTP {exc.status}: {exc.body}")
            continue
        print(f"probe {model}: 200, dim {len(vector)}")
        if not chosen and vector:
            chosen = model
    if chosen:
        print(f"chosen: {chosen}")
        return chosen
    if blocked and len(blocked) == len(CANDIDATES):
        raise SystemExit("Индексация остановлена. Ни одна модель не ответила:\n" + "\n".join(blocked))
    if blocked:
        raise SystemExit("Индексация остановлена:\n" + "\n".join(blocked))
    raise SystemExit("Ни одна embedding-модель не вернула вектор")


def _load_index() -> dict | None:
    """Читает уже собранный индекс, если он есть и это объект JSON."""
    if not INDEX_PATH.is_file():
        return None
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def _save_index(payload: dict) -> None:
    """Пишет индекс целиком. Вызывается после каждой пачки, чтобы обрыв не стирал прогресс."""
    INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def build(api_key: str, model: str) -> None:
    """Векторизует куски без вектора и дописывает их в rag/index.json."""
    chunks = chunk.load_chunks()
    previous = _load_index()
    saved: dict[str, list[float]] = {}
    if previous and previous.get("model") == model:
        for item in previous.get("chunks", []):
            vector = item.get("vector")
            if item.get("id") and vector:
                saved[item["id"]] = vector
    pending = [item for item in chunks if item["id"] not in saved]
    print(f"chunks: {len(chunks)}, pending: {len(pending)}, model: {model}")
    payload = {
        "model": model,
        "dim": gemini_http.OUTPUT_DIM,
        "chunks": [],
    }
    for item in chunks:
        if item["id"] in saved:
            payload["chunks"].append({**item, "vector": saved[item["id"]]})
    task_type = None if gemini_http.is_embedding_v2(model) else "RETRIEVAL_DOCUMENT"
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        texts = [
            gemini_http.prepare_document(model, item["heading"], item["text"])
            for item in batch
        ]
        titles = [item["heading"] for item in batch]
        vectors = gemini_http.embed_batch(
            api_key,
            model,
            texts,
            task_type=task_type,
            titles=titles,
        )
        if len(vectors) != len(batch):
            raise SystemExit(
                f"Модель вернула {len(vectors)} векторов на {len(batch)} кусков"
            )
        by_id = {item["id"]: vector for item, vector in zip(batch, vectors)}
        for item in batch:
            payload["chunks"].append({**item, "vector": by_id[item["id"]]})
        _save_index(payload)
        done = min(start + BATCH_SIZE, len(pending))
        print(f"embedded {done}/{len(pending)}")
        if done < len(pending):
            time.sleep(PAUSE_SEC)
    order = {item["id"]: i for i, item in enumerate(chunks)}
    payload["chunks"].sort(key=lambda item: order.get(item["id"], 10**9))
    _save_index(payload)
    print(f"index: {INDEX_PATH}")


def main() -> None:
    """--probe только проверяет модели. Без флага после выбора собирает индекс."""
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="Сборка RAG-индекса конспектов")
    parser.add_argument("--probe", action="store_true", help="Только проверить модели")
    args = parser.parse_args()
    api_key = gemini_http.load_api_key()
    model = probe(api_key)
    if args.probe:
        return
    build(api_key, model)


if __name__ == "__main__":
    main()
