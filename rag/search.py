"""Поиск по готовому rag/index.json: один вектор вопроса и топ кусков.

Считает косинус на обычной арифметике, без torch и numpy. Модель берёт из индекса,
чтобы вопрос векторизовался той же моделью, что и конспекты.
Агент вызывает: python rag/search.py "текст вопроса"
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import gemini_http

ROOT = gemini_http.ROOT
INDEX_PATH = ROOT / "rag" / "index.json"
TOP_K = 8


def _stdout_utf8() -> None:
    """Печать кусков на русском в консоли Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def cosine(left: list[float], right: list[float]) -> float:
    """Косинусная близость двух векторов одной длины. Ноль, если вектор пустой."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


def load_index() -> dict:
    """Читает индекс. Без файла просит сначала собрать его через index_build.py."""
    if not INDEX_PATH.is_file():
        raise SystemExit("Нет rag/index.json. Сначала: python rag/index_build.py")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def search(question: str) -> None:
    """Векторизует вопрос одним запросом и печатает топ кусков с файлом и заголовком."""
    index = load_index()
    model = str(index.get("model") or "")
    chunks = index.get("chunks") or []
    if not model or not chunks:
        raise SystemExit("Индекс пустой. Запусти python rag/index_build.py")
    api_key = gemini_http.load_api_key()
    task_type = None if gemini_http.is_embedding_v2(model) else "RETRIEVAL_QUERY"
    vector = gemini_http.embed_one(
        api_key,
        model,
        gemini_http.prepare_query(model, question),
        task_type=task_type,
    )
    ranked = []
    for item in chunks:
        stored = item.get("vector") or []
        ranked.append((cosine(vector, stored), item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    top = ranked[:TOP_K]
    print(f"model: {model}")
    print(f"hits: {len(top)}")
    for rank, (score, item) in enumerate(top, start=1):
        print()
        print(f"--- {rank} | score {score:.4f}")
        print(f"file: {item.get('file', '')}")
        print(f"heading: {item.get('heading', '')}")
        print("text:")
        print(item.get("text", "").strip())


def main() -> None:
    """Вопрос — все аргументы командной строки одной строкой."""
    _stdout_utf8()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Нужен вопрос: python rag/search.py "текст вопроса"')
    search(question)


if __name__ == "__main__":
    main()
