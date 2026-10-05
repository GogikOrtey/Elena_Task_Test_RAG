"""HTTP-клиент эмбеддингов Gemini на стандартной библиотеке.

Ходит в generativelanguage.googleapis.com: список моделей, один вектор
и пачка batchEmbedContents. Ключ берёт из .env в корне проекта (GEMINI_API_KEY).
Используют index_build.py (сборка индекса) и search.py (вектор вопроса).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://generativelanguage.googleapis.com/v1beta"
OUTPUT_DIM = 768


class ApiError(Exception):
    """Ответ Gemini с кодом, при котором повтор не поможет, либо лимит попыток исчерпан."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body[:500]
        super().__init__(f"HTTP {status}: {self.body}")


def load_api_key() -> str:
    """Читает GEMINI_API_KEY из корневого .env. Пустое значение считает ошибкой запуска."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        raise SystemExit("Нет файла .env в корне проекта")
    key = ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "GEMINI_API_KEY":
            key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("В .env нет GEMINI_API_KEY")
    return key


def _request(api_key: str, method: str, url: str, payload: dict | None = None) -> dict:
    """Один HTTP-запрос. На 429 и 5xx ждёт и повторяет, остальные коды отдаёт как ApiError."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    last_error: ApiError | None = None
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("x-goog-api-key", api_key)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = ApiError(exc.code, body)
            if exc.code == 429 or exc.code >= 500:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = ApiError(0, str(exc.reason))
            time.sleep(min(30, 2 ** attempt))
    assert last_error is not None
    raise last_error


def list_embed_models(api_key: str) -> list[str]:
    """Возвращает id моделей, в имени которых есть embed. Нужен пробнику перед индексацией."""
    found: list[str] = []
    token = ""
    while True:
        url = f"{BASE}/models?pageSize=100"
        if token:
            url += f"&pageToken={token}"
        page = _request(api_key, "GET", url)
        for item in page.get("models", []):
            name = str(item.get("name", ""))
            model_id = name.split("/", 1)[-1]
            if "embed" in model_id.lower():
                found.append(model_id)
        token = str(page.get("nextPageToken") or "")
        if not token:
            break
    return found


def _values(embedding: dict, output_dim: int) -> list[float]:
    """Достаёт числа вектора и обрезает до запрошенной размерности."""
    values = embedding.get("values") or []
    return [round(float(x), 6) for x in values[:output_dim]]


def embed_one(
    api_key: str,
    model: str,
    text: str,
    *,
    task_type: str | None,
    title: str | None = None,
    output_dim: int = OUTPUT_DIM,
) -> list[float]:
    """Один текст в один вектор. task_type передаётся только для gemini-embedding-001."""
    body: dict = {
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": output_dim,
    }
    if task_type:
        body["taskType"] = task_type
    if title and task_type == "RETRIEVAL_DOCUMENT":
        body["title"] = title[:200]
    url = f"{BASE}/models/{model}:embedContent"
    data = _request(api_key, "POST", url, body)
    return _values(data.get("embedding") or {}, output_dim)


def embed_batch(
    api_key: str,
    model: str,
    texts: list[str],
    *,
    task_type: str | None,
    titles: list[str] | None = None,
    output_dim: int = OUTPUT_DIM,
) -> list[list[float]]:
    """Пачка текстов. Если batchEmbedContents недоступен, считает по одному."""
    requests = []
    for i, text in enumerate(texts):
        item: dict = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": output_dim,
        }
        if task_type:
            item["taskType"] = task_type
        if titles and task_type == "RETRIEVAL_DOCUMENT":
            item["title"] = titles[i][:200]
        requests.append(item)
    url = f"{BASE}/models/{model}:batchEmbedContents"
    try:
        data = _request(api_key, "POST", url, {"requests": requests})
    except ApiError as exc:
        if exc.status != 404:
            raise
        vectors = []
        for i, text in enumerate(texts):
            title = titles[i] if titles else None
            vectors.append(
                embed_one(
                    api_key,
                    model,
                    text,
                    task_type=task_type,
                    title=title,
                    output_dim=output_dim,
                )
            )
        return vectors
    return [_values(item, output_dim) for item in data.get("embeddings", [])]


def is_embedding_v2(model: str) -> bool:
    """Embedding 2 не принимает task_type: задача пишется префиксом в самом тексте."""
    return model.startswith("gemini-embedding-2")


def prepare_document(model: str, heading: str, text: str) -> str:
    """Текст куска так, как его ждёт выбранная модель перед векторизацией."""
    if is_embedding_v2(model):
        title = heading.replace("|", "/") or "none"
        return f"title: {title} | text: {text}"
    return text


def prepare_query(model: str, question: str) -> str:
    """Текст вопроса так, как его ждёт выбранная модель."""
    if is_embedding_v2(model):
        return f"task: question answering | query: {question}"
    return question
