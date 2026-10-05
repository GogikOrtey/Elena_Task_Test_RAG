"""Нарезка markdown-конспектов на куски для RAG.

Идёт по заголовкам # / ## / ### и собирает цепочку заголовков.
Длинную секцию делит по абзацам, чтобы кусок уместился в лимит эмбеддинга.
Вызывается из index_build.py. Источник — папка Материал_тексты в корне проекта.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ROOT / "Материал_тексты"
MAX_CHARS = 1500
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Делит текст на части не длиннее limit. Сначала абзацы, потом предложения."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        units = [paragraph] if len(paragraph) <= limit else _split_long(paragraph, limit)
        for unit in units:
            if current and len(current) + 2 + len(unit) > limit:
                pieces.append(current)
                current = unit
            elif current:
                current = current + "\n\n" + unit
            else:
                current = unit
    if current:
        pieces.append(current)
    return pieces


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Режет абзац длиннее limit по предложениям, иначе по символам."""
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            for start in range(0, len(sentence), limit):
                parts.append(sentence[start : start + limit])
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        elif current:
            current = current + " " + sentence
        else:
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_file(path: Path) -> list[dict]:
    """Один markdown-файл в список кусков: id, file, heading, text."""
    rel = path.relative_to(ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    chunks: list[dict] = []

    def flush() -> None:
        heading = " > ".join(title for _, title in stack) or path.stem
        pieces = split_text("\n".join(buf))
        for index, piece in enumerate(pieces, start=1):
            label = heading if len(pieces) == 1 else f"{heading} (часть {index})"
            raw = f"{rel}\n{label}\n{piece}"
            chunk_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                {
                    "id": chunk_id,
                    "file": rel,
                    "heading": label,
                    "text": piece,
                }
            )

    for line in lines:
        match = HEADING.match(line)
        if match:
            flush()
            buf = []
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        buf.append(line)
    flush()
    return [item for item in chunks if item["text"].strip()]


def load_chunks() -> list[dict]:
    """Все .md из Материал_тексты, по имени файла. Пустая папка — ошибка запуска."""
    paths = sorted(MATERIALS.glob("*.md"))
    if not paths:
        raise SystemExit(f"В папке нет markdown: {MATERIALS}")
    chunks: list[dict] = []
    for path in paths:
        chunks.extend(chunk_file(path))
    return chunks
