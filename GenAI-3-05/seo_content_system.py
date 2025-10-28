"""
GenAI-3-05 — Система генерации SEO-контента

Функциональность:
1) Принимает тему и (опционально) список ключевых слов.
2) Если ключевые слова не заданы — генерирует синтетические.
3) Генерирует SEO title, meta description и краткое summary.
4) Проверяет длины и наличие ключевых слов.
5) Сохраняет результат в JSON.
6) Печатает краткий отчёт в консоль.

Работает локально через Ollama (OpenAI-совместимый API). Интернет не требуется.
"""

from __future__ import annotations
import os
import re
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# OpenAI-совместимый клиент (используется для обращения к Ollama)
from openai import OpenAI


# -------------------- Утилиты --------------------

def get_client():
    """
    Клиент для локальной модели Ollama.
    Работает через API, совместимый с OpenAI.
    Никаких ключей и внешних сервисов не требуется.
    """
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # любое непустое значение
    )


def detect_language(text: str) -> str:
    """Простая эвристика для определения языка (ru/en)."""
    cyr = sum('а' <= ch.lower() <= 'я' or ch == 'ё' for ch in text)
    lat = sum('a' <= ch.lower() <= 'z' for ch in text)
    return 'ru' if cyr >= lat else 'en'


def missing_keywords(text: str, keywords: Optional[List[str]]) -> List[str]:
    """Возвращает список отсутствующих ключевых слов в тексте."""
    if not keywords:
        return []
    t = text.lower()
    return [kw for kw in keywords if kw.lower() not in t]


# -------------------- Основные структуры --------------------

@dataclass
class FieldCheck:
    length: int
    max_allowed: int
    ok_length: bool
    missing_keywords: List[str]


@dataclass
class SEOResult:
    topic: str
    language: str
    keywords: List[str]
    meta_description: str
    title: str
    summary: str
    checks: Dict[str, FieldCheck]
    model_used: str
    timestamp: str


# -------------------- Основная система --------------------

class SEOContentSystem:
    """Реализация задания GenAI-3-05."""
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.client = get_client()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    # ---------- LLM запрос ----------
    def _chat(self, model: str, system: str, user: str, max_tokens: int = 200) -> str:
        """Вспомогательный метод для общения с моделью."""
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        text = getattr(msg, "content", None) or getattr(resp.choices[0], "text", None)
        if not text and hasattr(msg, "reasoning_content"):
            text = msg.reasoning_content
        return (text or "").strip()

    # ---------- Генерация ключевых слов ----------
    def generate_synthetic_keywords(self, topic: str, language: str, n: int = 8) -> List[str]:
        """Создаёт список ключевых слов, если пользователь не задал их вручную."""
        if language == "ru":
            system = "Ты SEO-специалист. Отвечай списком, по одному слову или фразе на строку."
            user = f"Сгенерируй {n} релевантных ключевых слов по теме: {topic}."
        else:
            system = "You are an SEO specialist. Reply with a list, one keyword or phrase per line."
            user = f"Generate {n} relevant keywords for the topic: {topic}."

        text = self._chat(self.model, system, user, max_tokens=200)
        kws = [re.sub(r'^[\-\d\.\)\s]+', '', line).strip() for line in text.splitlines() if line.strip()]
        seen, result = set(), []
        for kw in kws:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                result.append(kw)
        return result[:n]

    # ---------- Meta description ----------
    def generate_meta_description(self, topic: str, language: str, keywords: Optional[List[str]]) -> str:
        """Генерация meta description."""
        if language == "ru":
            sys = "Ты маркетолог. Ответь ТОЛЬКО meta description (до 150 символов)."
            kw = f" Включи слова: {', '.join(keywords)}." if keywords else ""
            user = f"Напиши meta description для сайта о {topic}.{kw}"
        else:
            sys = "You are a marketing specialist. Reply ONLY with a meta description (<=150 characters)."
            kw = f" Include words: {', '.join(keywords)}." if keywords else ""
            user = f"Write a meta description for a website about {topic}.{kw}"
        text = self._chat(self.model, sys, user, max_tokens=120)
        return text[:150].strip()

    # ---------- Title ----------
    def generate_title(self, topic: str, language: str, keywords: Optional[List[str]]) -> str:
        """Генерация SEO-заголовка."""
        if language == "ru":
            sys = "Ты SEO-копирайтер. Верни ТОЛЬКО заголовок (до 60 символов)."
            kw = f" Добавь: {', '.join(keywords)}." if keywords else ""
            user = f"Создай кликабельный title по теме: {topic}.{kw}"
        else:
            sys = "You are an SEO copywriter. Return ONLY a title (<=60 chars)."
            kw = f" Include: {', '.join(keywords)}." if keywords else ""
            user = f"Create a catchy SEO title about: {topic}.{kw}"
        text = self._chat(self.model, sys, user, max_tokens=60)
        return text[:60].strip()

    # ---------- Summary ----------
    def generate_summary(self, topic: str, language: str, keywords: Optional[List[str]]) -> str:
        """Генерация краткого описания (summary)."""
        if language == "ru":
            sys = "Ты редактор. Верни 1–2 предложения (до 300 символов)."
            kw = f" Включи: {', '.join(keywords)}." if keywords else ""
            user = f"Кратко опиши тему: {topic}.{kw}"
        else:
            sys = "You are an editor. Return 1–2 sentences (<=300 chars)."
            kw = f" Include: {', '.join(keywords)}." if keywords else ""
            user = f"Briefly describe the topic: {topic}.{kw}"
        text = self._chat(self.model, sys, user, max_tokens=180)
        return text[:300].strip()

    # ---------- Проверки ----------
    def _check_field(self, text: str, max_len: int, keywords: Optional[List[str]]) -> FieldCheck:
        """Проверяет длину и наличие ключевых слов."""
        return FieldCheck(
            length=len(text),
            max_allowed=max_len,
            ok_length=len(text) <= max_len,
            missing_keywords=missing_keywords(text, keywords),
        )

    # ---------- Основной сценарий ----------
    def run(
        self,
        topic: str,
        language: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        synth_keywords_if_missing: bool = True,
        out_json_path: Optional[str] = None,
    ) -> SEOResult:
        """Основной процесс генерации SEO-контента."""
        lang = language or detect_language(topic)
        kws = keywords or []
        if not kws and synth_keywords_if_missing:
            kws = self.generate_synthetic_keywords(topic, lang)

        meta = self.generate_meta_description(topic, lang, kws)
        title = self.generate_title(topic, lang, kws)
        summary = self.generate_summary(topic, lang, kws)

        checks = {
            "meta_description": self._check_field(meta, 150, kws),
            "title": self._check_field(title, 60, kws),
            "summary": self._check_field(summary, 300, kws),
        }

        result = SEOResult(
            topic=topic,
            language=lang,
            keywords=kws,
            meta_description=meta,
            title=title,
            summary=summary,
            checks={k: asdict(v) for k, v in checks.items()},
            model_used=self.model,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if out_json_path:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        return result


# -------------------- Форматированный отчёт --------------------

def pretty_report(res: SEOResult) -> str:
    """Форматированный отчёт для вывода в консоль."""
    ch = res.checks

    def mark(ok: bool) -> str:
        return "✓" if ok else "✗"

    meta_ok = ch['meta_description']['ok_length'] and not ch['meta_description']['missing_keywords']
    title_ok = ch['title']['ok_length'] and not ch['title']['missing_keywords']
    summary_ok = ch['summary']['ok_length'] and not ch['summary']['missing_keywords']

    def list_or_dash(items): return ", ".join(items) if items else "—"

    report = [
        f"Тема: {res.topic}",
        f"Язык: {res.language}",
        f"Ключевые слова: {', '.join(res.keywords) if res.keywords else '—'}",
        "",
        f"TITLE ({ch['title']['length']}/{ch['title']['max_allowed']}): {res.title}",
        f"  Длина ОК: {mark(ch['title']['ok_length'])}; Отсутствуют ключевые: {list_or_dash(ch['title']['missing_keywords'])}",
        "",
        f"META ({ch['meta_description']['length']}/{ch['meta_description']['max_allowed']}): {res.meta_description}",
        f"  Длина ОК: {mark(ch['meta_description']['ok_length'])}; Отсутствуют ключевые: {list_or_dash(ch['meta_description']['missing_keywords'])}",
        "",
        f"SUMMARY ({ch['summary']['length']}/{ch['summary']['max_allowed']}): {res.summary}",
        f"  Длина ОК: {mark(ch['summary']['ok_length'])}; Отсутствуют ключевые: {list_or_dash(ch['summary']['missing_keywords'])}",
        "",
        f"Модель: {res.model_used}",
        f"Время: {res.timestamp}",
        "",
        "Базовый критерий: контент создан.",
        f"Итоговая метрика (все требования выполнены): {mark(meta_ok and title_ok and summary_ok)}",
    ]
    return "\n".join(report)


# -------------------- CLI --------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GenAI-3-05: SEO Content System (локальная версия через Ollama)")
    parser.add_argument("topic", type=str, help="Тема/продукт/сайт")
    parser.add_argument("--keywords", nargs="+", help="Ключевые слова (через пробел)")
    parser.add_argument("-l", "--language", type=str, help="Язык: ru или en")
    parser.add_argument("-m", "--model", type=str, default="qwen2.5:3b-instruct", help="Имя модели Ollama")
    parser.add_argument("--no-synth", action="store_true", help="Не генерировать ключевые слова, если не заданы")
    parser.add_argument("-o", "--out", type=str, default="seo_output.json", help="Файл для сохранения JSON")
    args = parser.parse_args()

    system = SEOContentSystem(model=args.model)
    result = system.run(
        topic=args.topic,
        language=args.language,
        keywords=args.keywords,
        synth_keywords_if_missing=not args.no_synth,
        out_json_path=args.out,
    )
    print(pretty_report(result))


if __name__ == "__main__":
    main()
