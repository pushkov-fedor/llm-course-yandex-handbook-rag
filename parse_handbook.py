#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /home/fedor/Study/llm-course/parse_handbook.py

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup, Tag

# pip install requests beautifulsoup4 lxml markdownify unidecode
from markdownify import markdownify as md
from unidecode import unidecode


@dataclass
class TocItem:
    chapter_num: str
    chapter_title: str
    article_num: str
    article_title: str
    url: str
    file_path: Optional[str] = None


def slugify(s: str, max_len: int = 80) -> str:
    s = s.strip()
    s = unidecode(s)  # рус -> latin
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "item"
    return s[:max_len]


def clean_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def is_display_mode(span: Tag) -> bool:
    opts = span.get("data-options") or ""
    return "displayMode" in opts and "true" in opts.lower()


def normalize_math_in_html(container: Tag) -> None:
    """
    Заменяем <span class="yfm-latex" data-content="...">...</span>
    на текстовые маркеры $...$ / $$...$$, чтобы markdownify не съел формулу.
    """
    for sp in list(container.select("span.yfm-latex")):
        latex_raw = sp.get("data-content") or ""
        latex = unquote(latex_raw)

        if not latex:
            # fallback: иногда KaTeX кладёт TeX в annotation
            ann = sp.select_one("annotation[encoding='application/x-tex']")
            latex = ann.get_text(strip=True) if ann else ""

        if not latex:
            continue

        if is_display_mode(sp) or sp.name == "p" and "yfm-latex" in sp.get("class", []):
            repl = f"\n\n$$\n{latex}\n$$\n\n"
        else:
            repl = f"${latex}$"

        sp.replace_with(repl)


def normalize_images_in_html(container: Tag) -> None:
    """
    <figure><img src=... alt=...></figure> -> markdown картинка
    Обрабатываем figure теги, чтобы сохранить структуру изображений
    """
    for figure in list(container.select("figure.fig-img, figure")):
        img = figure.select_one("img")
        if not img:
            continue
        src = img.get("src") or ""
        alt = (img.get("alt") or "").strip()
        if src:
            # Заменяем весь figure на markdown изображение
            figure.replace_with(f"![{alt}]({src})\n")
    # Обрабатываем оставшиеся img без figure
    for img in list(container.select("img")):
        if img.find_parent("figure"):
            continue  # уже обработано выше
        src = img.get("src") or ""
        alt = (img.get("alt") or "").strip()
        if src:
            img.replace_with(f"![{alt}]({src})\n")


def extract_article_markdown(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#wysiwyg-client-content")
    if not content:
        return None

    # чистим "кнопки заметок" и т.п. — берём только сам wysiwyg
    normalize_math_in_html(content)
    normalize_images_in_html(content)

    # markdownify: конвертим HTML -> MD
    md_text = md(
        str(content),
        heading_style="ATX",
        bullets="-",
        strip=["span"],  # после нормализации math/span уже текст
    )

    # Исправляем экранированные подчеркивания в URL изображений
    # markdownify может экранировать _ в URL как \_, заменяем обратно
    def fix_image_urls(match):
        alt = match.group(1)
        url = match.group(2).replace("\\_", "_")
        return f"![{alt}]({url})"
    
    md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_image_urls, md_text)

    # подчистим мусор
    md_text = re.sub(r"\n[ \t]+\n", "\n\n", md_text)
    md_text = clean_spaces(md_text)
    return md_text


def parse_toc(main_html: str, main_url: str) -> List[TocItem]:
    soup = BeautifulSoup(main_html, "lxml")

    items: List[TocItem] = []

    # У тебя структура: top <ul> -> top <li> (глава) -> h2 (название главы) + inner <ul aria-label="..."> (статьи)
    # Классы хэшированные, поэтому цепляемся за семантику: h2 + вложенный ul с ссылками ./ml/article/...
    for top_li in soup.select("ul li"):
        h2 = top_li.select_one("h2")
        inner_ul = top_li.select_one("ul[aria-label]")

        if not h2 or not inner_ul:
            continue

        chapter_title_full = h2.get_text(" ", strip=True)  # "1. Введение"
        m = re.match(r"^\s*(\d+)\.\s*(.+)$", chapter_title_full)
        if not m:
            continue

        chapter_num = m.group(1)
        chapter_title = m.group(2)

        for li in inner_ul.select("li"):
            a = li.select_one("a[href]")
            num_span = li.select_one("span[class*='article-number']")
            title_span = li.select_one("span[class*='title']")

            if not a:
                continue
            href = a.get("href", "")
            if "/article/" not in href and "./ml/article/" not in href:
                continue

            article_num = num_span.get_text(strip=True) if num_span else ""
            article_title = title_span.get_text(" ", strip=True) if title_span else a.get_text(" ", strip=True)

            url = urljoin(main_url, href)
            items.append(
                TocItem(
                    chapter_num=chapter_num,
                    chapter_title=chapter_title,
                    article_num=article_num or f"{chapter_num}.?",
                    article_title=article_title,
                    url=url,
                )
            )

    # дедуп (на всякий)
    seen = set()
    uniq: List[TocItem] = []
    for it in items:
        key = (it.article_num, it.url)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return uniq


def http_get(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=40)
    r.raise_for_status()
    return r.text


def get_article_path(out_dir: Path, item: TocItem) -> Path:
    """Вычисляет путь до файла статьи (относительно out_dir)."""
    chapter_dir = f"{item.chapter_num}_{slugify(item.chapter_title)}"
    fname = f"{item.article_num}_{slugify(item.article_title)}.md"
    return out_dir / chapter_dir / fname


def save_article(out_dir: Path, item: TocItem, md_text: str) -> Path:
    fpath = get_article_path(out_dir, item)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# {item.article_num} {item.article_title}\n\n"
    )
    fpath.write_text(header + md_text, encoding="utf-8")
    return fpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", default="https://education.yandex.ru/handbook/ml", help="URL главной учебника")
    ap.add_argument("--out", default="handbook", help="папка для выгрузки")
    ap.add_argument("--limit", type=int, default=0, help="ограничить кол-во статей (0 = все)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept-Language": "ru,en;q=0.9",
        }
    )

    main_html = http_get(s, args.handbook)
    toc = parse_toc(main_html, args.handbook)

    if args.limit and args.limit > 0:
        toc = toc[: args.limit]

    # Вычисляем file_path для каждого item
    for item in toc:
        item.file_path = str(get_article_path(out_dir, item))

    index = {
        "handbook": args.handbook,
        "count": len(toc),
        "items": [asdict(x) for x in toc],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    ok, fail = 0, 0
    for it in toc:
        try:
            html = http_get(s, it.url)
            md_text = extract_article_markdown(html)
            if not md_text:
                raise RuntimeError("Не нашёл #wysiwyg-client-content (возможно, контент грузится JS)")

            save_article(out_dir, it, md_text)
            ok += 1
            print(f"[OK] {it.article_num} {it.article_title}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {it.article_num} {it.article_title} :: {e}")

    print(f"\nDone. ok={ok} fail={fail}")
    print(f"Index: {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()

