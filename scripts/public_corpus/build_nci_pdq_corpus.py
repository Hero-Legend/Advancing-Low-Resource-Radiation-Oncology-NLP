import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


INDEX_URL = "https://www.cancer.gov/publications/pdq/information-summaries"


def load_keywords(path):
    return [
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_category_links(session):
    response = session.get(INDEX_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(INDEX_URL, anchor["href"]).split("#")[0]
        if "/publications/pdq/information-summaries/" not in href:
            continue
        if href == INDEX_URL:
            continue
        if "/espanol/" in href:
            continue
        links.add(href)
    return sorted(links)


def discover_health_professional_links(session):
    links = set()
    for category_url in discover_category_links(session):
        response = session.get(category_url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = urljoin(category_url, anchor["href"]).split("#")[0]
            lowered = href.lower()
            if "/espanol/" in lowered:
                continue
            if "/hp/" in lowered and lowered.endswith("-pdq"):
                links.add(href)
                continue
            if lowered.endswith("-hp-pdq"):
                links.add(href)
    return sorted(links)


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_page(session, url):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    if soup.find("h1"):
        title = clean_text(soup.find("h1").get_text(" ", strip=True))
    elif soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

    updated = ""
    for tag in soup.find_all(["time", "meta"]):
        if tag.name == "time":
            updated = clean_text(tag.get_text(" ", strip=True))
            if updated:
                break
        if tag.name == "meta" and str(tag.get("property", "")).lower() in {"article:modified_time", "og:updated_time"}:
            updated = tag.get("content", "")
            if updated:
                break

    main_node = soup.find("main") or soup.find("article") or soup.find(id="cgvBody")
    if main_node is None:
        main_node = soup.body
    text = clean_text(main_node.get_text(" ", strip=True)) if main_node else ""
    return {"title": title, "updated": updated, "text": text}


def keyword_hit(text, keywords):
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    keywords = load_keywords(args.keywords_file)
    session = requests.Session()
    session.headers.update({"User-Agent": "codex-radonc-nlp/1.0 (nci-pdq-builder)"})

    links = discover_health_professional_links(session)
    if args.limit > 0:
        links = links[: args.limit]

    rows = []
    docs = []
    for url in links:
        payload = extract_page(session, url)
        combined = f"{payload['title']} {payload['text']}"
        if len(payload["text"]) < args.min_chars:
            continue
        if not keyword_hit(combined, keywords):
            continue
        doc_id = url.rstrip("/").split("/")[-2] + "_healthprofessional"
        row = {
            "doc_id": doc_id,
            "title": payload["title"],
            "url": url,
            "updated": payload["updated"],
            "char_count": len(payload["text"]),
        }
        rows.append(row)
        docs.append(
            {
                "source": "nci_pdq",
                "doc_id": doc_id,
                "title": payload["title"],
                "url": url,
                "updated": payload["updated"],
                "text": payload["text"],
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "title", "url", "updated", "char_count"])
        writer.writeheader()
        writer.writerows(rows)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(json.dumps({"candidate_links": len(links), "kept_documents": len(docs)}, indent=2))


if __name__ == "__main__":
    main()
