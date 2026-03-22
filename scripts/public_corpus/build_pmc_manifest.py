import argparse
import csv
import json
import math
import time
from pathlib import Path

import requests


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def load_keywords(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_query(keywords, field_tag):
    clauses = [f'"{keyword}"{field_tag}' for keyword in keywords]
    return f'(open access[filter]) AND (english[Language]) AND ({" OR ".join(clauses)})'


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_json(session, url, params, pause_seconds):
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    time.sleep(pause_seconds)
    return response.json()


def extract_article_id(entry, id_type):
    for item in entry.get("articleids", []):
        if str(item.get("idtype", "")).lower() == id_type:
            return item.get("value", "")
    direct_value = entry.get(id_type)
    return direct_value if direct_value not in (None, "") else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--retmax", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--field-tag", default="[Title/Abstract]")
    parser.add_argument("--email", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--pause-seconds", type=float, default=0.34)
    args = parser.parse_args()

    keywords = load_keywords(args.keywords_file)
    query = build_query(keywords, args.field_tag)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "codex-radonc-nlp/1.0 (public-corpus-manifest-builder)",
        }
    )

    shared_params = {}
    if args.email:
        shared_params["email"] = args.email
    if args.api_key:
        shared_params["api_key"] = args.api_key

    search_data = get_json(
        session,
        ESEARCH_URL,
        {
            "db": "pmc",
            "retmode": "json",
            "term": query,
            "retmax": args.retmax,
            **shared_params,
        },
        args.pause_seconds,
    )
    id_list = search_data["esearchresult"]["idlist"]
    total_hits = int(search_data["esearchresult"]["count"])

    rows = []
    for batch in batched(id_list, args.batch_size):
        summary_data = get_json(
            session,
            ESUMMARY_URL,
            {
                "db": "pmc",
                "retmode": "json",
                "id": ",".join(batch),
                **shared_params,
            },
            args.pause_seconds,
        )
        result = summary_data["result"]
        for uid in batch:
            entry = result.get(str(uid), {})
            pmcid = extract_article_id(entry, "pmcid")
            pmid = extract_article_id(entry, "pmid")
            doi = extract_article_id(entry, "doi")
            title = entry.get("title", "")
            authors = "; ".join(author.get("name", "") for author in entry.get("authors", []))
            pubdate = entry.get("pubdate", "")
            rows.append(
                {
                    "uid": str(uid),
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "pubdate": pubdate,
                    "authors": authors,
                    "query": query,
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
                    "bioc_json_url": (
                        f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
                        if pmcid
                        else ""
                    ),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid",
                "pmcid",
                "pmid",
                "doi",
                "title",
                "pubdate",
                "authors",
                "query",
                "source_url",
                "bioc_json_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(
        json.dumps(
            {
                "query": query,
                "keywords_count": len(keywords),
                "requested_retmax": args.retmax,
                "retrieved_rows": len(rows),
                "total_hits": total_hits,
                "summary_batches": math.ceil(len(id_list) / args.batch_size) if id_list else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
