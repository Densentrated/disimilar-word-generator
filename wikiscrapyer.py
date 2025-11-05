#!/usr/bin/env python3
"""
turn Vietnamese Wikipedia dump → vi_lemmas.json
Usage:
    pip install wikiextractor underthesea tqdm
Output:
    vi_lemmas.json   (50 000 most frequent lemmas)
"""

import os
import json
import re
import subprocess
import sys
from collections import Counter
from tqdm import tqdm
import underthesea as vi

DUMP_URL = (
    "https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2"
)
DUMP_FILE = "viwiki-latest-pages-articles.xml.bz2"
CORPUS_DIR = "corpus"
LEMMAS_FILE = "vi_lemmas.json"
TOP_N = 50_000


def download():
    if not os.path.exists(DUMP_FILE):
        print("Downloading Wikipedia dump …")
        _ = subprocess.run(["wget", "-q", "-O", DUMP_FILE, DUMP_URL], check=True)


def extract():
    if not os.path.exists(CORPUS_DIR):
        print("Extracting XML → raw text …")
        _ = subprocess.run(
            [
                sys.executable,
                "-m",
                "wikiextractor.WikiExtractor",
                DUMP_FILE,
                "-o",
                CORPUS_DIR,
                "--json",
            ],
            check=True,
        )


def build_freq():
    print("Tokenising and counting …")
    cnt = Counter()
    for root, _, files in os.walk(CORPUS_DIR):
        for f in tqdm([f for f in files if f.endswith(".bz2")]):
            with subprocess.Popen(
                ["bunzip2", "-c", os.path.join(root, f)],
                stdout=subprocess.PIPE,
                text=True,
                errors="ignore",
            ) as p:
                if p.stdout is None:
                    continue
                for line in p.stdout:  # type: ignore[union-attr]
                    if not line.strip():
                        continue
                    # WikiExtractor JSON format: {"text":"..."}
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    text = re.sub(r"<[^>]+>", "", text)
                    tokens = vi.word_tokenize(text.lower())
                    cnt.update(tokens)
    print(f"Saving top {TOP_N} lemmas → {LEMMAS_FILE}")
    json.dump(
        cnt.most_common(TOP_N),
        open(LEMMAS_FILE, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )


if __name__ == "__main__":
    download()
    extract()
    build_freq()
    print("Done – vi_lemmas.json ready.")
