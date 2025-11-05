# 1_extract_text.py
import subprocess
import pathlib
import shutil
import bz2
import json

DUMP = "viwiki-latest-pages-articles.xml.bz2"
OUT = "viwiki_txt"

# WikiExtractor: one sentence per line, keeps punctuation
_ = subprocess.run(
    [
        "python",
        "-m",
        "wikiextractor.WikiExtractor",
        "--json",  # each article → one json line
        "--no-templates",  # strip infoboxes, navboxes
        "--filter_disambig_pages",
        "--processes",
        "8",
        "-o",
        OUT,
        DUMP,
    ],
    check=True,
)

# Merge every part-*.bz2 into a single .txt
with open("viwiki_clean.txt", "w", encoding="utf-8") as fout:
    for f in pathlib.Path(OUT).rglob("wiki_*"):
        with bz2.open(f, "rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                _ = fout.write(data["text"].replace("\n", " ") + "\n")
shutil.rmtree(OUT)
print("✅ Vietnamese Wikipedia text → viwiki_clean.txt")
