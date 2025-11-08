"""
This file takes a wikipedia dumpfile, and turns it into a list of words
"""

import sys
import bz2
import re
from typing import Set
import gc


class WikiExtractor:
    """class that allows for the extraction of words from a wikipedia dump"""

    def __init__(self, dump_path: str):
        """Initialize the Wiki Extractor"""
        self.dump_path: str = dump_path
        self.timeout_seconds = 120  # 2 minute timeout
        self.temp_file = "temp_words_all.txt"
        self.total_words_written = 0

    def write_words_to_file(self, words: Set[str]):
        """Write words directly to temp file"""
        with open(self.temp_file, "a", encoding="utf-8") as f:
            for word in words:
                f.write(f"{word}\n")
        self.total_words_written += len(words)

    def has_vietnamese_diacritic(self, text: str) -> bool:
        """Check if text contains any Vietnamese diacritical characters"""
        vietnamese_chars = (
            "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        )
        return any(char in vietnamese_chars for char in text.lower())

    def extract_words(self, text: str) -> Set[str]:
        """Extract words from text, only from sentences containing Vietnamese diacritics"""
        # Remove wiki markup
        # Remove templates {{...}}
        text = re.sub(r"\{\{[^}]+\}\}", "", text)
        # Remove links but keep the text [[link|text]] -> text
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        # Remove references <ref>...</ref>
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)
        # Remove category and file references
        text = re.sub(
            r"\[\[(?:Category|Thể loại|File|Tập tin):([^\]]+)\]\]",
            "",
            text,
            re.IGNORECASE,
        )

        # Split text into sentences (split on common punctuation)
        sentence_pattern = r"[.!?;]\s+|\n+"
        sentences = re.split(sentence_pattern, text)

        # Vietnamese alphabet pattern
        word_pattern = r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+"

        meaningful_words = set()

        for sentence in sentences:
            # Only process sentence if it contains Vietnamese diacritics
            if self.has_vietnamese_diacritic(sentence):
                # Extract all words from this sentence
                words = re.findall(word_pattern, sentence)
                for word in words:
                    word_lower = word.lower()
                    # Keep words with at least 2 characters
                    if len(word_lower) >= 2:
                        meaningful_words.add(word_lower)

        return meaningful_words

    def parse_dump_regex(self):
        """Parse dump using regex instead of XML parser for better memory efficiency"""
        print(f"Parsing Wikipedia dump: {self.dump_path}")
        print("Processing all articles (no category filtering)")
        print("Using memory-efficient regex parsing...")

        # Clear temp file if it exists
        import os

        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

        articles_processed = 0
        articles_matched = 0

        try:
            # Open bz2 compressed file
            if self.dump_path.endswith(".bz2"):
                file_handle = bz2.open(self.dump_path, "rt", encoding="utf-8")
            else:
                file_handle = open(self.dump_path, "r", encoding="utf-8")

            # Buffer for current page
            in_page = False
            in_text = False
            current_text = []
            text_lines = 0

            for line in file_handle:
                # Detect page boundaries
                if "<page>" in line:
                    in_page = True
                    current_text = []
                    text_lines = 0
                elif "</page>" in line and in_page:
                    # Process accumulated page
                    if current_text:
                        articles_processed += 1
                        text = "".join(current_text)

                        # Extract words from all articles
                        words = self.extract_words(text)
                        if words:  # Only count as matched if we found Vietnamese words
                            articles_matched += 1
                            self.write_words_to_file(words)

                        # Clear text buffer to free memory
                        current_text = []
                        text_lines = 0

                    in_page = False

                    if articles_processed % 1000 == 0:
                        gc.collect()
                        print(
                            f"  Processed {articles_processed} articles, matched {articles_matched}, wrote {self.total_words_written} words to temp file"
                        )

                elif in_page:
                    # Detect text content
                    if "<text" in line:
                        in_text = True
                        # Extract text from same line if it's there
                        text_match = re.search(r"<text[^>]*>(.*)", line, re.DOTALL)
                        if text_match:
                            current_text.append(text_match.group(1))
                            text_lines += 1
                    elif "</text>" in line:
                        # Get remaining text before closing tag
                        text_match = re.search(r"(.*)</text>", line, re.DOTALL)
                        if text_match:
                            current_text.append(text_match.group(1))
                        in_text = False
                    elif in_text:
                        current_text.append(line)
                        text_lines += 1

                        # Prevent excessive memory usage from huge articles
                        if text_lines > 10000:
                            in_text = False
                            current_text = []

            file_handle.close()

            print(f"\nProcessing complete!")
            print(f"Total articles processed: {articles_processed}")
            print(f"Articles matched: {articles_matched}")
            print(f"Total words written (with duplicates): {self.total_words_written}")

        except Exception as e:
            print(f"Error parsing dump file: {e}")
            raise

    def parse_dump(self):
        """Parse the Wikipedia dump file - uses regex method for better memory efficiency"""
        return self.parse_dump_regex()

    def save_words(self, output_file: str = "extracted_words.txt"):
        """Deduplicate temp file and save to output file using external sort"""
        import os
        import subprocess

        if not os.path.exists(self.temp_file):
            print("No temporary file found. No words to save.")
            return

        print(f"Sorting and deduplicating words from {self.temp_file}...")
        print("Using external sort command for memory efficiency...")

        try:
            # Use system sort command with unique flag for memory-efficient deduplication
            # LC_ALL=C makes sort faster by using byte comparison
            sorted_file = f"{self.temp_file}.sorted"

            # Sort and deduplicate in one pass using external command
            subprocess.run(
                f"LC_ALL=C sort -u {self.temp_file} > {sorted_file}",
                shell=True,
                check=True,
            )

            # Count unique words
            word_count = 0
            with open(sorted_file, "r", encoding="utf-8") as f:
                for _ in f:
                    word_count += 1

            print(f"Total unique words: {word_count}")

            # Move sorted file to output
            os.rename(sorted_file, output_file)

            # Clean up temp file
            os.remove(self.temp_file)
            print(f"Removed temporary file {self.temp_file}")
            print(f"Words saved to {output_file}")

        except subprocess.CalledProcessError:
            print("External sort failed, falling back to streaming deduplication...")
            # Fallback: manual streaming deduplication (no in-memory set)
            sorted_file = f"{self.temp_file}.sorted"

            # First sort the file
            subprocess.run(
                f"LC_ALL=C sort {self.temp_file} > {sorted_file}",
                shell=True,
                check=True,
            )

            # Then deduplicate by streaming (only keep if different from previous)
            word_count = 0
            with open(sorted_file, "r", encoding="utf-8") as infile:
                with open(output_file, "w", encoding="utf-8") as outfile:
                    prev_word = None
                    for line in infile:
                        word = line.strip()
                        if word and word != prev_word:
                            outfile.write(f"{word}\n")
                            word_count += 1
                            prev_word = word

            print(f"Total unique words: {word_count}")
            os.remove(sorted_file)
            os.remove(self.temp_file)
            print(f"Words saved to {output_file}")

    def get_unique_words(self) -> Set[str]:
        """Return the set of unique words extracted from temp file"""
        import os

        all_words = set()

        # Read from temp file if it exists
        if os.path.exists(self.temp_file):
            with open(self.temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        all_words.add(word)

        return all_words


def main():
    if len(sys.argv) < 2:
        print("Usage: python wikepedia_parser.py <wikidump.xml.bz2> [output_file]")
        sys.exit(1)

    filename = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_words.txt"

    extractor = WikiExtractor(filename)
    extractor.parse_dump()
    extractor.save_words(output_file)

    print(f"\nExtraction complete! Check {output_file} for unique words.")


if __name__ == "__main__":
    main()
