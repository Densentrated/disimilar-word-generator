import logging
import os
from pathlib import Path
from typing import Optional

import requests


class WordListDownloader:
    def __init__(self, language_code: str, cache_dir: str = "./wordlist_cache"):
        """
        Initializes the downlooader with language code and a cache directory
        """

        self.language_code = language_code
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def download(self):
        """
        Downloads the wordlist file for the specified langauge code.
        """
        url = f"https://raw.githubusercontent.com/hermitdave/FrequencyWords/refs/heads/master/content/2018/{self.language_code}/{self.language_code}_full.txt"
        dest_path = os.path.join(self.cache_dir, f"{self.language_code}.txt")
        print(f"downloading wordlist from {url} to {dest_path}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded successfully: {dest_path}")
            return dest_path
        else:
            print(
                f"failed to download wordlist for {self.language_code}: {response.status_code}"
            )
            return None


if __name__ == "__main__":
    print("downloading wordlist")
    viet_downloader = WordListDownloader("vi")
    viet_downloader.download()
    print("downloaidng completed")
