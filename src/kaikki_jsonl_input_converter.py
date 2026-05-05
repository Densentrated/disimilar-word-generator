import json
import logging
from typing import List, Tuple

import pandas as pd

import input_format as input_format

logger = logging.getLogger(__name__)


class KaikkiJsonInputConverter:
    """Convert Kaikki JSONL input file to an InputFormat object.

    Behavior:
    - If a record has no `senses` key, the returned definition will be a
      descriptive failure message: "failed: missing 'senses'".
    - If `senses` exists but no gloss strings can be found, the returned
      definition will be: "failed: no glosses found in 'senses'".
    - If parsing the JSON line fails, the definition will be:
      "failed: parse_error: <short message>".
    - When glosses are present they are joined with " | " into a single string.
    """

    def Kaikki_To_Input(self, json_file: str) -> input_format.InputFormat:
        df = self._parse_jsonl(json_file)
        return input_format.InputFormat(df)

    def _parse_jsonl(self, json_file: str) -> pd.DataFrame:
        """Read a JSONL file and produce a DataFrame with the intermediary columns.

        Columns:
          - From-Language word
          - To-Language definitions (always a string; may contain a failure message)
        """
        words: List[str] = []
        definitions: List[str] = []

        with open(json_file, "r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=0):
                line = raw_line.strip()
                if not line:
                    # keep alignment by skipping empty line entirely
                    logger.debug("Skipping empty line %d", lineno)
                    continue

                try:
                    word, definition = self._parse_json(line)
                except json.JSONDecodeError as e:
                    # JSON syntax error -> return a clear failure message
                    logger.warning("JSON decode error on line %d: %s", lineno, e)
                    try:
                        # attempt to recover a word if possible
                        partial = json.loads(line.rstrip(", \n\r"))
                        word = partial.get("word", "")
                    except Exception:
                        word = ""
                    definition = f"failed: parse_error: {e.msg}"
                except Exception as e:
                    # Any other unexpected error -> record it as a failure message
                    logger.exception("Unexpected error parsing line %d", lineno)
                    try:
                        data = json.loads(line)
                        word = data.get("word", "")
                    except Exception:
                        word = ""
                    definition = f"failed: unexpected_error: {str(e)}"

                # For interactive runs we keep a simple print; callers can switch to logging.
                print(f"{word}: {definition}")
                words.append(word)
                definitions.append(definition)

        df = pd.DataFrame(
            {"From-Language word": words, "To-Language definition": definitions}
        )
        return df

    def _parse_json(self, json_text: str) -> Tuple[str, str]:
        """Parse a single JSONL line and extract (word, definition_string).

        Guarantees:
          - Always returns strings for both word and definition.
          - definition will be a failure message when extraction couldn't succeed.
        """
        data = json.loads(json_text)
        word = data.get("word", "")

        senses = data.get("senses", None)
        if senses is None:
            return word, "failed: missing 'senses'"

        collected_glosses: List[str] = []

        # Case: senses is a dict with 'glosses'
        if isinstance(senses, dict):
            g = senses.get("glosses")
            if isinstance(g, list):
                for item in g:
                    if isinstance(item, str) and item:
                        collected_glosses.append(item)

        # Case: senses is a list of sense objects (typical Kaikki structure)
        elif isinstance(senses, list):
            for sense in senses:
                if not isinstance(sense, dict):
                    continue
                g = sense.get("glosses")
                if isinstance(g, list):
                    for item in g:
                        if isinstance(item, str) and item:
                            collected_glosses.append(item)

        # If nothing was collected, return a failure message describing why
        if not collected_glosses:
            return word, "failed: no glosses found in 'senses'"

        # Otherwise join glosses into a single string for the DataFrame cell
        definition = " | ".join(collected_glosses)
        return word, definition
