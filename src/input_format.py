"""Input format holder.

Provides the `InputFormat` class which encapsulates a pandas
DataFrame with two columns: `word` and `definition`.
"""

from typing import Iterable, Optional, Tuple

import pandas as pd


class InputFormat:
    """Hold and manage a DataFrame with `word` and `definition` columns.

    Example:
        obj = InputFormat()
        obj.add("apple", "a fruit")
        df = obj.to_dataframe()
    """

    REQUIRED_COLUMNS = ["From-Language word", "To-Language definition"]

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            self._df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        else:
            self._df = df.copy()
            self._validate()

    def _validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self._df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def add(self, word: str, definition: str) -> None:
        """Append a new row to the internal DataFrame."""
        row = {"word": word, "definition": definition}
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the internal DataFrame."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        for _, row in self._df.iterrows():
            yield row["word"], row["definition"]


__all__ = ["InputFormat"]
