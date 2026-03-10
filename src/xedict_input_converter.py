import pandas as pd

import input_format


class XedictInputConverter:
    """Convert XEDICT format to the input format of the program"""

    def Xedict_To_Input(self, text_file: str) -> input_format.InputFormat:
        words = []
        definitions = []
        with open(text_file, "r", encoding="utf-8") as f:
            for linenum, raw_line in enumerate(f, start=0):
                line = raw_line.strip()
                word = ""
                definition = ""
                try:
                    sections = line.split(" : ")
                    word = sections[0]
                    definition = sections[1]
                except:
                    word = f'"{line}" is not parsable '
                words.append(word)
                definitions.append(definition)
                print(f"{word} : {definition}")

        df = pd.DataFrame(
            {"From-Language word": words, "To-Language definition": definitions}
        )

        return input_format.InputFormat(df)
