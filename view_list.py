import json
import pandas as pd

df = pd.DataFrame(
    json.load(open("top1000_distant_closest.json")),
    columns=["vietnamese", "closest_english", "cosine"],
)
print(df.head(20))
