import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "src": row["transliteration"].lower().strip(),
            "trg": row["translation"].lower().strip()
        })
    return data


def tokenize(text):
    return text.split()