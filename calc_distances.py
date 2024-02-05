import pandas as pd
import numpy as np
from rapidfuzz.distance.Levenshtein import normalized_similarity 

csv_filename = "Resultados individuais.csv"
csv_mod_filename = "Resultados individuais MOD.csv"

df = pd.read_csv(csv_filename)
queries_reference = df["PQL (reference)"]
queries_generated = df["PQL (generated)"]

similarities = []
for reference, generated in zip(queries_reference, queries_generated):
    similarity = 10*normalized_similarity(reference, generated)
    similarities.append(similarity)
df["Similarity"] = similarities
df.to_csv(csv_mod_filename, sep=",", index=False)
