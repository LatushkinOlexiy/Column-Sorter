from sentence_transformers import SentenceTransformer as st, util
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import math
import tkinter as tk
from tkinter import filedialog, Tk

model = st("all-MiniLM-L6-v2")
path = "C:\\Users\\admin\\Desktop\\Hackaton\\datasets\\demo\\"
all_files = [f for f in listdir(path) if isfile(join(path, f))]
column_list = []
datasets = []
df_names = []
root = Tk()
root.withdraw()

#make better fit for strings of column names
def normalize(col):
    col = col.lower()
    col = col.replace("_", " ")
    return col.strip()

#creates canonical dictionary for the model
def generate_canonical(df):
    canonical = {}
    for col in df.columns:
        canonical[col] = normalize(col)
    return canonical

#check if selected column name is gene to skip model processing (it confuses short column names like "exp" for expression for gene names)
def is_gene_symbol(col):
    if re.fullmatch(r"[A-Z]{2,6}\d*", col):
        return True
    if re.fullmatch(r"[A-Z][a-z]{2,5}\d*", col):
        return True
    if sum(char.isdigit() for char in col)>5:
        return True
    return False
#working with model
def build_canonical_vectors(canonical):
    return {k: model.encode(v) for k, v in canonical.items()}
#check which column from canonical dataset is most fitting
def match_column_embedding(col, canonical_vecs):
    vec = model.encode(normalize(col))

    best = None
    best_score = -1

    if is_gene_symbol(col):
        if col not in canonical:
            canonical[col] = normalize(col)
            canonical_vecs[col] = vec 
            print(f"new gene added: {col}")

    for canon_col, canon_vec in canonical_vecs.items():
       
        if is_gene_symbol(canon_col):
            if canon_col == col:
                best_score = math.inf
                best = canon_col
        else:
            score = util.cos_sim(vec, canon_vec).item()
            if score > best_score:
                best_score = score
                best = canon_col
        
    return best, best_score


#filling variables for further processing
canonical = generate_canonical(datasets[0])
canonical_vecs = build_canonical_vectors(canonical)

for dataset in all_files:
  df_names.append(dataset)
  new_dataset = pd.read_csv(path+dataset,sep='\t')
  datasets.append(new_dataset)
  new_columns = new_dataset.columns
  column_list.append(new_columns)
#print column names pre-processing
print("old columns: ")
for c in column_list:
    print(c)

#creates loop for renaming the columns of selected datasets and update canonical column list
for i, df in enumerate(datasets[1:], start=1):
    rename_map = {}

    for col in df.columns:
        best, score = match_column_embedding(col, canonical_vecs)
        rename_map[col] = best

    df.rename(columns=rename_map, inplace=True)
#creates loop for sorting columns according to standardized column list
for i, df in enumerate(datasets[1:], start=1):
    df = df.reindex(columns=canonical)
    datasets[i] = df
#prints column names after processing
print("new columns: ")
for c in datasets:
    print(c.columns)

#saves processed dataset in a directory
output_path = ""
output_path = filedialog.askdirectory()

for i, df in enumerate(datasets[1:],start=1):
    save_path = f"{output_path}/{df_names[i]}"
    print(save_path)
    df.to_csv(save_path,sep='\t')
