from config import *
from dataset_loader import *


df, rows = load_json_annotations(JSON_FOLDER)

num_rust = 0
num_scratch = 0
num_rivet_damage = 0
num_paint_peel = 0

for row in rows:
    print(f"ID: {row['id']}, Target: {row['target']}))")

    if row['target'] == 'rust':
        num_rust += 1
    if row['target'] == 'scratch':
        num_scratch += 1
    if row['target'] == 'rivet_damage':
        num_rivet_damage += 1
    if row['target'] == 'paint_peel':
        num_paint_peel += 1

print(rows.__len__(), "rows loaded from JSON annotations.")

print(f"Number of rust annotations: {num_rust}")
print(f"Number of scratch annotations: {num_scratch}")
print(f"Number of rivet damage annotations: {num_rivet_damage}")
print(f"Number of paint peel annotations: {num_paint_peel}")

