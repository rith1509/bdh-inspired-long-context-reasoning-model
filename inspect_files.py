import pandas as pd
import os

# 1. Inspect CSVs
csv_files = ["train.csv", "test.csv"]
for f in csv_files:
    if os.path.exists(f):
        print(f"\n=== INSPECTING {f} ===")
        try:
            df = pd.read_csv(f, nrows=3) # Read only top 3 rows
            print("Columns:", list(df.columns))
            print("First row sample:")
            print(df.iloc[0])
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"File not found: {f}")

# 2. Inspect Text Files (Check for weird headers)
txt_files = ["The Count of Monte Cristo.txt", "In search of the castaways.txt"]
for f in txt_files:
    if os.path.exists(f):
        print(f"\n=== INSPECTING {f} ===")
        try:
            with open(f, 'r', encoding='utf-8') as file:
                print(f"First 200 chars:\n{file.read(200)}...")
        except Exception as e:
            print(f"Error reading text file: {e}")