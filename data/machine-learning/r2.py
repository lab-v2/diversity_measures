import pandas as pd
import json
import statsmodels.api as sm
import os
from sklearn.metrics import r2_score

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def calculate_cumulative_values(df, column):
    cum_min = df[column].expanding().min()
    cum_max = df[column].expanding().max()
    return cum_min, cum_max

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def process_file(file_path):
    df = load_jsonl(file_path)

    # Calculate cumulative minimum and maximum entropy
    df['cum_min_entropy'], df['cum_max_entropy'] = calculate_cumulative_values(df, 'shannon_entropy')

    # Linear regression for cumulative minimum entropy
    X_min = sm.add_constant(df['cum_min_entropy'])
    model_min = sm.OLS(df['majority_correct'], X_min).fit()
    r2_min = model_min.rsquared

    # Linear regression for cumulative maximum entropy
    X_max = sm.add_constant(df['cum_max_entropy'])
    model_max = sm.OLS(df['majority_correct'], X_max).fit()
    r2_max = model_max.rsquared

    return r2_min, r2_max

def process_directory(directory_path):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            r2_min, r2_max = process_file(file_path)
            results[filename] = {
                'r2_min': r2_min,
                'r2_max': r2_max
            }
    return results

# Directory containing the JSONL files
directory_path = 'base-T0.7'

# Process the directory and print results
results = process_directory(directory_path)

for filename, r2_values in results.items():
    print(f'File: {filename}')
    print(f'R² for cumulative minimum entropy: {r2_values["r2_min"]}')
    print(f'R² for cumulative maximum entropy: {r2_values["r2_max"]}')
    print()
