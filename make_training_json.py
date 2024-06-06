import pandas as pd
import json
import os

# Read the Excel file
excel_file = 'FineTuning_RawDataSample.xlsx'  # Replace with your Excel file path
df = pd.read_excel(excel_file)

# Calculate the split indices
train_idx = int(len(df) * 0.90)
test_idx = train_idx + int(len(df) * 0.05)

# Split the DataFrame
df_train = df.iloc[:train_idx]
df_test = df.iloc[train_idx:test_idx]
df_valid = df.iloc[test_idx:]

columns = ["responsibilities", "skills", "experience"]

# Function to write DataFrame to a .jsonl file
def write_to_jsonl(dataframe, filename):
    with open(os.path.join('data', filename), 'w') as file:
        for index, row in dataframe.iterrows():
            for column in columns:
                description_cleaned = row['description_cleaned']
                answer = row[column]
                if pd.notnull(answer):
                    formatted_str = f"<s>[INST]What is or are the {column} listed in the following job description?\n{description_cleaned}\n[/INST]{column}/n{answer}</s>"
                    json_line = json.dumps({"text": formatted_str})
                    file.write(json_line + '\n')

# Write each subset to the corresponding file
write_to_jsonl(df_train, 'data/train.jsonl')
write_to_jsonl(df_test, 'data/test.jsonl')
write_to_jsonl(df_valid, 'data/valid.jsonl')