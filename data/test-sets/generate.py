import json

file = 'strategyQA.json'
path = f'../question-set/{file}'
output = 'strategyQA.jsonl'

# Load your JSON data
with open(path, 'r') as file:
    data = json.load(file)

# Prepare a list to store the new simplified records
# Select the last 100 entries directly
simplified_records = []
last_hundred_entries = data[-100:]  # Get the last 100 entries

# Iterate over the last 100 loaded data
for item in last_hundred_entries:
    # Extract the desired fields
    new_record = {
        "question_id": item["qid"],
        "question": item["question"],
        "answer": item["answer"]
    }
    simplified_records.append(new_record)

# Write the new records to a JSONL file
with open(output, 'w') as outfile:
    for record in simplified_records:
        json.dump(record, outfile)
        outfile.write('\n')  # Write a newline to separate entries in JSONL format

print(F"Conversion complete. Saved to '{output}'.")
