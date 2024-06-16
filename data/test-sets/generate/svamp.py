import json

file = 'svamp.json'
path = f'../../question-set/{file}'
output = '../svamp.jsonl'

with open(path, 'r') as file:
    data = json.load(file)

simplified_records = []
last_hundred_entries = data[-100:]  # Get the last 100 entries

prompt = "Solve the following math question. At the end, say 'the answer is [put your numbers here separated by commas]'.\n"

# Iterate over the last 100 loaded data
for item in last_hundred_entries:
    combined_question = f"{item['Body']} {item['Question']}"

    new_record = {
        "question_id": item["ID"],
        "question": f"{prompt} {combined_question}",
        "answer": item["Answer"]
    }
    simplified_records.append(new_record)

with open(output, 'w') as outfile:
    for record in simplified_records:
        json.dump(record, outfile)
        outfile.write('\n')  
print(F"Conversion complete. Saved to '{output}'.")
