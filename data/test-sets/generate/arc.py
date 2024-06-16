import json

file = 'ARC-Easy-Test.jsonl'
path = f'../../question-set/{file}'
output = '../ARC-Easy-Test-Simplified.jsonl'

# Read all lines from the input file
with open(path, 'r') as infile:
    lines = infile.readlines()

# Get the last 100 entries
last_hundred_entries = lines[-100:]

simplified_records = []

# Process the last 100 entries
for line in last_hundred_entries:
    item = json.loads(line.strip())
    question_stem = item['question']['stem']
    choices = item['question']['choices']
    labels = [choice['label'] for choice in choices]
    
    # Determine the prompt based on the existing labels
    if set(labels).issubset({"A", "B", "C", "D", "E"}):
        prompt = "Answer A, B, C, D or E. At the end, say 'the answer is [put your answer here]'.\n"
    elif set(labels).issubset({"A", "B", "C", "D"}):
        prompt = "Answer A, B, C, or D. At the end, say 'the answer is [put your answer here]'.\n"
    elif set(labels) <= {"1", "2", "3", "4"}:
        prompt = "Answer 1, 2, 3, 4. At the end, say 'the answer is [put your answer here]'.\n"
    else:
        prompt = "Answer the question by selecting the appropriate option. At the end, say 'the answer is [put your answer here]'.\n"

    choices_text = ' '.join([f"{choice['label']}. {choice['text']}" for choice in choices])
    combined_question = f"{question_stem} {choices_text}"

    new_record = {
        "question_id": item["id"],
        "question": f"{prompt} {combined_question}",
        "answer": item["answerKey"]
    }
    simplified_records.append(new_record)

# Write the simplified records to the output file
with open(output, 'w') as outfile:
    for record in simplified_records:
        json.dump(record, outfile)
        outfile.write('\n')

print(f"Conversion complete. Saved to '{output}'.")
