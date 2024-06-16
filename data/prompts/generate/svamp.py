import json

file = 'svamp.json'
path = f'../../question-set/{file}'
output = '../svamp.jsonl'

# Load your JSON data
with open(path, 'r') as file:
    data = json.load(file)

# Select the first 300 entries
first_three_hundred_entries = data[:300]

# Prepare to batch these entries into sets of 10
batches = [first_three_hundred_entries[i:i + 10] for i in range(0, len(first_three_hundred_entries), 10)]

with open(output, 'w') as outfile:
    for batch in batches:
        prompt_lines = []
        for item in batch:
            combined_question = f"{item['Body']} {item['Question']}"

            question_line = (
                # "Answer yes or no. At the end, say 'the answer is [put your answer here]'.\n"
                f"Question: {combined_question} \n"
                f"Equation: {item['Equation']}\n"
                f"Answer: {str(item['Answer']).lower()}\n"
            )
            prompt_lines.append(question_line)

        full_prompt = "\n".join(prompt_lines)
        batch_record = {"prompt": full_prompt}
        
        json.dump(batch_record, outfile)
        outfile.write('\n')

print(F"Batch conversion complete. Saved to '{output}'.")