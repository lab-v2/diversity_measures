import json

file = 'strategyQA.json'
path = f'../question-set/{file}'
output = 'strategyQA.jsonl'

# Load your JSON data
with open(path, 'r') as file:
    data = json.load(file)

# Select the first 300 entries
first_three_hundred_entries = data[:300]

# Prepare to batch these entries into sets of 10
batches = [first_three_hundred_entries[i:i + 10] for i in range(0, len(first_three_hundred_entries), 10)]

# Prepare to write these records to a JSONL file
with open(output, 'w') as outfile:
    for batch in batches:
        # Initialize a prompt structure for each batch
        prompt_lines = []
        for item in batch:
            # Format each question with its own prompt
            question_line = (
                "Answer yes or no. At the end, say 'the answer is [put your answer here]'.\n"
                f"Question: {item['question']} \n"
                f"Facts: {'; '.join(item['facts'])}\n"
                f"Answer: {str(item['answer']).lower()}\n"
            )
            prompt_lines.append(question_line)

        # Join all question lines into a single prompt for the batch
        full_prompt = "\n".join(prompt_lines)
        batch_record = {"prompt": full_prompt}
        
        # Dump the batch record as a JSON object and write a newline to separate entries
        json.dump(batch_record, outfile)
        outfile.write('\n')

print(F"Batch conversion complete. Saved to '{output}'.")