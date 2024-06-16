import json

file = 'ARC-Easy-Test.jsonl'
path = f'../../question-set/{file}'
output = '../ARC-Easy-Test-Simplified.jsonl'

with open(path, 'r') as infile:
    lines = infile.readlines()

data = [json.loads(line.strip()) for line in lines]

# Select the first 300 entries
first_three_hundred_entries = data[:300]

batches = [first_three_hundred_entries[i:i + 10] for i in range(0, len(first_three_hundred_entries), 10)]

def determine_prompt(labels):
    if set(labels).issubset({"A", "B", "C", "D", "E"}):
        return "Answer A, B, C, D or E. At the end, say 'the answer is [put your answer here]'.\n"
    elif set(labels).issubset({"A", "B", "C", "D"}):
        return "Answer A, B, C, or D. At the end, say 'the answer is [put your answer here]'.\n"
    elif set(labels).issubset({"1", "2", "3", "4"}):
        return "Answer 1, 2, 3, or 4. At the end, say 'the answer is [put your answer here]'.\n"
    else:
        return "Answer the question by selecting the appropriate option. At the end, say 'the answer is [put your answer here]'.\n"

with open(output, 'w') as outfile:
    for batch in batches:
        prompt_lines = []
        for item in batch:
            question_stem = item['question']['stem']
            choices = item['question']['choices']
            labels = [choice['label'] for choice in choices]
            prompt = determine_prompt(labels)

            choices_text = ' '.join([f"{choice['label']}) {choice['text']}" for choice in choices])
            question_text = f"Question: {question_stem}\nChoices: {choices_text}\nAnswer: {{{item['answerKey']}}}\n"

            prompt_lines.append(f"{prompt}{question_text}\n\n\n\n\n\n")

        full_prompt = ''.join(prompt_lines).strip()
        batch_record = {"prompt": full_prompt}
        
        json.dump(batch_record, outfile)
        outfile.write('\n')

print(f"Batch conversion complete. Saved to '{output}'.")
