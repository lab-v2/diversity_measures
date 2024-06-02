"""
The code below is an implementation of 
Section 4.1 Diversity-Based Prompt Selection
from the paper Diversity Measures: Domain Independent Proxies for Failure in Language Model Queries

GPT-3.5 responses are collected for various few shot variations.
This is the first step only, in later steps, prompt selection is implemented.
"""

import os

import pandas
from response_collector import ResponseCollector

draw_df = pandas.read_json(
    os.path.join("data", "test-sets", "draw.jsonl"), lines=True
)
csqa_df = pandas.read_json(
    os.path.join("data", "test-sets", "csqa.jsonl"), lines=True
)
last_letters_df = pandas.read_json(
    os.path.join("data", "test-sets", "last_letters.jsonl"),
    lines=True,
)
stqa_df = pandas.read_json(
    os.path.join("data", "test-sets", "strategyQA.jsonl"), lines=True
)
svamp_df = pandas.read_json(
    os.path.join("data", "test-sets", "svamp.jsonl"), lines=True
)

draw_few_shots = pandas.read_json(
    os.path.join("data", "prompts", "draw.jsonl"), lines=True
)
csqa_few_shots = pandas.read_json(
    os.path.join("data", "prompts", "csqa.jsonl"), lines=True
)
last_letters_few_shots = pandas.read_json(
    os.path.join("data", "prompts", "last_letters.jsonl"),
    lines=True,
)
stqas_few_shots = pandas.read_json(
    os.path.join("data", "prompts", "strategyQA.jsonl"),
    lines=True,
)
svamp_few_shots = pandas.read_json(
    os.path.join("data", "prompts", "svamp.jsonl"), lines=True
)
# gsm8k_few_shots = pandas.read_json(
#     os.path.join("data", "test-sets", "gsm8k.jsonl"), lines=True
# )


def draw_prompt(question_row, few_shots):
    """A function that is used to format the prompt for the DRAW-1K dataset."""

    index = question_row["question_id"]
    text = f"""
{few_shots['prompt']}
{question_row['question']}
Answer: 
    """.strip()
    return {"id": index, "text": text}


def csqa_prompt(question_row, few_shots):
    """A function that is used to format the prompt for the CSQA dataset."""
    index = question_row["question_id"]
    text = f"""
{few_shots['prompt']}
{question_row['question']}
"""
    return {"id": index, "text": text}


def last_letters_prompt(question_row, few_shots):
    """A function that is used to format the prompt for the Last Letters dataset."""
    index = question_row["question_id"]
    text = f"""
{few_shots['prompt']}
{question_row['question']}
"""
    return {"id": index, "text": text}

def stqa_prompt(question_row, few_shots):
    """A function that is used to format the prompt for the Strategy QA dataset."""
    index = question_row["question_id"]
    text = f"""
{few_shots['prompt']}
{question_row['question']}
"""
    return {"id": index, "text": text}

def svamp_prompt(question_row, few_shots):
    """A function that is used to format the prompt for the SVAMP dataset."""

    index = question_row["question_id"]
    text = f"""
{few_shots['prompt']}
{question_row['question']}
Answer: 
    """.strip()
    return {"id": index, "text": text}


# Collect GPT-3.5 responses at various temperature settings
for setting_name, setting in [
    ("T0.7", {"temperature": 0.7}),
]:
    TIMEOUT = 10000
    RETRY_TIME = 5
    RATE_LIMIT = 10
    CONFIGS = {"model": "gpt-3.5-turbo", "n": 20, **setting}
    response_collection = ResponseCollector(
        configs=CONFIGS,
        rate_limit=RATE_LIMIT,
        timeout=TIMEOUT,
        verbose=True,
        retry_time=RETRY_TIME,
    )

    file_path = os.path.join("data", "responses", f"few_shot-{setting_name}")

    # Collect GPT-3.5 responses for each dataset.
    for name, question_list, func, few_shots in [
        # ("draw", draw_df, draw_prompt, draw_few_shots),
        # ("csqa", csqa_df, csqa_prompt, csqa_few_shots),
        # ("last_letters", last_letters_df, last_letters_prompt, last_letters_few_shots),
        ("stqa", stqa_df, stqa_prompt, stqas_few_shots),
        # ("svamp", svamp_df, svamp_prompt, svamp_few_shots),

    ]:
        # Collect GPT-3.5 responses for each few-shot variation.
        for index, row in few_shots.iterrows():
            if index > 19: break
            modified_question_list = question_list.apply(
                lambda question_row: func(question_row, row), axis=1
            ).tolist()

            out_path = f"{file_path}/{name}"
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            response_collection.start(
                question_list=modified_question_list,
                out_path=f"{out_path}/sample_{index}.jsonl",
            )

print(f"✔ Completed: Data Collection for Few-shot")
