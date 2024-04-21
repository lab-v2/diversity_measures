"""
The code below is an implementation of 
Section 5 Evaluation
from the paper Diversity Measures: Domain Independent Proxies for Failure in Language Model Queries

GPT-3.5 responses are collected for standard prompting.
"""

import os

import pandas
from response_collector import ResponseCollector


def draw_prompt(row):
    text = f"Solve the following math question. At the end, say 'the answer is [put your numbers here separated by commas]'\nQuestion: {row['sQuestion']}"
    index = row["iIndex"]
    return {"text": text, "id": index}


def csqa_prompt(row):
    question = row["question"]["stem"]
    choices = row["question"]["choices"]
    index = row["id"]
    choices = [f"{choice['label']}) {choice['text']}" for choice in choices]
    choices = "\n".join(choices)
    text = f"Answer A, B, C, D or E. At the end, say 'the answer is [put your answer here]'.\nQuestion: {question}\nChoices: {choices}"
    return {"text": text, "id": index}


def last_letters_prompt(row):
    text = f"At the end, say 'the answer is [put the concatenated word here]'.\nQuestion: {row['question']}.\n "
    # text = row['question']
    index = row["iIndex"]
    return {"text": text, "id": index}

def stqa_prompt(row):
    text = f"Answer yes or no. At the end, say 'the answer is [put your answer here]'.\nQuestion: {row['question']}.\n "
    # text = row['question']
    index = row["qid"]
    return {"text": text, "id": index}


draw_df = pandas.read_json(
    os.path.join("data", "question-set", "draw.json"), lines=False
)
csqa_df = pandas.read_json(
    os.path.join("data", "question-set", "csqa.jsonl"), lines=True
)
last_letters_df = pandas.read_json(
    os.path.join("data", "question-set", "last_letters.jsonl"),
    lines=True,
)
stqa_df = pandas.read_json(
    os.path.join("data", "question-set", "strategyQA.json"), lines=False
)


draw_df = draw_df.apply(lambda row: draw_prompt(row), axis=1).to_list()
csqa_df = csqa_df.apply(lambda row: csqa_prompt(row), axis=1).to_list()
last_letters_df = last_letters_df.apply(lambda row: last_letters_prompt(row), axis=1).to_list()
stqa_df = stqa_df.apply(lambda row: stqa_prompt(row), axis=1).to_list()


for setting_name, setting in [
    ("T0.3", {"temperature": 0.3}),
    ("T0.5", {"temperature": 0.5}),
    ("T0.7", {"temperature": 0.7}),
    ("T0.8", {"temperature": 0.8}),
    ("T0.9", {"temperature": 0.9}),
]:
    TIMEOUT = 10000
    RETRY_TIME = 5
    RATE_LIMIT = 300
    CONFIGS = {"model": "gpt-3.5-turbo", "n": 20, **setting}
    sleepyask = ResponseCollector(
        configs=CONFIGS,
        rate_limit=RATE_LIMIT,
        timeout=TIMEOUT,
        verbose=True,
        retry_time=RETRY_TIME,
    )

    file_path = os.path.join("data", "responses", f"base-{setting_name}")

    for name, question_list in [
        # ("draw", draw_df),
        # ("csqa", csqa_df),
        # ("last_letters", last_letters_df),
        ("stqa", stqa_df)
    ]:
        text = f"Asking {name}"
        text = text + " " * (20 - len(text))

        out_path = f"{file_path}/{name}"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        sleepyask.start(
            question_list=question_list, out_path=f"{out_path}/sample_0.jsonl"
        )

print(f"✔ Completed: Data Collection for Standard Prompting")
