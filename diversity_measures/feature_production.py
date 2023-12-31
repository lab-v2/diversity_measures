from typing import List
from collections import Counter
from enum import Enum

import numpy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean

from evals.answer_extraction import extract_last_letters, extract_draw, extract_csqa
from utils.read_file import read_file
from utils import stats

class Dataset(Enum):
    LAST_LETTERS = 0
    CSQA = 1
    DRAW = 2
    
NUM_SAMPLES = 20

for DATASET in [
    Dataset.LAST_LETTERS,
    Dataset.CSQA,
    Dataset.DRAW
]:
    for VARIATION in [
        'base-T0.3',
        'base-T0.5',
        'base-T0.7',
        'base-T0.8',
        'base-T0.9'
    ]:
            
        if DATASET == Dataset.LAST_LETTERS:
            QUESTION_SET_FILE_PATH = f'data/question-set/last_letters.jsonl'
            RESPONSE_FILE_PATH = f'data/responses/{VARIATION}/last_letters/sample_0.jsonl'
            MID_FILE_PATH = f'data/machine-learning/{VARIATION}/last_letters.jsonl'
            QUESTION_SET_INDEX_NAME = 'iIndex'
            QUESTION_SET_ANSWER_NAME = 'answer'
            RESPONSE_INDEX_NAME = 'question_id'
            RESPONSE_SAMPLE_NAME = 'choices'
            EXTRACT_RESPONSE = lambda response: response['message']['content']
            ANSWER_EXTRACTION = extract_last_letters
            COMPARE_ANSWERS = lambda x, y: x.lower() == y.lower()
        if DATASET == Dataset.CSQA:
            QUESTION_SET_FILE_PATH = f'data/question-set/csqa.jsonl'
            RESPONSE_FILE_PATH = f'data/responses/{VARIATION}/csqa/sample_0.jsonl'
            MID_FILE_PATH = f'data/machine-learning/{VARIATION}/csqa.jsonl'
            QUESTION_SET_INDEX_NAME = 'id'
            QUESTION_SET_ANSWER_NAME = 'answerKey'
            RESPONSE_INDEX_NAME = 'question_id'
            RESPONSE_SAMPLE_NAME = 'choices'
            EXTRACT_RESPONSE = lambda response: response['message']['content']
            ANSWER_EXTRACTION = extract_csqa
            COMPARE_ANSWERS = lambda x, y: x.lower() == y.lower()
        if DATASET == Dataset.DRAW:
            QUESTION_SET_FILE_PATH = f'data/question-set/draw.json'
            RESPONSE_FILE_PATH = f'data/responses/{VARIATION}/draw/sample_0.jsonl'
            MID_FILE_PATH = f'data/machine-learning/{VARIATION}/draw.jsonl'
            QUESTION_SET_INDEX_NAME = 'iIndex'
            QUESTION_SET_ANSWER_NAME = 'lSolutions'
            RESPONSE_INDEX_NAME = 'question_id'
            RESPONSE_SAMPLE_NAME = 'choices'
            EXTRACT_RESPONSE = lambda response: response['message']['content']
            ANSWER_EXTRACTION = extract_draw
            COMPARE_ANSWERS = lambda response, answer: response.issubset(set(answer)) and len(response) > 0

        def get_distance(row):
            embeddings = row[EMBEDDING_NAME]
            average = row['average'][0]
            return [euclidean(e, average) for e in embeddings]

        def extract_answers(row, column):
            answers = []
            is_correct = []
            for index, element in enumerate(row[column]): 
                if index > NUM_SAMPLES: break
                response = EXTRACT_RESPONSE(element)
                # print(index, "Response:", response)
                # print(index, "Extracted Response: ", ANSWER_EXTRACTION(response))
                answers.append(ANSWER_EXTRACTION(response))
                # print(index, "Answers:", answers)
                is_correct.append(COMPARE_ANSWERS(answers[-1], row[QUESTION_SET_ANSWER_NAME]))
            row[RESPONSE_ANSWERS_NAME] = answers
            row[MAJORITY_ANSWER_LIST_CORRECT] = is_correct
            row['num_correct'] = sum(is_correct)
            return row

        def embed_answers(row):
            return model.encode(row)

        def get_average(row):
            return numpy.average(row, axis=0, keepdims=True)

        def get_majority(row):
            answers = row[RESPONSE_ANSWERS_NAME]
            if DATASET == DATASET.DRAW:
                answers = [frozenset(answer) for answer in answers] 
            counter = Counter(answers)
            distance = row['distance']
            row[MAJORITY_ANSWER_NAME] = counter.most_common()[0][0]
            for index, i in enumerate(answers):
                if i == row[MAJORITY_ANSWER_NAME]:
                    row[MAJORITY_ANSWER_DISTANCE] = distance[index]
                    break
            return row

        MODEL = 'all-MiniLM-L6-v2'
        RESPONSE_ANSWERS_NAME = 'answers'
        MAJORITY_ANSWER_NAME = 'majority_answer'
        MAJORITY_ANSWER_DISTANCE = 'majority_distance'
        MAJORITY_ANSWER_DISTANCE_SQUARED = 'majority_distance_squared'
        MAJORITY_CORRECT_NAME = 'majority_correct'

        MAJORITY_ANSWER_LIST_NAME = 'majority_answer_list'
        MAJORITY_ANSWER_LIST_DISTANCE = 'majority_distance_list'
        MAJORITY_ANSWER_LIST_CORRECT = 'majority_correct_list'

        ENTROPY_COLUMN = 'shannon_entropy'
        GINI_IMPURITY_COLUMN = 'gini_impurity'
        EMBEDDING_NAME = 'embedding'
        print(QUESTION_SET_FILE_PATH)

        model = SentenceTransformer(MODEL)

        question_set = read_file(QUESTION_SET_FILE_PATH)
        question_set = question_set[[QUESTION_SET_INDEX_NAME, QUESTION_SET_ANSWER_NAME]]
        responses = read_file(RESPONSE_FILE_PATH)

        joined = responses.set_index(RESPONSE_INDEX_NAME).join(question_set.set_index(QUESTION_SET_INDEX_NAME))

        # Reset the index and rename the 'index' column to 'question_id'
        joined.reset_index(inplace=True)
        joined.rename(columns={'index': 'question_id'}, inplace=True)
        

        joined = joined.apply(lambda row: extract_answers(row, RESPONSE_SAMPLE_NAME), axis=1)

        joined['temp'] = joined[RESPONSE_SAMPLE_NAME].apply(lambda row : [EXTRACT_RESPONSE(r) for r in row])
        joined[EMBEDDING_NAME] = joined['temp'].apply(lambda row: embed_answers(row))
        joined['average'] = joined[EMBEDDING_NAME].apply(lambda row: get_average(row))
        joined['distance'] = joined.apply(lambda row : get_distance(row), axis=1)
        joined = joined.apply(lambda row : get_majority(row), axis=1)
        joined[MAJORITY_CORRECT_NAME] = joined.apply(lambda row : COMPARE_ANSWERS(row[MAJORITY_ANSWER_NAME], row[QUESTION_SET_ANSWER_NAME]), axis=1)
        joined[ENTROPY_COLUMN] = joined[RESPONSE_ANSWERS_NAME].apply(lambda row : stats.shannon_entropy(row))
        joined[GINI_IMPURITY_COLUMN] = joined[RESPONSE_ANSWERS_NAME].apply(lambda row : stats.gini_impurity(row))
        joined[MAJORITY_ANSWER_DISTANCE_SQUARED] = joined[MAJORITY_ANSWER_DISTANCE].apply(lambda row : row * row)
        joined = joined[[RESPONSE_INDEX_NAME, MAJORITY_ANSWER_DISTANCE, MAJORITY_ANSWER_DISTANCE_SQUARED, ENTROPY_COLUMN, GINI_IMPURITY_COLUMN, MAJORITY_CORRECT_NAME, 'num_correct']]
        joined[[ RESPONSE_INDEX_NAME, MAJORITY_ANSWER_DISTANCE, MAJORITY_ANSWER_DISTANCE_SQUARED, ENTROPY_COLUMN, GINI_IMPURITY_COLUMN, MAJORITY_CORRECT_NAME, 'num_correct']].to_json(MID_FILE_PATH,lines=True, orient='records')
