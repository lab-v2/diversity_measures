import re

from evals.number_extraction import NumberExtractor

# =======================================
# This class is concerned with evaluating
# ChatGPT. It contains functions used to
# evaluate very specific datasets
# =======================================
def last_letters(response: str, answer: str):
    response = extract_last_letters(response)
    return response == answer

def extract_last_letters(response: str):
    response = response.lower()
    response = extract_answer(response)
    response = re.findall(r'[a-zA-Z]+', response)

    if len(response) == 0: return ''
    return response[0]


def extract_draw(response: str):
    response = response.lower()
    response = extract_answer(response)
    response = NumberExtractor.extract_decimals_from_text(response)

    responses = []
    for r in response:
        try: responses.append(eval(r))
        except: pass
    
    return set(responses)


def extract_csqa(response: str):
    response = response.lower()
    response = extract_answer(response)
    response = re.findall(r'a|b|c|d|e', response, flags=re.IGNORECASE)

    if len(response) == 0: return ''
    return response[0]

def extract_answer(response: str) -> str:
    response = __clean_response(response)

    extracted_answer = re.findall(r'the answer is.*', response, flags=re.IGNORECASE)
    if len(extracted_answer) == 0: return ''

    extracted_answer = extracted_answer[0]
    extracted_answer = re.sub(r'the answer is', '', extracted_answer, count=1, flags=re.IGNORECASE)

    if len(re.findall(r'the answer is', extracted_answer, flags=re.IGNORECASE)) > 0: return extract_answer(extracted_answer)
    return extracted_answer.strip()

def __clean_response(response: str) -> str:
    response = re.sub(r' +', ' ', response)
    return response
