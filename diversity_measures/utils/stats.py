from typing import List
import math
from typing import List, Dict
from collections import Counter

from scipy.stats import entropy

def probability_distribution(values : List[any]) -> Dict:
    # Edge cases
    if len(values) > 0 and type(values[0]) == set: values = [frozenset(value) for value in values]

    counter = Counter(values)
    for key in counter.keys():
        counter[key] = counter[key] / len(values)
    return counter

def shannon_entropy(values : List[any], **kwargs) -> float:
    prob = probability_distribution(values).values()
    return entropy(pk=list(prob), **kwargs)
    
def gini_impurity(values: List[any]) -> float:
    prob = probability_distribution(values).values()
    return 1 - sum([math.pow(p, 2) for p in prob])