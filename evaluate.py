
import numpy as np
import time
import random
random.seed(13)
from nltk.tokenize import word_tokenize

### distinct
def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i: i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores

if __name__ == "__main__":
    
    ### distinct
    f = open("", mode="r").readlines()
    data = []
    for i in f:
        if i != "":
            data.append(i.split('#*#*#')[1].strip())
        # data.append(i.split('#*#*#')[1]).strip()
        # data.append(i.strip())
    print(len(data))

    calc_distinct(data)
