import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
# from rouge import Rouge


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):

    def remove_newline(text):
        return text.split('\n\n')[0]
    
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_newline(
        remove_articles(remove_punc(lower(s)))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def calculate_acc(prediction, ground_truth):
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1
    else:
        return 0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


def eval_question_answering(inputs):

    tokenizer = SimpleTokenizer()
    
    if type(inputs) == list:
        inputlines = inputs
    else: # type(input) == str -- file path
        infile = open(inputs, 'r')
        inputlines = json.load(infile)
        infile.close()
    
    exact_match_count = 0
    has_answer_count = 0
    f1_scores = 0
    acc_scores = 0
    answer_lengths = []
    
    for _, line in enumerate(inputlines):

        answers = line['answer']
        output = line['model_output']
        if "the answer is" not in output:
            output = output.split(' || ')[0]
        output = output.split('the answer is')[-1]
        has_answer_count += has_answer(answers, output, tokenizer)
        exact_match_count += ems(output, answers)
        f1_scores += f1(output, answers)
        acc_scores += calculate_acc(output, answers)

        answer_lengths.append(len(output.split()))


    recall = round(has_answer_count/len(inputlines), 4)
    em = round(exact_match_count/len(inputlines), 4)
    f1_score = round(f1_scores/len(inputlines), 4)
    acc_score = round(acc_scores/len(inputlines), 4)
    avg_lens = round(np.mean(answer_lengths), 4)
    

    return recall, em, f1_score, acc_score,avg_lens
