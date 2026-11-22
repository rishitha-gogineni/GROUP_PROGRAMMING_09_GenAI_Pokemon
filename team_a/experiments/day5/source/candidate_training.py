"""Shared candidate objective helpers from the executed A-R1 run."""
import json
import numpy as np

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def pack(examples):
    sizes = np.array([len(example['x']) for example in examples], dtype=np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return (np.concatenate([example['x'] for example in examples]),
            np.concatenate([example['target'] for example in examples]), sizes, starts)

def distribution(scores, sizes, starts):
    maximum = np.maximum.reduceat(scores, starts)
    shifted = scores - np.repeat(maximum, sizes)
    exponentials = np.exp(shifted)
    totals = np.add.reduceat(exponentials, starts)
    probabilities = exponentials / np.repeat(totals, sizes)
    log_probabilities = shifted - np.repeat(np.log(totals), sizes)
    return probabilities, log_probabilities

def linear_metrics(examples, weights):
    x, target, sizes, starts = pack(examples)
    scores = x @ weights
    _, logp = distribution(scores, sizes, starts)
    hits, exact = [], []
    for index, (start, size) in enumerate(zip(starts, sizes)):
        predicted = int(np.argmax(scores[start:start+size]))
        hits.append(bool(examples[index]['target'][predicted] > 0))
        exact.append(predicted == examples[index]['chosen'])
    return {'decisions': len(examples), 'cross_entropy': float(-target @ logp / len(examples)),
            'teacher_top_set_accuracy': float(np.mean(hits)), 'exact_choice_accuracy': float(np.mean(exact)),
            'uniform_choice_top_set_accuracy': float(np.mean([np.count_nonzero(e['target']) / len(e['target']) for e in examples])),
            'uniform_cross_entropy': float(np.log(sizes).mean())}
