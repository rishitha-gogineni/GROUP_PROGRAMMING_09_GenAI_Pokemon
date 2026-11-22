"""Fitted A-R2 MLP for the shared ar1-visible-v1 candidate feature vectors."""

import json
import math
from pathlib import Path

from ...contracts import FeatureVector
from .ar2 import features

MODEL_ID = 'A-R2'
ARCHITECTURE = 'Small MLP imitation scorer'


class Model:
    """Load the recorded tanh network without a numerical library dependency."""

    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else Path(__file__).with_name('ar2') / 'model.json'
        model = json.loads(path.read_text(encoding='utf-8'))
        if model['model_id'] != MODEL_ID or model['schema_version'] != features.SCHEMA_VERSION:
            raise ValueError('A-R2 model schema mismatch.')
        if model['feature_names'] != list(features.FEATURE_NAMES) or model['activation'] != 'tanh':
            raise ValueError('A-R2 feature or activation mismatch.')
        size, hidden = len(features.FEATURE_NAMES), model['hidden_units']
        if type(hidden) is not int or hidden <= 0:
            raise ValueError('Invalid A-R2 hidden layer size.')
        if len(model['mean']) != size or len(model['scale']) != size or len(model['W1']) != size:
            raise ValueError('A-R2 input dimension mismatch.')
        if any(len(row) != hidden for row in model['W1']) or len(model['b1']) != hidden or len(model['w2']) != hidden:
            raise ValueError('A-R2 hidden dimension mismatch.')
        if not all(math.isfinite(value) for key in ['mean', 'scale', 'b1', 'w2'] for value in model[key]):
            raise ValueError('Nonfinite A-R2 parameters.')
        if not all(math.isfinite(value) for row in model['W1'] for value in row) or min(model['scale']) <= 0:
            raise ValueError('Invalid A-R2 weights or scales.')
        self.parameters = model
        self.units = []
        for j in range(hidden):
            terms = [(i, model['W1'][i][j] / model['scale'][i])
                     for i in range(size) if model['W1'][i][j] != 0.0]
            bias = model['b1'][j] - sum(model['mean'][i] * weight for i, weight in terms)
            self.units.append((bias, terms))

    def score_options(self, state: FeatureVector, options: list[FeatureVector]) -> list[float]:
        """Use an empty state; complete candidate vectors include state interactions."""
        if len(state) != 0:
            raise ValueError('A-R2 includes state interactions in each candidate; pass an empty state.')
        result = []
        for option in options:
            if len(option) != len(features.FEATURE_NAMES) or not all(math.isfinite(value) for value in option):
                raise ValueError('Invalid A-R2 candidate vector.')
            hidden = [math.tanh(bias + sum(option[i] * weight for i, weight in terms))
                      for bias, terms in self.units]
            score = sum(value * weight for value, weight in zip(hidden, self.parameters['w2']))
            if not math.isfinite(score):
                raise ValueError('Nonfinite A-R2 score.')
            result.append(score)
        return result

    def score_observation(self, observation) -> list[float]:
        """Score supported MAIN observations in their current legal option order."""
        return self.score_options([], features.candidate_features(observation))
