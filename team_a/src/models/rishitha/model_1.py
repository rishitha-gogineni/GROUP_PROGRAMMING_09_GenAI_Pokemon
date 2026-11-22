"""A-R1 scorer for the recorded ar1-visible-v1 candidate feature vectors."""

import json
import math
from pathlib import Path

from ...contracts import FeatureVector
from .ar1 import features

MODEL_ID = 'A-R1'
ARCHITECTURE = 'Linear contextual imitation scorer'


class Model:
    """Load fitted weights. State interactions are already in each candidate vector."""

    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else Path(__file__).with_name('ar1') / 'model.json'
        self.parameters = json.loads(path.read_text(encoding='utf-8'))
        if self.parameters['schema_version'] != features.SCHEMA_VERSION:
            raise ValueError('A-R1 feature schema mismatch.')
        if self.parameters['feature_names'] != list(features.FEATURE_NAMES):
            raise ValueError('A-R1 feature names or order do not match.')
        size = len(features.FEATURE_NAMES)
        for key in ('weights', 'mean', 'scale'):
            values = self.parameters[key]
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError('Invalid A-R1 parameters.')
        if min(self.parameters['scale']) <= 0:
            raise ValueError('A-R1 scales must be positive.')

    def score_options(self, state: FeatureVector, options: list[FeatureVector]) -> list[float]:
        """Pass an empty state and complete candidate vectors from candidate_features."""
        if len(state) != 0:
            raise ValueError('A-R1 includes state interactions in each option vector; pass an empty state.')
        parameters = self.parameters
        result = []
        for option in options:
            if len(option) != len(features.FEATURE_NAMES) or not all(math.isfinite(value) for value in option):
                raise ValueError('Invalid A-R1 candidate vector.')
            score = sum((value - mean) / scale * weight for value, mean, scale, weight in
                        zip(option, parameters['mean'], parameters['scale'], parameters['weights']))
            if not math.isfinite(score):
                raise ValueError('Nonfinite A-R1 score.')
            result.append(score)
        return result

    def score_observation(self, observation) -> list[float]:
        """Score a supported MAIN observation in its current legal option order."""
        return self.score_options([], features.candidate_features(observation))
