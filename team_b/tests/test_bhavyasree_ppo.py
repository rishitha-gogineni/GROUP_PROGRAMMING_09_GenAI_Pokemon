import torch

from team_b.src.models.bhavyasree_ppo.action_selection import choose_action
from team_b.src.models.bhavyasree_ppo.state_encoder import encode_options


class EqualLogitModel(torch.nn.Module):
    def forward(self, state, option_vectors):
        batch_size = state.shape[0]
        number_of_options = option_vectors.shape[1]

        logits = torch.zeros(
            batch_size,
            number_of_options,
        )

        values = torch.zeros(batch_size)

        return logits, values


def test_two_of_two_options_have_probability_one():
    observation = {
        "select": {
            "option": [
                {"type": 1},
                {"type": 2},
            ],
            "minCount": 2,
            "maxCount": 2,
        }
    }

    response, log_probability, value, logits = choose_action(
        EqualLogitModel(),
        observation,
        deterministic=False,
    )

    assert response == [0, 1]
    assert torch.isclose(
        torch.exp(log_probability),
        torch.tensor(1.0),
    )
    assert torch.isclose(
        log_probability,
        torch.tensor(0.0),
    )


def test_yes_and_no_are_distinguished():
    yes_observation = {
        "select": {
            "option": [{"type": "Yes"}],
        }
    }

    no_observation = {
        "select": {
            "option": [{"type": "No"}],
        }
    }

    yes_vector = encode_options(yes_observation)[0]
    no_vector = encode_options(no_observation)[0]

    assert yes_vector[0] != no_vector[0]


def test_yes_and_no_remain_distinguishable_when_positions_are_swapped():
    yes_no_observation = {
        "select": {
            "option": [
                {"type": "Yes"},
                {"type": "No"},
            ],
        }
    }

    no_yes_observation = {
        "select": {
            "option": [
                {"type": "No"},
                {"type": "Yes"},
            ],
        }
    }

    yes_no_vectors = encode_options(yes_no_observation)
    no_yes_vectors = encode_options(no_yes_observation)

    assert yes_no_vectors[0][0] != no_yes_vectors[0][0]
    assert yes_no_vectors[1][0] != no_yes_vectors[1][0]
