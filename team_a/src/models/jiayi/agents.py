from pathlib import Path
import shutil

from ... import features, heuristic
from ...common_runtime import stage_agent


POLICIES = {"A-J1": ("aj1", "aj1_bc.json"), "A-J2": ("aj2", "aj2_bc.json")}


def stage_policy(model_id, destination, deck_file):
    folder = Path(__file__).resolve().parent
    entry, artifact = POLICIES[model_id]
    metadata = stage_agent({"label": model_id, "entrypoint": folder / entry / "main.py",
                            "assets": []}, destination, deck_file)
    shutil.copyfile(folder / artifact, Path(destination) / "model.json")
    metadata["model_artifact"] = artifact
    return metadata


def choose_action(model, observation):
    if not features.supports_observation(observation):
        return heuristic.agent(observation)
    encoded = features.encode_observation(observation)
    scores = model.score_options(encoded["state"], encoded["options"])
    best = max(range(len(scores)), key=scores.__getitem__)
    return [encoded["option_indices"][best]]
