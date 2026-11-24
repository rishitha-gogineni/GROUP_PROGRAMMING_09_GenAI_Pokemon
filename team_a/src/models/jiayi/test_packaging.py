import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from ... import features
from .model_1 import Model as Linear
from .model_2 import Model as MLP
from .package_submissions import COMMON_FILES, POLICIES, build_submission


ROOT = Path(__file__).resolve().parents[4]


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "team_a/experiments/day4/training/train.jsonl").open() as handle:
            cls.observation = json.loads(next(handle))["observation"]

    def test_archives_are_minimal_and_run_without_repository_access(self):
        loaders = {"A-J1": Linear, "A-J2": MLP}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deck = root / "official_deck.csv"
            deck.write_text("\n".join(str(value) for value in range(1, 61)) + "\n")
            for model_id, loader in loaders.items():
                with self.subTest(model=model_id):
                    archive = build_submission(model_id, deck, root / "archives")
                    extracted = root / (model_id + "_extracted")
                    extracted.mkdir()
                    with tarfile.open(archive, "r:gz") as handle:
                        members = handle.getmembers()
                        self.assertTrue(all(member.isfile() and not member.issym() and not member.islnk()
                                            for member in members))
                        self.assertFalse(any(Path(member.name).is_absolute() or ".." in Path(member.name).parts
                                             for member in members))
                        handle.extractall(extracted)
                    expected = {"main.py", "model.json", "deck.csv", POLICIES[model_id]["loader"],
                                *COMMON_FILES}
                    self.assertEqual({member.name for member in members}, expected)
                    self.assertEqual([int(value) for value in (extracted / "deck.csv").read_text().splitlines()],
                                     list(range(1, 61)))
                    encoded = features.encode_observation(self.observation)
                    scores = loader(extracted / "model.json").score_options(encoded["state"], encoded["options"])
                    expected_action = [encoded["option_indices"][max(range(len(scores)), key=scores.__getitem__)]]
                    self.run_isolated_contract(extracted, self.observation, expected_action)

    def run_isolated_contract(self, extracted, observation, expected_action):
        fallback = copy.deepcopy(observation)
        fallback["select"] = {"type": 8, "context": 0, "minCount": 2, "maxCount": 2,
                              "option": [{}, {}, {}]}
        script = '''import inspect, json, os, sys
from pathlib import Path
package = Path(sys.argv[1]).resolve()
os.chdir(package)
sys.path[:] = [str(package)] + json.loads(sys.argv[2])
namespace = {}
path = package / "main.py"
exec(compile(path.read_text(), str(path), "exec"), namespace)
assert "__file__" not in namespace
callables = [value for value in namespace.values() if callable(value)]
agent = namespace.get("agent")
assert callable(agent) and callables[-1] is agent
configuration = {"__raw_path__": str(path), "seed": 91000}
assert inspect.signature(agent).bind({}, configuration)
assert agent({"select": None}, configuration) == list(range(1, 61))
payload = json.loads(sys.stdin.read())
assert agent(payload["observation"], configuration) == payload["expected"]
fallback = agent(payload["fallback"], configuration)
selection = payload["fallback"]["select"]
assert isinstance(fallback, list) and len(fallback) == selection["maxCount"]
assert len(set(fallback)) == len(fallback)
assert all(type(index) is int and 0 <= index < len(selection["option"]) for index in fallback)
'''
        standard_library = [path for path in sys.path if "site-packages" not in path and str(ROOT) not in path]
        payload = json.dumps({"observation": observation, "expected": expected_action,
                              "fallback": fallback})
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(extracted),
                                 json.dumps(standard_library)], input=payload, text=True,
                                cwd=extracted.parent, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
