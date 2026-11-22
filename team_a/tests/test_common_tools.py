"""Standard-library tests. Synthetic SDK fixtures never load the native engine."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from team_a.src import features
from team_a.src.common_runtime import DEFAULT_CONFIG, load_config, stage_agent
from team_a.src.models.rishitha.ar1 import features as frozen_features
from team_a.src.models.rishitha.model_1 import Model as Linear
from team_a.src.models.rishitha.model_2 import Model as MLP


class FeatureTests(unittest.TestCase):
    def test_frozen_features_and_fitted_model_interfaces(self):
        rows = (REPO / 'team_a/experiments/day4/training/train.jsonl').read_text().splitlines()[:3]
        models = [Linear(), MLP()]
        for line in rows:
            observation = json.loads(line)['observation']
            encoded = features.encode_observation(observation)
            self.assertEqual(encoded['schema_version'], 'ar1-visible-v1')
            self.assertEqual(encoded['state'], [])
            self.assertEqual(encoded['options'], frozen_features.candidate_features(observation))
            self.assertEqual(encoded['option_indices'], list(range(len(observation['select']['option']))))
            self.assertEqual(len(features.FEATURE_NAMES), 375)
            self.assertEqual(len(features.describe_state(observation)), 21)
            for model in models:
                actual = model.score_options(encoded['state'], encoded['options'])
                expected = model.score_observation(observation)
                self.assertTrue(all(math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10) for a, b in zip(actual, expected)))

    def test_unsupported_decisions_are_explicit(self):
        for observation in [None, {}, {'select': None}, {'current': {}, 'select': {}},
                            {'current': {'players': [None, None], 'yourIndex': 0}, 'select': {'type': 0, 'option': [{}, {}], 'minCount': 1, 'maxCount': 1}}]:
            self.assertFalse(features.supports_observation(observation))
            with self.assertRaises(ValueError):
                features.encode_options(observation)
        original = json.loads((REPO / 'team_a/experiments/day4/training/train.jsonl').read_text().splitlines()[0])['observation']
        for selection in [{'type': 1}, {'maxCount': 2}, {'option': [{}]}]:
            changed = copy.deepcopy(original)
            changed['select'].update(selection)
            self.assertFalse(features.supports_observation(changed))


FAKE_GAME = '''from types import SimpleNamespace
from .sim import Battle
actor = 0
def observation(result):
    return {'current': {'result': result, 'yourIndex': actor, 'players': [{}, {}]},
            'select': {'type': 0, 'context': 0, 'minCount': 1, 'maxCount': 1,
                       'option': [{'type': 14}, {'type': 13, 'attackId': 1}]}, 'logs': []}
def battle_start(deck0, deck1, reverse_player=False):
    global actor
    actor = int(reverse_player)
    Battle.battle_ptr = 1
    return observation(-1), SimpleNamespace(errorPlayer=None, errorType=None)
def battle_select(action):
    return observation(actor if action[0] == 1 else 1 - actor)
def battle_finish():
    Battle.battle_ptr = None
'''

AGENT = '''import json
import os
def agent(observation, configuration=None):
    assert '__file__' not in globals()
    folder = os.path.dirname(configuration['__raw_path__'])
    assert os.getcwd() != folder
    if observation.get('select') is None:
        with open(os.path.join(folder, 'deck.csv')) as handle:
            return [int(line) for line in handle if line.strip()]
    with open(os.path.join(folder, 'model.json')) as handle:
        return [json.load(handle)['choice']]
'''


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ptcg_tests_')
        self.root = Path(self.temporary.name)
        self.sdk = self.root / 'synthetic_sdk'
        (self.sdk / 'cg').mkdir(parents=True)
        (self.sdk / 'cg/__init__.py').write_text('')
        (self.sdk / 'cg/sim.py').write_text('class Battle:\n    battle_ptr = None\n')
        (self.sdk / 'cg/game.py').write_text(FAKE_GAME)
        (self.sdk / 'deck.csv').write_text('1\n' * 60)
        self.candidate = self.make_agent('candidate', 1)
        self.opponent = self.make_agent('opponent', 0)
        self.config = load_config()
        self.config['deck']['expected_sha256'] = None
        self.config['sdk_expected_sha256'] = {}
        self.config['process_timeout_seconds'] = 10
        self.config['validation_fixture'] = 'Pure Python synthetic SDK; not Pokemon gameplay evidence.'
        self.config['agents'] = {'fixture': {'entrypoint': str(self.opponent / 'main.py'), 'assets': ['model.json']}}
        self.config['package_validation']['opponents'] = ['fixture']
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))

    def tearDown(self):
        self.temporary.cleanup()

    def make_agent(self, name, choice):
        directory = self.root / name
        directory.mkdir()
        (directory / 'main.py').write_text(AGENT)
        (directory / 'deck.csv').write_text('1\n' * 60)
        (directory / 'model.json').write_text(json.dumps({'choice': choice}))
        (directory / 'notes.txt').write_text('Not included unless explicitly selected.')
        return directory

    def command(self, script, output, extra=()):
        command = [sys.executable, str(REPO / 'team_a/scripts' / script), '--agent-dir', str(self.candidate),
                   '--config', str(self.config_path), '--output-dir', str(output), *extra]
        environment = os.environ.copy()
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        return subprocess.run(command, cwd=self.root, env=environment, text=True, capture_output=True, timeout=30)

    def test_evaluation_seats_loading_and_saved_results(self):
        output = self.root / 'evaluation'
        result = self.command('evaluate.py', output, ['--opponent-dir', str(self.opponent), '--sdk-dir', str(self.sdk), '--games', '4'])
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['win'], 4)
        self.assertTrue(summary['local_checks_passed'])
        rows = json.loads((output / 'report.json').read_text())['battles']
        self.assertEqual([row['candidate_seat'] for row in rows], [0, 1, 0, 1])
        self.assertEqual([row['reverse_player'] for row in rows], [False, False, True, True])
        self.assertTrue((output / 'battle_results.csv').is_file())
        repeated = self.command('evaluate.py', output, ['--opponent-dir', str(self.opponent), '--sdk-dir', str(self.sdk)])
        self.assertNotEqual(repeated.returncode, 0)

    def test_invalid_actions_fail_without_hiding_results(self):
        (self.candidate / 'model.json').write_text(json.dumps({'choice': 9}))
        output = self.root / 'invalid'
        result = self.command('evaluate.py', output, ['--opponent-dir', str(self.opponent), '--sdk-dir', str(self.sdk), '--games', '4'])
        self.assertNotEqual(result.returncode, 0)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertFalse(summary['local_checks_passed'])
        self.assertGreater(summary['invalid_responses'], 0)
        self.assertGreater(summary['error'], 0)

    def test_build_only_is_never_ready_and_selects_explicit_files(self):
        (self.candidate / 'main.py').write_text("raise RuntimeError('Build-only must not execute agent code.')\n" + AGENT)
        output = self.root / 'unvalidated'
        result = self.command('package_agent.py', output, ['--build-only', '--asset', 'notes.txt'])
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / 'package_manifest.json').read_text())
        self.assertFalse(manifest['package_ready'])
        self.assertFalse((output / 'submission.tar.gz').exists())
        with tarfile.open(output / 'candidate_unvalidated.tar.gz') as archive:
            self.assertEqual(set(archive.getnames()), {'main.py', 'model.json', 'deck.csv', 'notes.txt'})

    def test_fresh_archive_is_validated_before_publication(self):
        output = self.root / 'packaged'
        result = self.command('package_agent.py', output, ['--sdk-dir', str(self.sdk), '--games', '2'])
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / 'package_manifest.json').read_text())
        self.assertTrue(manifest['package_ready'])
        self.assertEqual(set(manifest['files']), {'main.py', 'model.json', 'deck.csv'})
        self.assertTrue((output / 'submission.tar.gz').is_file())
        report = json.loads((output / 'archive_check_1/report.json').read_text())
        self.assertTrue(report['all_passed'])
        self.assertTrue(all(check['passed'] for check in report['contract_checks']))

    def test_failed_archive_validation_never_publishes_ready_file(self):
        (self.candidate / 'model.json').write_text(json.dumps({'choice': 9}))
        output = self.root / 'bad_package'
        result = self.command('package_agent.py', output, ['--sdk-dir', str(self.sdk), '--games', '2'])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads((output / 'package_manifest.json').read_text())['package_ready'])
        self.assertFalse((output / 'submission.tar.gz').exists())

    def test_worker_exit_preserves_parent_failure_and_partial_report(self):
        (self.candidate / 'main.py').write_text(AGENT.replace("with open(os.path.join(folder, 'model.json')) as handle:",
                                                           "os._exit(7)\n    with open(os.path.join(folder, 'model.json')) as handle:"))
        output = self.root / 'worker_exit'
        result = self.command('evaluate.py', output, ['--opponent-dir', str(self.opponent), '--sdk-dir', str(self.sdk), '--games', '2'])
        self.assertNotEqual(result.returncode, 0)
        completion = json.loads((output / 'completion.json').read_text())
        self.assertFalse(completion['passed'])
        self.assertEqual(completion['process']['returncode'], 7)
        self.assertIsNotNone(json.loads((output / 'report.json').read_text())['current_game_id'])

    def test_modified_package_assets_are_not_published(self):
        changed = AGENT.replace("return [json.load(handle)['choice']]", "value = json.load(handle)['choice']\n    with open(os.path.join(folder, 'model.json'), 'w') as handle:\n        json.dump({'choice': value, 'changed': True}, handle)\n    return [value]")
        (self.candidate / 'main.py').write_text(changed)
        output = self.root / 'mutating_package'
        result = self.command('package_agent.py', output, ['--sdk-dir', str(self.sdk), '--games', '2'])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads((output / 'package_manifest.json').read_text())['package_ready'])
        self.assertFalse((output / 'submission.tar.gz').exists())
        self.assertEqual(json.loads((self.candidate / 'model.json').read_text()), {'choice': 1})

    def test_asset_traversal_and_deck_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            stage_agent({'entrypoint': self.candidate / 'main.py', 'label': 'bad', 'assets': ['../outside.txt']}, self.root / 'bad_stage')
        self.config['deck']['expected_sha256'] = '0' * 64
        self.config_path.write_text(json.dumps(self.config))
        output = self.root / 'wrong_deck'
        result = self.command('package_agent.py', output, ['--build-only'])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads((output / 'package_manifest.json').read_text())['package_ready'])

    def test_unsupported_match_configuration_is_rejected(self):
        self.config['unit'] = 'best_of_three_match'
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):
            load_config(self.config_path)


if __name__ == '__main__':
    unittest.main()
