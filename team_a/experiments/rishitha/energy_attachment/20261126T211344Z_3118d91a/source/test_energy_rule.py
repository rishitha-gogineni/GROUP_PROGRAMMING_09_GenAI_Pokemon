"""Synthetic checks only. No game engine or trained policy is imported."""
from copy import deepcopy
import unittest

from energy_rule import energy_decision, missing_energy
from energy_summary import assess, summarize_phase


def fixture():
    cards = {'3': {'cardType': 5, 'energyType': 3, 'attacks': []},
             '10': {'cardType': 0, 'attacks': [101]}, '11': {'cardType': 0, 'attacks': [102]}}
    attacks = {'101': {'energies': [3, 3], 'text': ''}, '102': {'energies': [3], 'text': ''}}
    def pokemon(card_id, count, hp=100):
        return {'id': card_id, 'hp': hp, 'maxHp': 100, 'energies': [3]*count, 'energyCards': [{'id': 3}]*count}
    own = {'hand': [{'id': 3}], 'active': [pokemon(10, 2)], 'bench': [pokemon(11, 0)]}
    observation = {'current': {'yourIndex': 0, 'players': [own, {}]},
                   'select': {'type': 0, 'minCount': 1, 'maxCount': 1,
                              'option': [{'type': 8, 'area': 2, 'index': 0, 'inPlayArea': 4, 'inPlayIndex': 0},
                                         {'type': 8, 'area': 2, 'index': 0, 'inPlayArea': 5, 'inPlayIndex': 0},
                                         {'type': 13, 'attackId': 101}]}}
    return observation, cards, attacks


class EnergyRuleChecks(unittest.TestCase):
    def test_cost_matching(self):
        self.assertEqual(missing_energy([3, 0], [3, 2]), 0)
        self.assertEqual(missing_energy([3, 3], [3, 2]), 1)
        self.assertEqual(missing_energy([3, 0, 0], [3]), 2)
        self.assertEqual(missing_energy([3, 3, 0], [3, 2]), 1)
        self.assertEqual(missing_energy([], []), 0)

    def test_ready_active_prepares_bench(self):
        obs, cards, attacks = fixture()
        before = deepcopy(obs)
        action, detail = energy_decision(obs, [0], cards, attacks)
        self.assertEqual(action, [1])
        self.assertTrue(detail['eligible'] and detail['changed'])
        self.assertEqual(detail['targets'][0]['missing_after'], 0)
        self.assertEqual(obs, before)

    def test_active_needs_energy(self):
        obs, cards, attacks = fixture()
        obs['current']['players'][0]['active'][0]['energies'].pop()
        obs['current']['players'][0]['active'][0]['energyCards'].pop()
        action, detail = energy_decision(obs, [0], cards, attacks)
        self.assertEqual(action, [0])
        self.assertEqual(detail['reason'], 'active_still_needs_energy')

    def test_all_active_attacks_must_be_ready(self):
        obs, cards, attacks = fixture()
        cards['10']['attacks'].append(103)
        attacks['103'] = {'energies': [3, 3, 3], 'text': ''}
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_other_choices_and_selection_types_unchanged(self):
        obs, cards, attacks = fixture()
        self.assertEqual(energy_decision(obs, [2], cards, attacks)[0], [2])
        self.assertEqual(energy_decision(obs, [1], cards, attacks)[0], [1])
        for selection_type in (1, 4, 8, 9):
            obs['select']['type'] = selection_type
            self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_unknown_metadata_unchanged(self):
        obs, cards, attacks = fixture()
        del cards['10']
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])
        obs, cards, attacks = fixture()
        del cards['11']
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_special_energy_unchanged(self):
        obs, cards, attacks = fixture()
        cards['3']['cardType'] = 6
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])
        obs, cards, attacks = fixture()
        obs['current']['players'][0]['active'][0]['energies'] = [10, 3]
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_energy_effect_guards(self):
        for text in ('Discard 2 Energy from this Pokémon.',
                     'This attack does 20 damage for each Energy attached to this Pokémon.',
                     'Return an attached Energy to your hand.'):
            obs, cards, attacks = fixture()
            attacks['101']['text'] = text
            action, detail = energy_decision(obs, [0], cards, attacks)
            self.assertEqual(action, [0], text)
            self.assertEqual(detail['reason'], 'keep_active_for_energy_effect')

    def test_wrong_energy_and_already_ready_bench_unchanged(self):
        obs, cards, attacks = fixture()
        attacks['102']['energies'] = [2]
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])
        obs, cards, attacks = fixture()
        bench = obs['current']['players'][0]['bench'][0]
        bench.update(energies=[3], energyCards=[{'id': 3}])
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_same_energy_card_is_preserved(self):
        obs, cards, attacks = fixture()
        obs['select']['option'][1]['index'] = 1
        obs['current']['players'][0]['hand'].append({'id': 3})
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [0])

    def test_health_breaks_equal_deficit_tie(self):
        obs, cards, attacks = fixture()
        own = obs['current']['players'][0]
        own['bench'][0]['hp'] = 20
        own['bench'].append({'id': 11, 'hp': 80, 'maxHp': 100, 'energies': [], 'energyCards': []})
        obs['select']['option'].append({'type': 8, 'area': 2, 'index': 0, 'inPlayArea': 5, 'inPlayIndex': 1})
        self.assertEqual(energy_decision(obs, [0], cards, attacks)[0], [3])

    def test_summary_rejects_missing_or_failed_games(self):
        report = {'contract_passed': True, 'status': 'passed', 'games': [{'status': 'passed', 'outcome': 'win'}]}
        self.assertTrue(summarize_phase(report, 1, True)['complete_and_valid'])
        self.assertFalse(summarize_phase(report, 2, True)['complete_and_valid'])
        self.assertFalse(summarize_phase(report, 1, False)['complete_and_valid'])
        report['diagnostics_truncated'] = True
        self.assertFalse(summarize_phase(report, 1, True)['complete_and_valid'])

    def test_assessment_requires_changes_and_consistency(self):
        rows = []
        for batch in range(2):
            for name, score in [('original_vs_ar1', .4), ('energy_vs_original', .6), ('energy_vs_ar1', .6)]:
                rows.append({'batch': batch, 'comparison': name, 'score_rate': score,
                             'complete_and_valid': True, 'rule_changes': 0 if name.startswith('original') else 2})
        self.assertEqual(assess(rows, 2), 'promising_pilot_repeat_before_selection')
        rows[-1]['score_rate'] = .3
        self.assertEqual(assess(rows, 2), 'no_consistent_gain_across_both_batches')
        for row in rows:
            row['rule_changes'] = 0
        self.assertEqual(assess(rows, 2), 'no_target_changes_observed')
        self.assertEqual(assess(rows[:-1], 2), 'incomplete_or_invalid')


if __name__ == '__main__':
    unittest.main(verbosity=2)
