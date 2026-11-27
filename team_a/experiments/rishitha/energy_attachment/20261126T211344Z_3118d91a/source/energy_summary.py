"""Aggregate completed games without hiding failed or incomplete runs."""
from common_benchmark_core import wilson


def summarize_phase(report, requested, process_ok):
    games = report.get('games', [])
    valid = [row for row in games if row.get('status') == 'passed' and row.get('outcome') in ('win', 'loss', 'draw')]
    wins = sum(row['outcome'] == 'win' for row in valid)
    losses = sum(row['outcome'] == 'loss' for row in valid)
    draws = sum(row['outcome'] == 'draw' for row in valid)
    invalid = sum(row.get('invalid_responses', 0) for row in games)
    rejected = sum(row.get('engine_rejections', 0) for row in games)
    complete = (process_ok and report.get('status') == 'passed' and report.get('contract_passed') is True
                and len(valid) == requested and len(games) == requested and invalid == rejected == 0
                and not report.get('diagnostics_truncated', False))
    return {'requested': requested, 'valid': len(valid), 'failed_or_unfinished': requested-len(valid),
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins/len(valid) if valid else None,
            'score_rate': (wins+0.5*draws)/len(valid) if valid else None,
            'win_rate_wilson_95': wilson(wins, len(valid)),
            'invalid_responses': invalid, 'engine_rejections': rejected,
            'rule_eligible': sum(row.get('rule_eligible', 0) for row in games),
            'rule_changes': sum(row.get('rule_changes', 0) for row in games),
            'complete_and_valid': complete}


def aggregate(rows, name):
    values = [row for row in rows if row['comparison'] == name]
    totals = {key: sum(row[key] for row in values) for key in
              ['requested', 'valid', 'failed_or_unfinished', 'wins', 'losses', 'draws',
               'invalid_responses', 'engine_rejections', 'rule_eligible', 'rule_changes']}
    n = totals['valid']
    return {'comparison': name, **totals,
            'win_rate': totals['wins']/n if n else None,
            'score_rate': (totals['wins']+0.5*totals['draws'])/n if n else None,
            'win_rate_wilson_95': wilson(totals['wins'], n),
            'complete_and_valid': bool(values) and all(row['complete_and_valid'] for row in values)}


def assess(rows, batches):
    required = ('original_vs_ar1', 'energy_vs_original', 'energy_vs_ar1')
    mapping = {(row['batch'], row['comparison']): row for row in rows}
    expected = {(batch, name) for batch in range(batches) for name in required}
    if set(mapping) != expected or len(mapping) != len(rows) or not all(row['complete_and_valid'] for row in rows):
        return 'incomplete_or_invalid'
    if sum(row['rule_changes'] for row in rows if row['comparison'].startswith('energy_')) == 0:
        return 'no_target_changes_observed'
    positive = all(mapping[batch, 'energy_vs_original']['score_rate'] > .5 and
                   mapping[batch, 'energy_vs_ar1']['score_rate'] > mapping[batch, 'original_vs_ar1']['score_rate']
                   for batch in range(batches))
    return 'promising_pilot_repeat_before_selection' if positive else 'no_consistent_gain_across_both_batches'
