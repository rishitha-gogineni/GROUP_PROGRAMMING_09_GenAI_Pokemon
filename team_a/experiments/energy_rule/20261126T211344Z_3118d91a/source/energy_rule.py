"""Conservative attachment target experiment. Other choices stay unchanged."""
from collections import Counter
import re
import unicodedata

RULE_VERSION = 'energy-target-v1'


def missing_energy(cost, attached):
    need, have = Counter(cost), Counter(attached)
    colored = sum(max(0, count-have[color]) for color, count in need.items() if color != 0)
    used = sum(min(count, have[color]) for color, count in need.items() if color != 0)
    colorless = max(0, need[0] - (len(attached)-used))
    return colored + colorless


def card_attacks(card, cards, attacks):
    metadata = cards.get(str(card.get('id')))
    if not metadata or not metadata.get('attacks'):
        return None
    values = []
    for attack_id in metadata['attacks']:
        attack = attacks.get(str(attack_id))
        if not attack or not isinstance(attack.get('energies'), list):
            return None
        if any(type(color) is not int or color not in range(10) for color in attack['energies']):
            return None
        values.append(attack)
    return values


def basic_attached(card, cards):
    attached = card.get('energies')
    if not isinstance(attached, list) or any(type(color) is not int or color not in range(1, 10) for color in attached):
        return None
    physical = card.get('energyCards')
    if not isinstance(physical, list) or len(physical) != len(attached):
        return None
    for item in physical:
        metadata = cards.get(str(item.get('id'))) if isinstance(item, dict) else None
        if not metadata or metadata.get('cardType') != 5:
            return None
    return attached


def has_energy_effect(attacks):
    for attack in attacks:
        text = unicodedata.normalize('NFKD', attack.get('text') or '').encode('ascii', 'ignore').decode().lower()
        refuel = 'discard' in text and 'energy' in text and 'from this pokemon' in text
        scales = bool(re.search(r'for each.{0,100}energy.{0,100}attached', text))
        returns = 'energy' in text and 'return' in text and 'attached' in text
        if refuel or scales or returns:
            return True
    return False


def energy_decision(observation, original_action, cards, attacks):
    detail = {'rule_version': RULE_VERSION, 'eligible': False, 'changed': False,
              'reason': 'outside_supported_attachment', 'original_action': list(original_action),
              'action': list(original_action), 'targets': []}
    selection = observation.get('select') or {}
    if (selection.get('type') != 0 or selection.get('minCount') != 1 or selection.get('maxCount') != 1
            or len(original_action) != 1):
        return original_action, detail
    options = selection.get('option') or []
    original = options[original_action[0]]
    if original.get('type') != 8 or original.get('area') != 2 or original.get('inPlayArea') != 4:
        return original_action, detail
    detail['eligible'] = True
    current = observation.get('current') or {}
    players, who = current.get('players'), current.get('yourIndex')
    if not isinstance(players, list) or len(players) != 2 or type(who) is not int or who not in (0, 1):
        detail['reason'] = 'unknown_state'
        return original_action, detail
    own = players[who]
    hand, active = own.get('hand') or [], own.get('active') or []
    index = original.get('index')
    if type(index) is not int or index not in range(len(hand)) or len(active) != 1:
        detail['reason'] = 'unknown_source_or_active'
        return original_action, detail
    energy = cards.get(str(hand[index].get('id')))
    if not energy or energy.get('cardType') != 5 or type(energy.get('energyType')) is not int:
        detail['reason'] = 'unsupported_energy_card'
        return original_action, detail
    color = energy['energyType']
    if color not in range(1, 10):
        detail['reason'] = 'unsupported_energy_type'
        return original_action, detail
    active_moves = card_attacks(active[0], cards, attacks)
    active_energy = basic_attached(active[0], cards)
    if active_moves is None or active_energy is None:
        detail['reason'] = 'unknown_active_metadata_or_special_energy'
        return original_action, detail
    detail['active_attack_deficits'] = [missing_energy(move['energies'], active_energy) for move in active_moves]
    if has_energy_effect(active_moves):
        detail['reason'] = 'keep_active_for_energy_effect'
        return original_action, detail
    if any(detail['active_attack_deficits']):
        detail['reason'] = 'active_still_needs_energy'
        return original_action, detail
    bench = own.get('bench') or []
    candidates = []
    seen = set()
    for option_index, option in enumerate(options):
        if (option.get('type') != 8 or option.get('area') != 2 or option.get('index') != index
                or option.get('inPlayArea') != 5):
            continue
        target_index = option.get('inPlayIndex')
        if type(target_index) is not int or target_index not in range(len(bench)) or target_index in seen:
            continue
        seen.add(target_index)
        target = bench[target_index]
        moves, attached = card_attacks(target, cards, attacks), basic_attached(target, cards)
        if moves is None or attached is None:
            continue
        before = min(missing_energy(move['energies'], attached) for move in moves)
        after = min(missing_energy(move['energies'], attached+[color]) for move in moves)
        hp, maximum = target.get('hp'), target.get('maxHp')
        if type(hp) not in (int, float) or type(maximum) not in (int, float) or maximum <= 0 or hp <= 0:
            continue
        item = {'option_index': option_index, 'bench_index': target_index, 'card_id': target.get('id'),
                'missing_before': before, 'missing_after': after, 'hp_fraction': hp/maximum}
        detail['targets'].append(item)
        if before > 0 and after < before:
            candidates.append(((after, -hp/maximum, target_index, option_index), option_index))
    if not candidates:
        detail['reason'] = 'no_supported_bench_target_benefits'
        return original_action, detail
    chosen = min(candidates)[1]
    action = [chosen]
    detail.update(changed=True, reason='active_ready_prepare_bench', action=action)
    return action, detail
