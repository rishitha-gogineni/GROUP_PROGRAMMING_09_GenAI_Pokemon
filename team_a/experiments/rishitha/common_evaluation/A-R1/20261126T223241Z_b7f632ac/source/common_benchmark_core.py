"""Neutral shared benchmark helpers. No team model or private policy is included."""
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile

OPPONENTS = ('sample_random', 'first_options', 'last_options')
DRAW_RULE = 'At most three games; stop at two wins. At the cap, more game wins takes the match; equal wins is a draw.'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def deck_ids(path):
    values = [int(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(values) != 60:
        raise ValueError('deck.csv must contain 60 integer card IDs.')
    return values


def extract_agent(path, destination, maximum_bytes=512*1024*1024, maximum_files=5000):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in member.name:
                raise ValueError('Unsafe archive path: ' + member.name)
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('Only regular files and directories are supported.')
            name = relative.as_posix()
            if name in seen or name == '.':
                raise ValueError('Duplicate or invalid archive file: ' + name)
            seen.add(name)
            total += member.size
            if len(seen) > maximum_files or total > maximum_bytes:
                raise ValueError('Archive exceeds the notebook extraction budget.')
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member).read()
            if len(content) != member.size:
                raise ValueError('Archive member size mismatch.')
            target.write_bytes(content)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    if not {'main.py', 'deck.csv'} <= seen:
        raise ValueError('The archive must contain main.py and deck.csv at its root.')
    return {'unpacked_bytes': total, 'files': {name: file_hash(destination / name) for name in sorted(seen)}}


def validate_action(action, selection):
    if not isinstance(action, list) or not all(type(index) is int for index in action):
        raise ValueError('Actions must be a list of Python integer option indices.')
    options = selection['option']
    if not selection['minCount'] <= len(action) <= selection['maxCount']:
        raise ValueError('Invalid selection count.')
    if len(set(action)) != len(action) or any(index < 0 or index >= len(options) for index in action):
        raise ValueError('Duplicate or out-of-range option indices.')


def reference_action(name, observation, deck, rng):
    if observation.get('select') is None:
        return list(deck)
    selection = observation['select']
    count, total = selection['maxCount'], len(selection['option'])
    if name == 'sample_random':
        action = rng.sample(range(total), count)
    elif name == 'first_options':
        action = list(range(count))
    elif name == 'last_options':
        action = list(range(total-count, total))
    else:
        raise ValueError('Unknown reference opponent.')
    validate_action(action, selection)
    return action


@contextmanager
def package_context(directory):
    previous_cwd, previous_path = os.getcwd(), list(sys.path)
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def load_candidate(directory):
    directory = Path(directory).resolve()
    path = directory / 'main.py'
    namespace = {}
    with package_context(directory):
        exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), namespace)
    callables = [value for value in namespace.values() if callable(value)]
    function = namespace.get('agent')
    if not callable(function) or not callables or callables[-1] is not function:
        raise ValueError('agent must be the final callable in main.py.')
    signature = inspect.signature(function)
    configuration = {'__raw_path__': str(path)}
    sentinel = object()
    try:
        signature.bind(sentinel, configuration)
        mode = 'two_positional_arguments'
    except TypeError:
        try:
            signature.bind(sentinel, configuration=configuration)
            mode = 'configuration_keyword'
        except TypeError:
            signature.bind(sentinel)
            mode = 'one_argument'

    def invoke(observation, seed):
        configuration['seed'] = seed
        with package_context(directory):
            if mode == 'two_positional_arguments':
                return function(observation, configuration)
            if mode == 'configuration_keyword':
                return function(observation, configuration=configuration)
            return function(observation)

    return invoke, mode


def match_outcome(outcomes):
    if not outcomes or any(value not in ('win', 'loss', 'draw') for value in outcomes):
        return 'error'
    wins, losses = outcomes.count('win'), outcomes.count('loss')
    if wins >= 2:
        return 'win'
    if losses >= 2:
        return 'loss'
    if len(outcomes) < 3:
        return None
    return 'win' if wins > losses else 'loss' if losses > wins else 'draw'


def wilson(wins, total):
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = wins/total, 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def score_bound(mean, total):
    if not total or mean is None:
        return None
    radius = math.sqrt(math.log(40)/(2*total))
    return [max(0.0, mean-radius), min(1.0, mean+radius)]


def summarize(report, requested, process_ok):
    matches = report.get('matches', [])
    valid = [row for row in matches if row.get('outcome') in ('win', 'loss', 'draw')]
    wins = sum(row['outcome'] == 'win' for row in valid)
    losses = sum(row['outcome'] == 'loss' for row in valid)
    draws = sum(row['outcome'] == 'draw' for row in valid)
    errors = sum(row.get('outcome') == 'error' for row in matches)
    games = [game for row in matches for game in row.get('games', [])]
    invalid = sum(game.get('invalid_responses', 0) for game in games)
    rejected = sum(game.get('engine_rejections', 0) for game in games)
    complete = (process_ok and report.get('contract_passed') is True and report.get('status') == 'passed'
                and len(valid) == requested and errors == 0 and invalid == rejected == 0)
    score = (wins+.5*draws)/len(valid) if valid else None
    decisions = sum(game.get('candidate_decisions', 0) for game in games)
    return {'requested_matches': requested, 'valid_matches': len(valid), 'error_matches': errors,
            'unfinished_or_unstarted_matches': requested-len(valid)-errors,
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins/len(valid) if valid else None, 'score_rate': score,
            'win_rate_wilson_95': wilson(wins, len(valid)),
            'score_rate_hoeffding_95': score_bound(score, len(valid)),
            'games_played': len(games), 'invalid_responses': invalid, 'engine_rejections': rejected,
            'candidate_decisions': decisions,
            'candidate_mean_decision_ms': sum(game.get('candidate_total_decision_ms', 0) for game in games)/decisions if decisions else None,
            'candidate_max_decision_ms': max([game.get('candidate_max_decision_ms', 0) for game in games] or [0]),
            'complete_and_valid': complete}
