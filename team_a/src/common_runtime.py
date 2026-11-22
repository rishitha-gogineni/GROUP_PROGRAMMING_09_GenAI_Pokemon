"""Shared filesystem, package and standalone-agent helpers."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'team_a/configs/evaluation.json'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('unit') != 'individual_battle':
        raise ValueError('This command supports individual_battle, not a full match harness.')
    if config.get('seat_schedule') != 'alternate' or config.get('reverse_player_schedule') != 'toggle_every_two':
        raise ValueError('Unsupported seat or reverse-player schedule.')
    if config.get('confidence_interval', {}).get('method') != 'wilson_95_binary_win':
        raise ValueError('The supported interval is Wilson 95 percent for binary wins.')
    if config.get('draw_score') != 0.5:
        raise ValueError('The score rate currently gives half a point for draws.')
    for key in ['games', 'max_steps', 'process_timeout_seconds']:
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('Configuration requires a positive integer: ' + key)
    seeds = config.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Configure explicit nonnegative Python seeds.')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Configured Python seeds must be distinct.')
    return config


def select_seeds(config, games=None, seed_base=None):
    count = config['games'] if games is None else games
    if type(count) is not int or count <= 0:
        raise ValueError('Games must be positive.')
    if seed_base is not None:
        if seed_base < 0:
            raise ValueError('Seed base must be nonnegative.')
        return list(range(seed_base, seed_base + count))
    if count > len(config['seeds']):
        raise ValueError('Provide --seed-base when requesting more games than the configured seed list.')
    return config['seeds'][:count]


def resolve_agent(config, alias=None, directory=None, assets=()):
    if (alias is None) == (directory is None):
        raise ValueError('Choose one named candidate or one agent directory.')
    if alias is not None:
        if alias not in config.get('agents', {}):
            raise ValueError('Unknown configured agent: ' + alias)
        entry = config['agents'][alias]
        return {'label': alias, 'entrypoint': (REPO_ROOT / entry['entrypoint']).resolve(),
                'assets': list(entry.get('assets', [])) + list(assets)}
    directory = Path(directory).resolve()
    return {'label': directory.name, 'entrypoint': directory / 'main.py', 'assets': list(assets)}


def new_output(path, sources=()):
    path = Path(path).resolve()
    for source in sources:
        if path.is_relative_to(Path(source).resolve()):
            raise ValueError('Keep output outside agent source folders.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('Use a new or empty output directory: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_agent(spec, destination, deck_file=None):
    entry = Path(spec['entrypoint']).resolve()
    root = entry.parent
    if not entry.is_file():
        raise FileNotFoundError('Agent entry point not found: ' + str(entry))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = {'main.py': entry, 'deck.csv': Path(deck_file).resolve() if deck_file is not None else root / 'deck.csv'}
    if (root / 'model.json').is_file():
        inputs['model.json'] = root / 'model.json'
    for name in spec.get('assets', []):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Assets must use safe relative paths: ' + name)
        source = root / relative
        if not source.exists() or source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Asset missing or outside the agent folder: ' + name)
        children = sorted(source.rglob('*')) if source.is_dir() else [source]
        for child in children:
            if child.is_symlink():
                raise ValueError('Symlink assets are not supported: ' + str(child))
            if not child.is_file() or '__pycache__' in child.parts or child.suffix == '.pyc':
                continue
            key = child.relative_to(root).as_posix()
            if key in inputs and inputs[key].resolve() != child.resolve():
                raise ValueError('Asset would replace a reserved package file: ' + key)
            inputs[key] = child
    for name, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError('Required package file missing: ' + str(source))
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    deck = [int(line) for line in (destination / 'deck.csv').read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError('deck.csv must contain 60 integer card IDs.')
    return {'label': spec['label'], 'source_entrypoint': str(entry), 'deck_sha256': sha256(destination / 'deck.csv'),
            'files': {name: sha256(destination / name) for name in sorted(inputs)}}


def verify_decks(candidate, opponent, config):
    expected = config.get('deck', {}).get('expected_sha256')
    if config.get('deck', {}).get('require_same', True) and candidate['deck_sha256'] != opponent['deck_sha256']:
        raise ValueError('This benchmark requires the same deck for both agents.')
    if expected and any(item['deck_sha256'] != expected for item in [candidate, opponent]):
        raise ValueError('Deck hash differs from the configured sample benchmark.')


def verify_sdk(directory, config):
    directory = Path(directory).resolve()
    for name in ['cg/game.py', 'cg/sim.py']:
        if not (directory / name).is_file():
            raise FileNotFoundError('SDK directory must contain ' + name)
    for name, expected in config.get('sdk_expected_sha256', {}).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha256(directory / relative) != expected:
            raise ValueError('SDK file does not match the configured version: ' + name)
    return {str(path.relative_to(directory)): sha256(path) for path in sorted((directory / 'cg').rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}


def check_import_collisions(candidate, opponent):
    def modules(directory):
        return {p.relative_to(directory).parts[0].removesuffix('.py') for p in Path(directory).rglob('*.py')
                if p.name != 'main.py'}
    overlap = modules(candidate) & modules(opponent)
    if overlap:
        raise ValueError('Use standalone entries or unique Python asset namespaces for both agents: ' + ', '.join(sorted(overlap)))


@contextmanager
def agent_import_path(directory):
    previous = list(sys.path)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = previous


def load_agent(directory):
    directory = Path(directory).resolve()
    path = directory / 'main.py'
    namespace = {}
    with agent_import_path(directory):
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    functions = [value for value in namespace.values() if callable(value)]
    if not functions or functions[-1].__name__ != 'agent':
        raise ValueError('The last callable in main.py must be agent.')
    callback = functions[-1]
    configuration = {'__raw_path__': str(path)}

    def invoke(observation):
        with agent_import_path(directory):
            return callback(observation, configuration)

    return invoke, namespace
