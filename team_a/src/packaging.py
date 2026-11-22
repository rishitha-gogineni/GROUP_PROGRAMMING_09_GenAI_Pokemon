"""Build an explicit standalone archive and validate a fresh extraction."""

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import traceback
import uuid

from .common_runtime import (DEFAULT_CONFIG, load_config, new_output, resolve_agent, save_json,
                             sha256, stage_agent, verify_decks)
from .evaluation import run_pair


def build_and_extract(source, archive_path, extracted, expected_files):
    ast.parse((source / 'main.py').read_text(encoding='utf-8'))
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in sorted(expected_files):
            archive.add(source / name, arcname=name, recursive=False)
    extracted.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected_files) or {item.name for item in members} != set(expected_files):
            raise ValueError('Archive member list differs from the selected files.')
        for item in members:
            relative = Path(item.name)
            if not item.isfile() or item.issym() or item.islnk() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unexpected archive member: ' + item.name)
            payload = archive.extractfile(item).read()
            if payload != (source / item.name).read_bytes():
                raise ValueError('Archive bytes do not match the staged source: ' + item.name)
            target = extracted / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return {name: sha256(extracted / name) for name in sorted(expected_files)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Package a standalone agent and test a fresh extraction before publishing submission.tar.gz.')
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument('--candidate', help='Configured agent, such as ar2.')
    candidate.add_argument('--agent-dir', type=Path)
    parser.add_argument('--asset', action='append', default=[])
    parser.add_argument('--deck-file', type=Path)
    parser.add_argument('--sdk-dir', type=Path, help='Required for archive battle validation.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--opponent', action='append', help='Configured validation opponent; repeat to select more than one.')
    parser.add_argument('--games', type=int, help='Archive validation battles per opponent.')
    parser.add_argument('--seed-base', type=int)
    parser.add_argument('--build-only', action='store_true', help='Create candidate_unvalidated.tar.gz without executing agents. It is never marked ready.')
    args = parser.parse_args(argv)
    if not args.build_only and args.sdk_dir is None:
        parser.error('--sdk-dir is required unless --build-only is used.')
    output = None
    manifest = {'built': False, 'package_ready': False, 'kaggle_status': 'not_submitted',
                'scope': 'Local archive and direct SDK checks, not official Kaggle acceptance.'}
    try:
        config = load_config(args.config)
        spec = resolve_agent(config, args.candidate, args.agent_dir, args.asset)
        output = new_output(args.output_dir, [spec['entrypoint'].parent])
        save_json(output / 'package_manifest.json', manifest)
        settings = config['package_validation']
        games = settings['games_per_opponent'] if args.games is None else args.games
        seed_base = settings['seed_base'] if args.seed_base is None else args.seed_base
        if type(games) is not int or games <= 0 or type(seed_base) is not int or seed_base < 0:
            raise ValueError('Archive games must be positive and seed base nonnegative.')
        opponents = settings['opponents'] if args.opponent is None else args.opponent
        if not args.build_only and (not opponents or len(set(opponents)) != len(opponents)):
            raise ValueError('Choose one or more distinct archive opponents.')
        deck = args.deck_file
        if deck is None and args.sdk_dir is not None and (args.sdk_dir / 'deck.csv').is_file():
            deck = args.sdk_dir / 'deck.csv'
        with tempfile.TemporaryDirectory(prefix='ptcg_package_') as temporary:
            temporary = Path(temporary)
            staged = temporary / 'source'
            info = stage_agent(spec, staged, deck)
            verify_decks(info, info, config)
            candidate_archive = temporary / 'candidate.tar.gz'
            extracted = temporary / 'fresh_extraction'
            hashes = build_and_extract(staged, candidate_archive, extracted, info['files'])
            manifest.update(built=True, candidate=info, files=hashes, sha256=sha256(candidate_archive), bytes=candidate_archive.stat().st_size)
            if args.build_only:
                shutil.copyfile(candidate_archive, output / 'candidate_unvalidated.tar.gz')
                manifest['status'] = 'built_unvalidated'
                manifest['note'] = 'No agent or SDK code executed. This artifact is not a validated submission.'
                save_json(output / 'package_manifest.json', manifest)
                print(json.dumps(manifest, indent=2))
                return 0
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
            checks = {}
            for index, label in enumerate(opponents):
                opponent_spec = resolve_agent(config, alias=label)
                opponent_dir = temporary / ('opponent_' + str(index))
                opponent_info = stage_agent(opponent_spec, opponent_dir, deck)
                verify_decks(info, opponent_info, config)
                seeds = list(range(seed_base + index*games, seed_base + (index+1)*games))
                passed, summary = run_pair(extracted, opponent_dir, args.sdk_dir, output / ('archive_check_' + str(index+1)),
                                           config, seeds, run_id, 'archive_' + label,
                                           {'candidate': info, 'opponent': opponent_info})
                checks[label] = {'passed': passed, 'summary': summary}
                manifest['archive_checks'] = checks
                save_json(output / 'package_manifest.json', manifest)
            ready = bool(checks) and all(item['passed'] for item in checks.values())
            if ready and any(sha256(extracted / name) != digest for name, digest in hashes.items()):
                raise ValueError('The agent modified a packaged file during validation.')
            if ready:
                target = output / 'submission.tar.gz'
                shutil.copyfile(candidate_archive, target)
                if sha256(target) != manifest['sha256']:
                    raise ValueError('Final archive hash mismatch.')
            manifest.update(package_ready=ready, status='ready' if ready else 'validation_failed')
        save_json(output / 'package_manifest.json', manifest)
        print(json.dumps(manifest, indent=2))
        return 0 if manifest['package_ready'] else 1
    except Exception as error:
        manifest.update(package_ready=False, status='error', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        if output is not None:
            (output / 'submission.tar.gz').unlink(missing_ok=True)
            save_json(output / 'package_manifest.json', manifest)
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
