"""Team B card lookup, card-data validation and deck checks. Owner: Chelsi.

Primary source: the engine's own card table (cg.api.all_card_data / all_attack),
because its IDs are exactly what the simulator uses. The English CSV
(EN_Card_Data_R2_full.csv) is checked against it; note the CSV grain is one row
per (card, attack), so a card ID can appear on several rows.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import engine

# cg.api.CardType values
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY = range(7)
CARD_TYPE_NAMES = ["pokemon", "item", "tool", "supporter", "stadium", "basic_energy", "special_energy"]
N_ENERGY_TYPES = 12  # cg.api.EnergyType 0..11
COLORLESS, RAINBOW = 0, 10


@dataclass(frozen=True)
class AttackInfo:
    attack_id: int
    name: str
    damage: int
    cost: tuple[int, ...]  # EnergyType per required energy


@dataclass(frozen=True)
class CardInfo:
    card_id: int
    name: str
    card_type: int
    hp: int
    energy_type: int
    weakness: int | None
    resistance: int | None
    retreat_cost: int
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    tera: bool
    ace_spec: bool
    evolves_from: str | None
    attacks: tuple[AttackInfo, ...] = field(default_factory=tuple)

    @property
    def is_pokemon(self) -> bool:
        return self.card_type == POKEMON

    @property
    def is_energy(self) -> bool:
        return self.card_type in (BASIC_ENERGY, SPECIAL_ENERGY)

    @property
    def max_damage(self) -> int:
        return max((a.damage for a in self.attacks), default=0)

    @property
    def max_attack_cost(self) -> int:
        return max((len(a.cost) for a in self.attacks), default=0)

    @property
    def min_attack_cost(self) -> int:
        return min((len(a.cost) for a in self.attacks), default=0)

    @property
    def prize_value(self) -> int:
        """Prizes the opponent takes when this Pokemon is knocked out."""
        return 3 if self.mega_ex else 2 if self.ex else 1


class CardTable:
    """ID -> CardInfo and attack ID -> AttackInfo, from the running engine."""

    def __init__(self, cards: dict[int, CardInfo], attacks: dict[int, AttackInfo]):
        self.cards = cards
        self.attacks = attacks

    def __contains__(self, card_id: int) -> bool:
        return card_id in self.cards

    def get(self, card_id: int | None) -> CardInfo | None:
        return self.cards.get(card_id) if card_id is not None else None

    def attack(self, attack_id: int | None) -> AttackInfo | None:
        return self.attacks.get(attack_id) if attack_id is not None else None


@lru_cache(maxsize=1)
def load_card_table() -> CardTable:
    api = engine.api()
    attacks = {
        a.attackId: AttackInfo(a.attackId, a.name, int(a.damage or 0), tuple(int(e) for e in a.energies))
        for a in api.all_attack()
    }
    cards = {}
    for c in api.all_card_data():
        cards[c.cardId] = CardInfo(
            card_id=c.cardId,
            name=c.name,
            card_type=int(c.cardType),
            hp=int(c.hp or 0),
            energy_type=int(c.energyType or 0),
            weakness=None if c.weakness is None else int(c.weakness),
            resistance=None if c.resistance is None else int(c.resistance),
            retreat_cost=int(c.retreatCost or 0),
            basic=bool(c.basic),
            stage1=bool(c.stage1),
            stage2=bool(c.stage2),
            ex=bool(c.ex),
            mega_ex=bool(c.megaEx),
            tera=bool(c.tera),
            ace_spec=bool(c.aceSpec),
            evolves_from=c.evolvesFrom,
            attacks=tuple(attacks[a] for a in c.attacks if a in attacks),
        )
    return CardTable(cards, attacks)


def card_vector(info: CardInfo | None) -> list[float]:
    """Fixed-length numeric card features (public card attributes only).
    Unknown/None card -> all zeros with the `unknown` flag set."""
    if info is None:
        return [1.0] + [0.0] * (len(CARD_TYPE_NAMES) + N_ENERGY_TYPES + 12)
    type_onehot = [1.0 if info.card_type == t else 0.0 for t in range(len(CARD_TYPE_NAMES))]
    energy_onehot = [1.0 if info.energy_type == e else 0.0 for e in range(N_ENERGY_TYPES)]
    return [0.0] + type_onehot + energy_onehot + [
        info.hp / 100.0,
        info.retreat_cost / 4.0,
        float(info.basic),
        float(info.stage1),
        float(info.stage2),
        float(info.ex),
        float(info.mega_ex),
        float(info.ace_spec),
        info.max_damage / 100.0,
        info.max_attack_cost / 4.0,
        info.min_attack_cost / 4.0,
        float(len(info.attacks)),
    ]


CARD_VECTOR_SIZE = len(card_vector(None))


# --------------------------------------------------------------------------- validation

def validate_card_csv(csv_path: str | Path, table: CardTable | None = None) -> dict:
    """Compare the English card CSV with the engine card table. Returns a report dict."""
    table = table or load_card_table()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids, bad_ids = [], 0
    for r in rows:
        try:
            ids.append(int(r["Card ID"]))
        except (KeyError, ValueError):
            bad_ids += 1
    unique = set(ids)
    engine_ids = set(table.cards)
    name_mismatch = sorted(
        cid for cid in unique & engine_ids
        if next(r["Card Name"] for r in rows if r["Card ID"] == str(cid)) != table.cards[cid].name
    )
    return {
        "csv_rows": len(rows),
        "csv_unique_card_ids": len(unique),
        "csv_rows_with_bad_id": bad_ids,
        "csv_ids_repeated_on_multiple_rows": sum(1 for n in Counter(ids).values() if n > 1),
        "engine_cards": len(engine_ids),
        "in_csv_not_engine": sorted(unique - engine_ids),
        "in_engine_not_csv": sorted(engine_ids - unique),
        "name_mismatches": name_mismatch,
    }


def deck_problems(deck: list[int], table: CardTable | None = None) -> list[str]:
    """Basic deck-legality checks; returns a list of problems (empty == looks valid).
    The engine/Kaggle validation remains the final authority."""
    table = table or load_card_table()
    problems = []
    if len(deck) != 60:
        problems.append(f"deck has {len(deck)} cards, expected 60")
    unknown = sorted({c for c in deck if c not in table})
    if unknown:
        problems.append(f"unknown card IDs: {unknown}")
        return problems
    infos = [table.cards[c] for c in deck]
    if not any(i.is_pokemon and i.basic for i in infos):
        problems.append("no Basic Pokemon")
    by_name = Counter(i.name for i in infos if i.card_type != BASIC_ENERGY)
    for name, n in by_name.items():
        if n > 4:
            problems.append(f"{n} copies of {name} (max 4)")
    if sum(i.ace_spec for i in infos) > 1:
        problems.append("more than 1 ACE SPEC card")
    return problems


def deck_summary(deck: list[int], table: CardTable | None = None) -> list[tuple[int, str, str, int]]:
    """(card_id, name, type, copies) rows for reports."""
    table = table or load_card_table()
    counts = Counter(deck)
    return [
        (cid, table.cards[cid].name, CARD_TYPE_NAMES[table.cards[cid].card_type], n)
        for cid, n in sorted(counts.items())
    ]
