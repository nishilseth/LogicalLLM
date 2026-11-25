"""Parse / compare PokerBench action strings: fold | check | call | bet N | raise N."""
import re
from typing import NamedTuple, Optional

ACTION_RE = re.compile(r"\b(fold|check|call|bet|raise)\b(?:\s+(\d+(?:\.\d+)?))?", re.I)
STRICT_RE = re.compile(r"\s*(fold|check|call|(?:bet|raise)\s+\d+(?:\.\d+)?)\s*\.?\s*", re.I)
SIZED = {"bet", "raise"}


class Action(NamedTuple):
    kind: str
    amount: Optional[float] = None

    def __str__(self):
        return self.kind if self.amount is None else f"{self.kind} {self.amount:g}"


def parse_action(text: str) -> Optional[Action]:
    """First action mentioned in `text`, or None. Sized actions without a size
    (e.g. a bare "raise") are returned with amount=None."""
    m = ACTION_RE.search(text or "")
    if not m:
        return None
    kind = m.group(1).lower()
    amt = float(m.group(2)) if (m.group(2) and kind in SIZED) else None
    return Action(kind, amt)


def is_strict_format(text: str) -> bool:
    """True if the completion is *only* a single well-formed action."""
    return bool(STRICT_RE.fullmatch(text or ""))
