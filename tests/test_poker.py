import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from logicalllm.poker.actions import parse_action, is_strict_format, Action
from logicalllm.poker.reward import poker_reward, MAX_REWARD
from logicalllm.poker.evaluate import score


def test_parse():
    assert parse_action("fold") == Action("fold")
    assert parse_action("Raise 84.5 chips") == Action("raise", 84.5)
    assert parse_action("I would bet 10") == Action("bet", 10)
    assert parse_action("raise") == Action("raise", None)
    assert parse_action("???") is None


def test_strict_format():
    assert is_strict_format("call") and is_strict_format("bet 12") and is_strict_format("raise 3.5.")
    assert not is_strict_format("I think call") and not is_strict_format("bet") and not is_strict_format("")


def test_reward_ordering():
    gold = "bet 20"
    exact = poker_reward("bet 20", gold)
    close = poker_reward("bet 15", gold)
    far = poker_reward("bet 2", gold)
    wrong_type = poker_reward("raise 20", gold)
    garbage = poker_reward("hmm", gold)
    assert exact == MAX_REWARD
    assert exact > close > far > wrong_type >= garbage == 0.0
    assert poker_reward("fold", "fold") == MAX_REWARD
    assert poker_reward("fold please", "fold") < MAX_REWARD  # format penalty, still gets type credit


def test_score_metrics():
    r = score(["call", "fold", "bet 10", "xx"], ["call", "check", "bet 12", "fold"])
    assert r["parse_rate"] == 0.75 and r["type_accuracy"] == 0.5 and r["exact_match"] == 0.25
