"""Unit tests for src/taxonomy helpers."""
from src.taxonomy import (
    CATEGORY_CODES, RULE_PRECEDENCE,
    normalize, is_idk, gold_ngrams,
    any_ngram_in_chunks, is_tersification,
)


def test_normalize_lowercases_and_strips_punct():
    assert normalize("  Hello, World!  ") == "hello world"
    assert normalize("$5,234.00") == "5 234 00"
    assert normalize("I don't know.") == "i don t know"


def test_is_idk_matches_common_forms():
    assert is_idk("I don't know")
    assert is_idk("i dont know")
    assert is_idk("I don’t know.")
    assert is_idk("I don't know the answer yet")
    assert not is_idk("The company does not disclose this")
    assert not is_idk("")


def test_gold_ngrams_returns_3plus_word_ngrams():
    ngrams = gold_ngrams("Apple reported revenue of $394 billion in fiscal 2022")
    assert "apple reported revenue" in ngrams
    assert "394 billion in" in ngrams
    # stop-word-only ngrams should be rejected
    assert "of the in" not in ngrams


def test_gold_ngrams_rejects_short_gold():
    assert gold_ngrams("Yes") == set()
    assert gold_ngrams("$5B") == set()


def test_any_ngram_in_chunks_substring_match():
    ngrams = {"apple reported revenue"}
    chunks_with = ["In fiscal 2022, Apple reported revenue of $394B."]
    chunks_without = ["Microsoft posted $198B in fiscal 2022."]
    assert any_ngram_in_chunks(ngrams, chunks_with)
    assert not any_ngram_in_chunks(ngrams, chunks_without)


def test_is_tersification_substring_both_directions():
    # Pred is inside gold; length ratio ok.
    assert is_tersification(
        pred="$53,803 million",
        gold="Cisco's total revenue for fiscal 2024 was $53,803 million.",
    )
    # Gold inside pred.
    assert is_tersification(
        pred="The answer, based on the 10-K, is $53,803 million.",
        gold="$53,803 million",
    )
    # Totally different strings.
    assert not is_tersification(pred="$50,000 million", gold="$53,803 million")
    # Substring but length ratio too small.
    assert not is_tersification(pred="10", gold="The revenue grew by 10 percent year-on-year.")


def test_category_codes_and_precedence_consistent():
    assert set(RULE_PRECEDENCE) <= set(CATEGORY_CODES)
    # A5 beats NF beats A3 beats A4 beats B4 beats B5 beats B2 beats B3
    order = RULE_PRECEDENCE
    assert order.index("A5") < order.index("NF")
    assert order.index("NF") < order.index("A3")
    assert order.index("A4") < order.index("B4")
