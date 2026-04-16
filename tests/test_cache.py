"""Tests for src.cache.Cache — covers hit, miss, roundtrip, key uniqueness, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cache import Cache


@pytest.fixture()
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache")


def test_miss_returns_none(cache: Cache) -> None:
    assert cache.get("nonexistent_key") is None


def test_put_then_get_roundtrip(cache: Cache) -> None:
    key = cache.key_for("gemini-1.5-flash", "hello", {"temperature": 0.0})
    value = {"answer": "world", "tokens": 3}
    cache.put(key, value)
    assert cache.get(key) == value


def test_hit_after_write(cache: Cache) -> None:
    key = cache.key_for("m", "p", {})
    cache.put(key, {"x": 1})
    assert cache.get(key) is not None


def test_key_for_is_deterministic() -> None:
    k1 = Cache.key_for("m", "p", {"a": 1, "b": 2})
    k2 = Cache.key_for("m", "p", {"b": 2, "a": 1})  # key order differs
    assert k1 == k2


def test_different_params_produce_different_keys() -> None:
    base = Cache.key_for("m", "p", {"temperature": 0.0})
    assert Cache.key_for("m", "p", {"temperature": 0.7}) != base
    assert Cache.key_for("m2", "p", {"temperature": 0.0}) != base
    assert Cache.key_for("m", "p2", {"temperature": 0.0}) != base


def test_key_length_and_shape() -> None:
    k = Cache.key_for("m", "p", {})
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_persistence_across_instances(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    c1 = Cache(cache_dir)
    key = c1.key_for("m", "p", {"n": 5})
    c1.put(key, {"payload": [1, 2, 3]})

    c2 = Cache(cache_dir)
    assert c2.get(key) == {"payload": [1, 2, 3]}


def test_file_layout_uses_two_char_prefix(cache: Cache) -> None:
    key = cache.key_for("m", "p", {})
    cache.put(key, {"v": 1})
    expected = cache.cache_dir / key[:2] / f"{key[2:]}.json"
    assert expected.is_file()


def test_overwrite_is_last_writer_wins(cache: Cache) -> None:
    key = cache.key_for("m", "p", {})
    cache.put(key, {"v": 1})
    cache.put(key, {"v": 2})
    assert cache.get(key) == {"v": 2}


def test_corrupt_file_returns_none(cache: Cache) -> None:
    key = cache.key_for("m", "p", {})
    path = cache._path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert cache.get(key) is None


def test_unicode_values_roundtrip(cache: Cache) -> None:
    key = cache.key_for("m", "คำถาม", {"lang": "th"})
    value = {"answer": "สวัสดี", "emoji": "✓"}
    cache.put(key, value)
    assert cache.get(key) == value
