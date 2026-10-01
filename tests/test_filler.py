from bench_cache.filler import _VOCAB, TOKENS_PER_WORD, filler


def test_filler_is_deterministic_and_starts_with_its_tag() -> None:
    a = filler(500, tag="k/t1", seed=3)
    assert a == filler(500, tag="k/t1", seed=3)
    assert a != filler(500, tag="k/t1", seed=4)
    assert a.startswith("[k/t1]\n")
    assert filler(500, tag="k/t2", seed=3).split("\n", 1)[1] == a.split("\n", 1)[1]


def test_filler_is_lorem_ipsum_sized_by_tokens_per_word() -> None:
    body = filler(3000, seed=0)
    words = body.split()
    assert len(words) == round(3000 / TOKENS_PER_WORD)
    assert body.startswith(body[0].upper()) and body.rstrip().endswith(".")
    assert {w.strip(",.").lower() for w in words} <= set(_VOCAB)


def test_zero_tokens_is_just_the_tag() -> None:
    assert filler(0, tag="t") == "[t]\n"
