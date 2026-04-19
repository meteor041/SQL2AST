from src.build_pairs import build_pairs_for_sample


def _pairs(scored, max_pairs=0, min_margin=0.0):
    return build_pairs_for_sample(
        scored=scored,
        gold_sql="SELECT gold",
        prompt_str="prompt",
        sample_id=1,
        db_id="db",
        max_pairs=max_pairs,
        min_margin=min_margin,
    )


def test_builds_all_correct_wrong_pairs_even_when_wrong_is_closer():
    pairs = _pairs([
        ("correct_a", 0.4, True),
        ("correct_b", 0.6, True),
        ("wrong_a", 0.1, False),
        ("wrong_b", 0.8, False),
    ])

    sql_pairs = {(p.chosen, p.rejected) for p in pairs}

    assert ("correct_a", "wrong_a") in sql_pairs
    assert ("correct_a", "wrong_b") in sql_pairs
    assert ("correct_b", "wrong_a") in sql_pairs
    assert ("correct_b", "wrong_b") in sql_pairs
    assert ("wrong_a", "correct_a") not in sql_pairs
    assert ("wrong_a", "correct_b") not in sql_pairs


def test_builds_wrong_wrong_pairs_by_distance():
    pairs = _pairs([
        ("wrong_near", 0.2, False),
        ("wrong_mid", 0.5, False),
        ("wrong_far", 0.9, False),
    ])

    sql_pairs = {(p.chosen, p.rejected) for p in pairs}

    assert ("wrong_near", "wrong_mid") in sql_pairs
    assert ("wrong_near", "wrong_far") in sql_pairs
    assert ("wrong_mid", "wrong_far") in sql_pairs
    assert ("wrong_far", "wrong_near") not in sql_pairs


def test_skips_correct_correct_and_equal_distance_wrong_wrong_pairs():
    pairs = _pairs([
        ("correct_a", 0.1, True),
        ("correct_b", 0.2, True),
        ("wrong_a", 0.5, False),
        ("wrong_b", 0.5, False),
    ])

    sql_pairs = {(p.chosen, p.rejected) for p in pairs}

    assert ("correct_a", "correct_b") not in sql_pairs
    assert ("correct_b", "correct_a") not in sql_pairs
    assert ("wrong_a", "wrong_b") not in sql_pairs
    assert ("wrong_b", "wrong_a") not in sql_pairs
