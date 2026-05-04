from src.train_sft import split_train_eval_data


def test_split_train_eval_data_is_deterministic():
    records = [{"sample_id": idx} for idx in range(100)]

    train_a, eval_a = split_train_eval_data(records, eval_ratio=0.05, seed=42)
    train_b, eval_b = split_train_eval_data(records, eval_ratio=0.05, seed=42)

    assert len(train_a) == 95
    assert len(eval_a) == 5
    assert train_a == train_b
    assert eval_a == eval_b


def test_split_train_eval_data_disables_eval_when_ratio_is_zero():
    records = [{"sample_id": idx} for idx in range(10)]

    train_split, eval_split = split_train_eval_data(records, eval_ratio=0.0, seed=42)

    assert train_split == records
    assert eval_split == []
