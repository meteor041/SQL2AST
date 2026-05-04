import json

import pytest

from src.utils import cscsql_prompt


class _FakeProcessDatasetModule:
    def __init__(self):
        self.sample_calls = []
        self.prepare_calls = []
        self.searcher_calls = []
        self.retrieve_calls = []

    def sample_table_values(self, db_file_path, table_names, limit_num):
        self.sample_calls.append((db_file_path, table_names, limit_num))
        return {"example.table": ["v1", "v2"]}

    class LuceneSearcher:
        def __init__(self, path):
            self.path = path

    def prepare_input_output_pairs(
        self,
        data,
        ek_key,
        db_id2relevant_hits,
        sampled_db_values_dict,
        db_info,
        source,
        output_key,
        mode,
    ):
        self.prepare_calls.append(
            {
                "data": data,
                "ek_key": ek_key,
                "db_id2relevant_hits": db_id2relevant_hits,
                "sampled_db_values_dict": sampled_db_values_dict,
                "db_info": db_info,
                "source": source,
                "output_key": output_key,
                "mode": mode,
            }
        )
        return {"input_seq": "PROMPT"}

    def obtain_n_grams(self, question_text, max_n):
        return [question_text[:3]]

    def retrieve_relevant_hits(self, searcher, queries):
        self.retrieve_calls.append((searcher.path, tuple(queries)))
        return {query: [{"id": "demo_table.id-**-0", "contents": "v1"}] for query in queries}


@pytest.fixture(autouse=True)
def _clear_prompt_caches():
    cscsql_prompt._load_db_info_map.cache_clear()
    cscsql_prompt._sampled_db_values.cache_clear()
    yield
    cscsql_prompt._load_db_info_map.cache_clear()
    cscsql_prompt._sampled_db_values.cache_clear()


def test_build_cscsql_prompt_delegates_to_cscsql_module(tmp_path, monkeypatch):
    db_root = tmp_path / "train_databases"
    db_root.mkdir()
    (db_root / "demo.sqlite").write_text("", encoding="utf-8")
    (db_root / "train_tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "demo",
                    "table_names_original": ["demo_table"],
                    "table_names": ["demo_table"],
                    "column_names_original": [[-1, "*"], [0, "id"]],
                    "column_names": [[-1, "*"], [0, "id"]],
                    "column_types": ["text", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    fake_module = _FakeProcessDatasetModule()
    monkeypatch.setattr(
        cscsql_prompt,
        "_load_process_dataset_module",
        lambda: fake_module,
    )

    prompt = cscsql_prompt.build_cscsql_prompt(
        question="How many rows?",
        db_id="demo",
        database_root=db_root,
        evidence="Count all rows.",
        gold_sql="SELECT COUNT(*) FROM demo_table",
        value_limit_num=4,
        source="ches",
        mode="eval",
    )

    assert prompt == "PROMPT"
    assert fake_module.sample_calls == [
        (str(db_root / "demo.sqlite"), ["demo_table"], 4)
    ]
    assert len(fake_module.prepare_calls) == 1

    call = fake_module.prepare_calls[0]
    assert call["data"]["question"] == "How many rows?"
    assert call["data"]["evidence"] == "Count all rows."
    assert call["data"]["SQL"] == "SELECT COUNT(*) FROM demo_table"
    assert call["ek_key"] == "evidence"
    assert call["db_id2relevant_hits"] is None
    assert call["sampled_db_values_dict"] == {"example.table": ["v1", "v2"]}
    assert call["db_info"]["db_id"] == "demo"
    assert call["source"] == "ches"
    assert call["output_key"] == "SQL"
    assert call["mode"] == "eval"


def test_resolve_tables_json_path_supports_sibling_dev_tables(tmp_path):
    db_root = tmp_path / "dev_databases"
    db_root.mkdir()
    metadata_path = tmp_path / "dev_tables.json"
    metadata_path.write_text("[]", encoding="utf-8")

    resolved = cscsql_prompt.resolve_tables_json_path(db_root)

    assert resolved == metadata_path


def test_build_cscsql_prompt_uses_relevant_hits_when_index_exists(tmp_path, monkeypatch):
    db_root = tmp_path / "train_databases"
    db_root.mkdir()
    (db_root / "demo.sqlite").write_text("", encoding="utf-8")
    (db_root / "train_tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "demo",
                    "table_names_original": ["demo_table"],
                    "table_names": ["demo_table"],
                    "column_names_original": [[-1, "*"], [0, "id"]],
                    "column_names": [[-1, "*"], [0, "id"]],
                    "column_types": ["text", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    index_root = tmp_path / "train_db_contents_index" / "demo"
    index_root.mkdir(parents=True)

    fake_module = _FakeProcessDatasetModule()
    monkeypatch.setattr(
        cscsql_prompt,
        "_load_process_dataset_module",
        lambda: fake_module,
    )
    monkeypatch.setenv("CSC_SQL_DB_CONTENT_INDEX_PATH", str(index_root.parent))

    prompt = cscsql_prompt.build_cscsql_prompt(
        question="How many rows?",
        db_id="demo",
        database_root=db_root,
        evidence="Only count active ones.",
        gold_sql="SELECT COUNT(*) FROM demo_table",
        value_limit_num=4,
        source="ches",
        mode="eval",
    )

    assert prompt == "PROMPT"
    assert len(fake_module.prepare_calls) == 1
    call = fake_module.prepare_calls[0]
    assert call["db_id2relevant_hits"] is not None
    assert "demo" in call["db_id2relevant_hits"]
    assert fake_module.retrieve_calls
