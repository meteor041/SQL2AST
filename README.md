# SQL Evaluation and AST Export

This repository has two main scripts:

- `eval.py`: evaluate generated SQL candidates against gold SQL by executing them on SQLite databases.
- `sql_to_ast.py`: convert SQL in JSON files into sqlglot AST JSON.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The required parser dependency is `sqlglot`.

## 2. Configure `.location`

Create or edit `.location` in the repository root:

```ini
TRAIN_DATA_PATH=/data/huwenp/emb/data/ches/train.json
TRAIN_DATABASE_PATH=/data/huwenp/emb/data/ches/train_databases
SQL_PATH=/data/huwenp/emb/data/ches/candidates/16_qwen7B_train
EVAL_OUTPUT_PATH=eval_results
```

Required keys:

- `TRAIN_DATA_PATH`: train JSON containing gold SQL and `db_id`.
- `TRAIN_DATABASE_PATH`: directory containing SQLite databases.
- `SQL_PATH`: candidate SQL JSON file or directory.

Optional key:

- `EVAL_OUTPUT_PATH`: output directory for `eval.py`. Defaults to `eval_results` if not set.

Candidate SQL files should be named like:

```text
544_movies_4.json
```

Each file should contain records with an `all_sqls` list.

## 3. Run `eval.py`

Evaluate all candidate files configured by `.location`:

```bash
python eval.py
```

Write results to a custom directory:

```bash
python eval.py --output-dir eval_results
```

Evaluate a specific candidate file or directory:

```bash
python eval.py --input-dir data/1_movie_platform.json --output-dir eval_results
```

Limit the number of files for a quick smoke test:

```bash
python eval.py --limit 5 --output-dir eval_results
```

Use more CPU workers:

```bash
python eval.py --num-cpus 4 --output-dir eval_results
```

Compare result rows ignoring row order:

```bash
python eval.py --ignore-order --output-dir eval_results
```

Expected output:

- Per-query files such as `eval_results/544_movies_4_eval.json`
- Summary file: `eval_results/summary.json`

Each per-query eval file contains:

- `correct_set`: SQL candidates whose execution result matches gold SQL.
- `wrong_set`: SQL candidates whose execution result does not match gold SQL.
- `file_errors`: file-level evaluation errors, if any.

## 4. Run `sql_to_ast.py`

Convert one eval result file to AST JSON:

```bash
python sql_to_ast.py eval_results/544_movies_4_eval.json -o eval_results_ast/544_movies_4_ast.json --dialect sqlite
```

Convert all eval result files:

```bash
python sql_to_ast.py eval_results -o eval_results_ast --pattern '*_eval.json' --dialect sqlite
```

Deduplicate SQL strings inside each set before parsing:

```bash
python sql_to_ast.py eval_results -o eval_results_ast --pattern '*_eval.json' --dialect sqlite --deduplicate
```

Print a single converted file to stdout:

```bash
python sql_to_ast.py eval_results/544_movies_4_eval.json --dialect sqlite
```

Expected output:

```json
{
  "metadata": {
    "sample_id": 544,
    "db_id": "movies_4",
    "source_file": "eval_results/544_movies_4_eval.json",
    "gold_count": 1,
    "correct_count": 5,
    "wrong_count": 7,
    "parsed_gold_count": 1,
    "parsed_correct_count": 5,
    "parsed_wrong_count": 6,
    "parsed_count": 12,
    "error_count": 1
  },
  "gold": [
    {
      "sql": "...",
      "normalized_sql": "..."
    }
  ],
  "correct": [
    {
      "sql": "...",
      "normalized_sql": "..."
    }
  ],
  "wrong": [
    {
      "sql": "...",
      "normalized_sql": "..."
    }
  ],
  "errors": []
}
```

Each `normalized_sql` is produced by parsing with sqlglot and re-serializing.
To compute distance metrics, re-parse it with `sqlglot.parse_one(normalized_sql)`.

For directory input, files are mapped like this:

```text
eval_results/544_movies_4_eval.json -> eval_results_ast/544_movies_4_ast.json
```

## 5. Legacy `all_sqls` normalization

`sql_to_ast.py` still supports the old sampled SQL format directly:

```bash
python sql_to_ast.py data/1_movie_platform.json -o data_ast/1_movie_platform_ast.json --dialect sqlite
```

This preserves the original records and adds `normalized_sqls` (a list of
`{"sql": "...", "normalized_sql": "..."}` objects, one per unique SQL string).

## 6. Typical Full Workflow

```bash
pip install -r requirements.txt

python eval.py --output-dir eval_results

python sql_to_ast.py eval_results -o eval_results_ast --pattern '*_eval.json' --dialect sqlite
```

After this workflow:

- `eval_results/` contains execution-evaluation results.
- `eval_results_ast/` contains gold/correct/wrong SQL with normalized forms.
