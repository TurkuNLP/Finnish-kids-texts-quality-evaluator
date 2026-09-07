# Perturbation and dataset workflow

This document describes the canonical local workflow. It does not depend on
cluster batch jobs. Paths are relative to the repository root.

## Custom-dataset layout

Each custom dataset starts with `original.jsonl`:

```text
data/custom_datasets/<dataset>/
  original.jsonl
  perturbations/
    perturbation_manifest.json
    <method>/<run_id>/<target_layer>.jsonl
  scores/
    <scoring_method>/<scoring_run_id>.jsonl
    <scoring_method>/<scoring_run_id>.errors.jsonl
    <scoring_method>/<scoring_run_id>.metadata.json
```

An original row requires `custom_id` and `text`. String and integer source IDs
are accepted and normalized to strings. The source ID identifies a document;
`candidate_id` identifies one exact original or perturbation candidate.

The manifest is the authoritative layer index. Directory scanning is not used
to infer layers. Every candidate records its method, run, source and target
layers, and exact `parent_candidate_id`. A new layer may therefore start from
an original or any existing layer, including one produced by another method.

### Fixed source partitions for staged experiments

For a large custom dataset, assign source-level partitions once before
generation. This preserves train/dev/test isolation across every perturbation
method and makes 50k, 100k, and larger training subsets nested rather than
independently resampled:

```bash
# Preview only; this does not change original.jsonl.
python -m scripts.assign_dataset_partitions \
  --dataset nemotron-cc-high-propella-eng \
  --dev-size 10000 --test-size 10000 --train-block-size 50000 \
  --seed 42 --dry-run

# Apply the reviewed plan atomically.
python -m scripts.assign_dataset_partitions \
  --dataset nemotron-cc-high-propella-eng \
  --dev-size 10000 --test-size 10000 --train-block-size 50000 \
  --seed 42
```

The command adds `partition` to each original record and writes
`partition_manifest.json`. The standard high-propella plan has `dev` and
`test` partitions of 10k sources each, ten `train_01`–`train_10` blocks of
50k sources, and `train_remainder`. Assignment is deterministic and
stratified by source-document character-length decile.

Generate expensive perturbations only for the needed pilot scope:

```bash
python -m scripts.generate_perturbations \
  --dataset nemotron-cc-high-propella-eng \
  --source-layer 0 --method llm_sampled --run-id sampled-pilot-v1 \
  --source-partitions dev test train_01
```

The selected partition labels are recorded in the layer manifest. The same
option works when generating from a nonzero source layer because selection is
always resolved through the canonical original source.

## Generate one layer

Use `scripts/generate_perturbations.py` for a single generation:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_sampled \
  --run-id sampled-dynamic-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B
```

To perturb an existing layer, identify its method and run:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 1 \
  --source-method llm_sampled \
  --source-run-id sampled-dynamic-v1 \
  --method trad_multi \
  --run-id trad-after-llm-v1 \
  --target-layer 2 \
  --language en \
  --n-edits 3
```

### CSC batch-job examples

The wrappers in `updated_sbatch_jobs/` use the same canonical commands. For a
sampled-LLM pilot, submit the dedicated wrapper with environment variables:

```bash
DATASET=my_dataset RUN_ID=sampled-dynamic-v1 LIMIT=1000 \
  sbatch updated_sbatch_jobs/pert_job.sh
```

`pert_job.sh` deliberately does not expose fixed edit-count, target-dimension,
or severity settings: `llm_sampled` samples all three per source text. Set
`MAX_MODEL_LEN` only to change the LLM context ceiling; set `MAX_RETRIES` only
when intentionally changing the default three retries. The generic wrapper is
equivalent when explicit command-line arguments are preferable:

```bash
sbatch updated_sbatch_jobs/generate_perturbations.sh \
  --dataset my_dataset --source-layer 0 --method llm_sampled \
  --run-id sampled-dynamic-v1 --model-path Qwen/Qwen3.5-27B
```

`target_layer` defaults to `source_layer + 1`. A perturbed source requires
both `source_method` and `source_run_id`. Existing outputs are protected; use
`--overwrite` only when replacement is intentional. A reusable method config
can be passed with `--method-config`; explicit CLI values take precedence.

### Recover failed inputs without regenerating successful ones

If a completed layer contains skipped inputs, rerun the original command with
the same dataset, source, method, run ID, partition selection, and generation
configuration, adding `--retry-failed`:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset --source-layer 0 --method llm_sampled \
  --run-id sampled-dynamic-v1 --model-path Qwen/Qwen3.5-27B \
  --retry-failed
```

This retries only source candidates that still lack an output, preserves all
existing candidate rows, and atomically replaces the same layer file and its
manifest entry with the merged result. It does not create another run. Each
retry records its effective seed, attempted and recovered counts, and any
remaining failures in that layer's manifest. `--retry-failed` cannot be used
with `--overwrite`.

For the sampled-LLM wrapper, use the same environment values with
`RETRY_FAILED=1`; for example, `DATASET=my_dataset RUN_ID=sampled-dynamic-v1
RETRY_FAILED=1 sbatch updated_sbatch_jobs/pert_job.sh`.

### Available methods

| Method | Meaning |
| --- | --- |
| `llm_zero_shot` | Fixed-prompt LLM perturbation retained for the ablation. |
| `llm_sampled` | Samples edit types, fluency dimensions, and severity for the LLM prompt. |
| `unieval` | UniEval-style insertion, deletion, and shuffle noise. |
| `trad_single` | One applicable traditional perturbation. |
| `trad_multi` | Multiple applicable traditional perturbations. |
| `unieval_trad` | UniEval noise followed by traditional edits. |

The traditional implementation is multilingual. English inflection uses
Lemminflect; other supported languages use UniMorph. Language-specific
operations are excluded where they are not applicable. The former
`unieval_summinflect` variant is represented by `unieval_trad`.

### Sampled LLM options

```json
{
  "model": "Qwen/Qwen3.5-27B",
  "language": "english",
  "edit_catalog": "data/perturbation_prompts/english/edit_types.jsonl",
  "require_dimension_coverage": true,
  "weights": {
    "unnecessary_circumlocution": 1.0,
    "odd_collocation": 1.0
  },
  "seed": 42
}
```

For `llm_sampled`, the edit count is sampled deterministically from each
source text's character length; it is not a configuration option. Target
dimensions are sampled uniformly from the dimensions in the edit catalog. The
number selected is sampled uniformly from one through the smaller of the edit
count and the number of catalog dimensions. Severity is sampled uniformly and
deterministically from `weak`, `medium`, and `strong`; it is not a
configuration option. vLLM and model dependencies load only when an LLM method
is executed. When a selected dimension has fewer catalog operations than the
sampled edit count, operations may repeat so that the requested count is
preserved.
Each sampling decision uses its own deterministic seed derived from the
candidate identity, so changing one sampling stage does not reshuffle the
others.

LLM outputs that are empty, unchanged, or longer than their requested
character limit are retried up to three times. If the final output is otherwise
valid but still over the length limit, it is retained and marked with
`length_limit_exceeded`, `output_chars`, `max_output_chars`, and
`retry_attempts` in its candidate metadata.

### Edit-count provenance

Every generated candidate row has an `edit_count` field. For both
`llm_sampled` and traditional methods, it is the number of recorded
`perturbation_edits` and must equal the length of that array. In particular,
it is not the number of target dimensions or a severity level. For
`llm_sampled`, it is the per-text sampled number of required edit operations;
for traditional methods, it is the number of operations that actually made a
change. Historical rows without this field remain readable as `null`.

### Traditional options

```json
{
  "language": "en",
  "n_noise": 1,
  "n_edits": 3,
  "operations": ["jumble", "subject_verb_dis", "typos"],
  "seed": 42
}
```

`operation` selects the operation for `trad_single`. `operations` restricts
the pool for `trad_multi` and `unieval_trad`.

## Score canonical candidates

Scores attach to exact candidates and are separated by method and score run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1 \
  --methods llm_sampled trad_multi \
  --perturbation-run-ids sampled-dynamic-v1 trad-after-llm-v1 \
  --target-layers 1 2 \
  --reference-policy parent
```

`reference-policy original` uses the source original; `parent` uses the exact
parent candidate. Score files retain both candidate and reference identities.
All stored scores are higher-is-better.

The custom-dataset scorer also supports `geval_gpt54mini_fluency`, which uses
the existing G-Eval scorer with the pinned GPT-5.4-mini snapshot, and
`menlo_themis_fluency`, which uses the Themis vLLM scorer with the MENLO
five-point fluency rubric. Both score candidates only; their prompt, rubric,
model, parser, and decoding settings are retained in score-run metadata.

For example, G-Eval scoring uses the pinned GPT-5.4-mini judge and an optional
response cache:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type geval_gpt54mini_fluency \
  --scoring-run-id geval-gpt54mini-v1 \
  --geval-cache-path data/evals/my_dataset_geval_cache.json
```

The Themis/MENLO scorer runs locally through vLLM:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type menlo_themis_fluency \
  --scoring-run-id menlo-themis-v1 \
  --themis-model-name PKU-ONELab/Themis \
  --themis-tensor-parallel-size 1
```

These custom-dataset commands score candidate text only. They do not pass the
original or parent text to either judge, even when `--reference-policy` is
used for score provenance. The reference policy controls stored candidate
identity only.

## Build a Hugging Face dataset

The standalone builder accepts canonical CLI fields or an `HFBuildSpec` JSON
object such as `configs/hf_build.example.json`.

```bash
python scripts/build_hf_dataset.py \
  --datasets my_dataset \
  --output-name my_dataset_hf \
  --include-methods llm_sampled trad_multi \
  --include-runs sampled-dynamic-v1 trad-after-llm-v1 \
  --include-layers 1 2 \
  --composition balanced \
  --pair-policy parent_child \
  --score-names bertscore_f1 \
  --score-run-ids bertscore-v1
```

For a partitioned one-method 50k pilot, fixed `dev` and `test` sources are
included automatically and `train_01` is selected explicitly:

```bash
python -m scripts.build_hf_dataset \
  --datasets nemotron-cc-high-propella-eng \
  --output-name en/unieval_pilot_50k \
  --include-methods unieval --include-runs unieval-v1 --include-layers 1 \
  --train-partitions 1
```

`--train-partitions 1 2` builds the nested 100k version. Partition numbers
must be the contiguous prefix `1..N`; ratio splitting and `downsample_size`
are not used in this mode. The builder rejects a selected source with no
candidate matching the method/run/layer filters.

Equivalent config-based use:

```bash
python scripts/build_hf_dataset.py --config configs/hf_build.example.json
```

The original is included automatically. Method, run, and layer filters are
independent; omitting one includes all values for that field.

Composition is performed separately for each source document:

| Policy | Selection |
| --- | --- |
| `all` | Every selected candidate. |
| `balanced` | The same candidate count from each available method. |
| `weighted` | Sampling without replacement using `method_weights`. |
| `source_exclusive` | Assign each source document to one method. |
| `fixed_per_source` | Keep up to `samples_per_source` candidates per method. |

Pair policies are applied only after source-safe splitting:

| Policy | Result |
| --- | --- |
| `none` | One aligned candidate group per source document. |
| `parent_child` | One pair for each selected exact graph edge. |
| `original_only` | Original versus each selected perturbation. |
| `all_unequal_layers` | Every same-source pair with different target layers. |
| `cross_source_unmatched` | Different-source, unequal-layer pairs within one dataset, with reuse bounded by `reuse_limit`. |

`score_names` selects scoring methods. If multiple score runs exist for a
selected candidate and method, specify `score_run_ids`; ambiguity is rejected.

The output is a `DatasetDict` with `train`, `dev`, and `test`. Sources are
split before composition or pairing. Rows preserve aligned text, target layer,
candidate ID, method, run, parent, source-layer, and score arrays.

## Run a complete configured workflow

`scripts/prepare_dataset.py` uses the same generation and HF contracts. Copy
`configs/workflow.example.json`, adjust it, then run:

```bash
python scripts/prepare_dataset.py generate --config configs/workflow.example.json
python scripts/prepare_dataset.py build-hf --config configs/workflow.example.json
python scripts/prepare_dataset.py run-all --config configs/workflow.example.json
```

Generation options belong inside each entry's `config` object:

```json
{
  "schema_version": 1,
  "dataset": "my_dataset",
  "dataset_root": "data/custom_datasets",
  "seed": 42,
  "generations": [
    {
      "method": "llm_sampled",
      "run_id": "sampled-dynamic-v1",
      "source_layer": 0,
      "target_layer": 1,
      "config": {
        "model": "Qwen/Qwen3.5-27B"
      }
    }
  ],
  "hf": {
    "output_name": "my_dataset_sampled",
    "include_methods": ["llm_sampled"],
    "include_runs": ["sampled-dynamic-v1"],
    "include_layers": [1],
    "composition": "all",
    "pair_policy": "none"
  }
}
```

Within a workflow, `hf.datasets` defaults to the top-level dataset and
`hf.seed` defaults to the workflow seed. Presets are available for
`zero_shot_ablation`, `sampled_llm_ablation`, and `traditional_comparison`.

## Legacy import

Historical folders are accepted only through explicit migration:

```bash
python scripts/import_legacy_dataset.py \
  --dataset my_dataset \
  --source-directory perturbed_layers \
  --method llm_zero_shot \
  --run-id legacy-import
```

New generation, scoring, and HF construction use only canonical repositories
and manifests. Deprecated scripts are not extension points.
