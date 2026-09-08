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

### Import the English corpus after vLLM filtering

The fixed English source corpus is
`nemotron-cc-high-propella-custom-eng`. Its provenance is:

1. `nemotron-cc-high-actual` was filtered across 21 source crawls to roughly
   5.5 million documents. A document was retained only when it had 200--20,000
   characters and passed the Propella conditions `content_ratio=complete_content`,
   `content_integrity=complete`, and `content_quality` equal to `excellent` or
   `high`.
2. Every tenth retained document was sampled, yielding exactly **557,017**
   source rows in `nemotron-cc-high-propella-eng`.
3. The custom vLLM filter was run over all 557,017 rows. It evaluates each
   document relative to its genre and rejects incoherent, boilerplate-heavy,
   spammy, templated, code-like, list-dominated, metadata-dominated, or
   otherwise low-quality text. It permits minor defects and short-form genres
   only when there is a substantial excellent section. The raw assessment is
   stored in `passes_filters`.
4. Only a valid assessment with `decision="PASS"` and
   `contains_substantial_high_quality_section=true` was retained. This yielded
   exactly **84,554** documents: 451,780 explicit FAIL rows, 20,679 malformed
   assessments, and 4 internally inconsistent assessments were excluded.
5. A seeded, source-level character-length-decile partition produced 15,000
   `dev` sources, 15,000 `test` sources, and 54,554 `train_01` sources.

The mass filter writes one output row for every input row and stores its JSON
assessment in `passes_filters`. The importer reconstructs the pass-only
canonical corpus from that output:

```bash
# Validate the full input and preview the accepted/rejected counts.
python scripts/import_filtered_custom_dataset.py \
  --input /path/to/completed-filter-output.jsonl \
  --dataset nemotron-cc-high-propella-custom-eng \
  --dry-run

# Write original.jsonl and filter_import_manifest.json atomically.
python scripts/import_filtered_custom_dataset.py \
  --input /path/to/completed-filter-output.jsonl \
  --dataset nemotron-cc-high-propella-custom-eng \
  --overwrite
```

The importer fails closed: null, malformed, FAIL, or internally inconsistent
assessments are excluded. Passing rows retain their original metadata plus the
parsed filter assessment and input line number under `filter_provenance`. The
manifest records the input/output paths and SHA-256 checksums, the exact
acceptance rule, and counts for every outcome. Duplicate passing source IDs,
invalid texts, and existing derived perturbation/score/partition artifacts are
hard errors.

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
method. The fixed English corpus uses 15,000 sources for each held-out
source-text partition and all 54,554 remaining sources for training:

```bash
# Preview only; this does not change original.jsonl.
python -m scripts.assign_dataset_partitions \
  --dataset nemotron-cc-high-propella-custom-eng \
  --dev-size 15000 --test-size 15000 --train-block-size 54554 \
  --seed 42 --dry-run

# Apply the reviewed plan atomically.
python -m scripts.assign_dataset_partitions \
  --dataset nemotron-cc-high-propella-custom-eng \
  --dev-size 15000 --test-size 15000 --train-block-size 54554 \
  --seed 42
```

The command adds `partition` to each original record and writes
`partition_manifest.json`. The completed English-corpus manifest records the
seed (`42`), character-length-decile stratification, input and output hashes,
and the exact `dev=15,000`, `test=15,000`, `train_01=54,554` assignments.

Generate a partition-selected LLM layer by supplying the model explicitly:

```bash
python -m scripts.generate_perturbations \
  --dataset nemotron-cc-high-propella-custom-eng \
  --source-layer 0 --method llm_sampled --run-id sampled-pilot-v1 \
  --model-path Qwen/Qwen3.5-27B \
  --source-partitions dev test train_01
```

The selected partition labels are recorded in the layer manifest. The same
option works when generating from a nonzero source layer because selection is
always resolved through the canonical original source.

## Generate one layer

Run one method per layer with `scripts/generate_perturbations.py`. The command
records the full effective configuration, deterministic seed, source selection,
candidate ancestry, method-specific edit evidence, and output-file checksum in
the perturbation manifest.

### Canonical methods

The only generative methods are:

| Method | Number of edits | Generation procedure |
| --- | --- | --- |
| `llm_single` | Exactly 1 | Sample one catalog operation, one target dimension, and one severity, then ask the LLM to apply it. |
| `llm_sampled` | 1--5 | Sample a length-conditioned number of catalog operations, dimensions, and severity, then ask the LLM to apply them. |
| `trad_single` | Exactly 1 | Sample one applicable operation from the five-operation traditional mix. |
| `trad_sampled` | 1--5 | Sample a length-conditioned number of traditional edits. |

For both sampled methods, the requested edit count is sampled uniformly from
`1..min(5, floor(character_length / 500))`. A text shorter than 500 characters
therefore receives one edit. This count is deterministic for a source candidate
and seed.

Run the two LLM workflows from originals as follows. `--model-path` identifies
the local or Hugging Face model served by the LLM runner.

```bash
# One LLM edit per original source.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_single \
  --run-id llm-single-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B

# One to five LLM edits per original source, conditional on text length.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method llm_sampled \
  --run-id llm-sampled-v1 \
  --target-layer 1 \
  --model-path Qwen/Qwen3.5-27B
```

Run the two traditional workflows from originals as follows. `--language en`
uses Lemminflect; other supported languages use UniMorph.

```bash
# Exactly one traditional edit per original source.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method trad_single \
  --run-id trad-single-v1 \
  --target-layer 1 \
  --language en

# One to five traditional edits per original source, conditional on text length.
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 0 \
  --method trad_sampled \
  --run-id trad-sampled-v1 \
  --target-layer 1 \
  --language en
```

To generate from an existing perturbation layer, provide the exact source
method and run ID. For example:

```bash
python scripts/generate_perturbations.py \
  --dataset my_dataset \
  --source-layer 1 \
  --source-method llm_sampled \
  --source-run-id llm-sampled-v1 \
  --method trad_sampled \
  --run-id trad-sampled-after-llm-v1 \
  --target-layer 2 \
  --language en
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

`llm_zero_shot` and the previous traditional pathways are archived historical
ablations, not valid generation methods.

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

For `llm_sampled`, the edit count is not a configuration option. Target
dimensions are sampled uniformly from the dimensions in the edit catalog. The
number selected is sampled uniformly from one through the smaller of the edit
count and the number of catalog dimensions. Severity is sampled uniformly and
deterministically from `weak`, `medium`, and `strong`; it is not a
configuration option. When a selected dimension has fewer catalog operations than the
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

### Traditional sampling and verification

`trad_single` always realizes one edit. `trad_sampled` uses the same
per-text length rule as `llm_sampled`: it samples uniformly from one through
`min(5, floor(character_length / 500))`. For every requested edit position,
the five operations begin equally likely. An operation that cannot make a
substantive change is removed from that position's pool and another remaining
operation is drawn uniformly. After a successful edit, the full five-operation
pool is restored, so later positions sample with replacement.

The only traditional configuration normally needed is `language` (plus the
global deterministic `seed`). Fixed edit counts and operation-restriction
flags are intentionally not exposed.

The five operations are equally likely at the start of every requested edit:

1. UniEval-style token-span repetition/insertion.
2. UniEval-style token-span deletion.
3. UniEval-style token-span shuffle.
4. Same-lemma finite-verb agreement corruption.
5. Same-lemma random morphology alteration, including a possible POS change.

Repetition and deletion always select at least one token; shuffle selects at
least two and must alter their order. Deletion that would empty a text is
inapplicable. Morphology edits require evidence that the replacement has the
same lemma but different recorded features. If an operation is inapplicable,
it is removed only from the current edit's sampling pool and another remaining
operation is sampled. A successful edit restores all five operations for the
next requested edit, so sampled edits are drawn with replacement.

## Score canonical candidates

Scores attach to exact candidates and are separated by method and score run:

```bash
python scripts/score_custom_dataset.py \
  --dataset-name my_dataset \
  --scoring-type bertscore_f1 \
  --scoring-run-id bertscore-v1 \
  --methods llm_sampled trad_sampled \
  --perturbation-run-ids sampled-dynamic-v1 trad-sampled-after-llm-v1 \
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
  --include-methods llm_sampled trad_sampled \
  --include-runs sampled-dynamic-v1 trad-sampled-after-llm-v1 \
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
  --datasets nemotron-cc-high-propella-custom-eng \
  --output-name en/trad_single_pilot_50k \
  --include-methods trad_single --include-runs trad-single-v1 --include-layers 1 \
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
`single_llm_ablation`, `sampled_llm_ablation`, and `traditional_comparison`.

## Legacy import

Historical folders are accepted only through explicit migration:

```bash
python scripts/import_legacy_dataset.py \
  --dataset my_dataset \
  --source-directory perturbed_layers \
  --method llm_single \
  --run-id legacy-import
```

New generation, scoring, and HF construction use only canonical repositories
and manifests. Deprecated scripts are not extension points.
