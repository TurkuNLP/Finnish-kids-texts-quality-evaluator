# Repository architecture

The project creates controlled fluency perturbations and uses them to train and
evaluate candidate-only text quality scorers.

## End-to-end flow

```text
custom original.jsonl
  -> canonical perturbation repository and manifest
  -> optional candidate score runs
  -> source-safe HF DatasetDict
  -> regression or pairwise FE training rows
  -> trained candidate-only scorer
  -> shared evaluation suite
```

The canonical candidate graph is the boundary shared by generation, scoring,
and dataset construction. No stage reconstructs identity from filenames or
text equality.

## Fixed English source corpus

The primary English source corpus is
`nemotron-cc-high-propella-custom-eng`. It is distinct from the human-labeled
English evaluation suite. Its 84,554 documents were produced from a 557,017-row
every-tenth-document sample of an approximately 5.5-million-document,
21-crawl `nemotron-cc-high-actual` collection. Before sampling, documents had
to be 200--20,000 characters and satisfy Propella
`content_ratio=complete_content`, `content_integrity=complete`, and
`content_quality in {excellent, high}` filters. A custom genre-aware vLLM
quality filter then retained only valid PASS assessments with a substantial
high-quality section.

The corpus has immutable source-level partitions: 15,000 `dev`, 15,000 `test`,
and 54,554 `train_01` documents. Assignment uses seed 42 and document
character-length deciles. The import and partition manifests retain input/output
hashes, outcome counts, and hashes of the source IDs assigned to each partition.

## Canonical repository

For each custom dataset:

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

The manifest lists every committed method/run/layer, its source layer, source
method and run, configuration hash, content hash, counts, and output path.
Candidate records carry:

- dataset and stable base-text identity;
- globally stable candidate identity;
- perturbation method, source family, and run;
- source and target layers;
- exact parent candidate identity;
- method-specific generation metadata.

An original is represented as a stable candidate at layer 0. Every perturbed
candidate has exactly one parent. Repository validation rejects missing
parents, cross-document ancestry, duplicate candidate IDs, path/provenance
disagreement, and content-hash mismatches.

## Perturbation layer

`clumsification_code/perturbations/` contains the reusable method registry and
generation service. `scripts/generate_perturbations.py` is the single-layer
client; `scripts/prepare_dataset.py` can execute several declared generations.

Canonical LLM method names are `llm_single`, `llm_sampled`; the only active
traditional names are `trad_single` and `trad_sampled`. Both sample from the
same five-operation mix: UniEval-style repetition, deletion, and shuffle;
agreement corruption; and random same-lemma morphology. LLM implementations share a
runner boundary and load vLLM only when needed. The sampled method uses a
versioned JSONL edit catalog and deterministically samples, for each
candidate, the edit count, target dimensions, operations, and severity from
its stable seed.

Traditional perturbation is multilingual: English morphology uses
Lemminflect and other supported languages use UniMorph. The registry exposes
one interface regardless of implementation language.

Generation requests identify their source by `(source_method, source_run_id,
source_layer)`, or by layer 0 for originals. Consequently, layers can be
continued within one method or chained across methods without copying files.

## Candidate scoring

`clumsification_code/scoring/` loads tasks from canonical manifests and writes
versioned score records. Each record identifies:

```text
(dataset, base_text, candidate, perturbation method/run,
 scoring method/run, exact reference candidate)
```

Method, perturbation run, and target-layer filters are applied before scoring.
Reference policy is explicit: either the original candidate or the exact
parent. Score direction is normalized to higher-is-better. Errors and run
metadata are stored beside, but separately from, successful scores.

## HF composition and pairing

`clumsification_code/data/hf_dataset.py` traverses the repository graph. It
does not use historical layer directories. `HFBuildSpec` controls:

- datasets, perturbation methods, runs, and target layers;
- composition policy and method weights;
- pair policy and reuse limit;
- scoring methods and score runs;
- source-safe split ratios, downsampling, and seed.

Splitting occurs on `(dataset_name, base_text_id)` before composition and
pairing. This also applies to cross-source unmatched pairs, so all candidates
derived from one original remain in one split.

Grouped HF rows contain aligned arrays for text, layer, candidate ID, method,
run, parent ID, source layer/method/run, and requested scores. Exact provenance
therefore survives selection and shuffling.

`scripts/build_hf_dataset.py` and the HF stage of
`scripts/prepare_dataset.py` call the same `build_hf_dataset(HFBuildSpec, ...)`
service. CLI and JSON configuration are two front ends to one implementation.

## Training boundary

`clumsification_code/data/flattening.py` validates source isolation and turns
grouped HF rows into explicit training rows:

- regression: one candidate and one finite scalar target;
- pairwise: one chosen/rejected pair, with lower perturbation layer treated as
  the preferred candidate for layer-based supervision.

The FE model is candidate-only at inference. Both objectives use the same
encoder, resolved pooling rule, and scalar linear head. Teacher scores,
sources, references, method names, and layer identities are supervision or
audit metadata, never inference inputs.

`scripts/train_fe_model.py` is the canonical FE training entrypoint and uses
the Hugging Face Trainer. Evaluation adapters under
`clumsification_code/evals/` expose shared candidate-scoring interfaces to the
benchmark runner.

### Full pairwise training recipe

`updated_sbatch_jobs/train_fe_pairwise_full.sh` is the production recipe for
the approximately 390k-pair UniEval-style training dataset. It trains
`Qwen/Qwen3-Embedding-0.6B` with last-token pooling selected automatically by
the backbone profile, FlashAttention 2, and a 32,768-token limit. On four
GPUs, its per-device batch size of 32 and accumulation of 1 yield a global
batch of 128 pairs.

Validation runs every 39 updates (4,992 pairs) and checkpoints are saved every
195 updates (24,960 pairs), so each retained checkpoint has a fresh validation
metric. Training has a three-epoch upper bound and ends earlier after three
successive saved checkpoints fail to improve pairwise validation accuracy.
The job retains up to 50 model-only checkpoints, sufficient for the full
three-epoch ceiling and inexpensive enough to keep for the external evaluation
suite. Model-only checkpoints intentionally omit optimizer and scheduler state:
they can be evaluated with `load_fe_model`, but are not resumable training
checkpoints.

```bash
sbatch updated_sbatch_jobs/train_fe_pairwise_full.sh \
  /absolute/path/to/unieval_pairwise_dataset \
  outputs/fe_qwen3_unieval_pairwise_full
```

### Staged one-method pilot

For one-method pilots, `updated_sbatch_jobs/train_fe_pairwise_pilot.sh` trains
on the fixed `train_01` source partition. In the current English corpus this
is 54,554 sources, alongside separate 15,000-source `dev` and `test`
source-text partitions. With one original and one selected perturbation per
source, the job executes 390 updates (49,920 pairs)
and saves five checkpoints at 9,984-pair intervals. It uses FlashAttention 2,
32 train pairs/GPU, and 48 evaluation pairs/GPU. Validation begins after 20k
pairs and stopping requires three saved checkpoints without at least a 0.003
pairwise-accuracy gain.

```bash
sbatch updated_sbatch_jobs/train_fe_pairwise_pilot.sh \
  data/hf_datasets/en/unieval_pilot_50k \
  outputs/fe_qwen3_unieval_pilot_50k
```

## Configuration contracts

`clumsification_code/data/schemas.py` defines versioned original, candidate,
score, manifest, generation, workflow, and HF-build contracts. Unknown config
fields are rejected. Example configs live under `configs/` and the complete
field descriptions are in `docs/PERTURBATION_CONFIGS.md`.

## Legacy boundary

`scripts/import_legacy_dataset.py` is the only supported route from historical
layer folders into the canonical repository. Legacy readers and deprecated
experiment scripts are not used by generation, scoring, HF building, training,
or evaluation. They are moved to ignored local archive folders during cleanup.

## Source-tree policy

Tracked files are limited to the paper's central methodology: reusable code,
user-facing local scripts, stable configs, prompt assets, and stable
documentation. Datasets, generated outputs, tests, notebooks, figures,
changing internal documents, archives, and all cluster batch jobs remain
untracked.
