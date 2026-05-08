# SOPO

**SOPO (Socratic Policy Optimization)** is a multi-turn RL training method for hallucination detection. Compared to standard GRPO, SOPO introduces a teacher model that provides Socratic feedback when the student's answer is below a score threshold, guiding self-correction across turns.

## Released Models

| Model | URL |
|---|---|
| SOPO 4B | [https://huggingface.co/OverwhelmingFit/SOPO-4B](#) |
| SOPO 1.7B | [https://huggingface.co/OverwhelmingFit/SOPO-1.7B](#) |

## Methods

| Script | Method | Teacher Feedback |
|---|---|---|
| `train_sopo.sh` | SOPO | Yes |
| `train_grpo.sh` | GRPO | No |
| `train_capo.sh` | CAPO | No |

## Hardware

- **GRPO / CAPO**: 2 nodes × 8 GPUs (configured via `allocation_mode: sglang:d8p1t1+megatron:d2p1t4`)
- **SOPO**: same as above, plus an additional 4 GPUs on a separate node for serving the teacher model

## Configuration

Edit `examples/havor/config_sopo.yaml` before training:

```yaml
user_root: /path/to/your/root     # base path
areal_path: ${user_root}/...      # AReaL installation directory
model_path: ${user_root}/...      # cold-start SFT model (see below)
data_path: ${user_root}/...       # RL training data (see below)
```

| Resource | URL |
|---|---|
| Cold-start SFT model | [https://huggingface.co/OverwhelmingFit/SOPO-Coldstart-ragtruth5k](#) |
| RL training data | [https://huggingface.co/datasets/OverwhelmingFit/SOPO-RL-RAGTruth10k](#) |

## Training

### Step 1 — Main node

```bash
bash train_sopo.sh
```

This will:
1. Start a Ray head node on `MASTER_IP:MASTER_PORT`
2. Export teacher model env vars (`TEACHER_MODEL_URL`, `TEACHER_MODEL_NAME`)
3. Submit the training job via `areal.launcher.ray`

Logs are written to `logs/train_refinder_<timestamp>.log`.

### Step 2 — Other nodes

On each additional node, run:

```bash
bash ray_join.sh
```

This joins the worker node to the existing Ray cluster.

## SOPO-specific Environment Variables

Only required for `train_sopo.sh`:

| Variable | Description |
|---|---|
| `TEACHER_MODEL_URL` | OpenAI-compatible API endpoint of the teacher model |
| `TEACHER_MODEL_NAME` | Model name used in API requests |

Start the teacher model with vLLM before launching training:

```bash
vllm serve /path/to/Qwen3-Next-80B-A3B-Instruct \
    --served-model-name Qwen3-Next-80B-A3B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4
```

Then set the env vars in `train_sopo.sh` accordingly:

```bash
export TEACHER_MODEL_URL="http://<teacher-node-ip>:8000/v1"
export TEACHER_MODEL_NAME="Qwen3-Next-80B-A3B-Instruct"
```

## Evaluation

Evaluation runs inference on the [RAGTruth](https://arxiv.org/abs/2401.00396) test set and computes span-level and case-level F1/P/R for hallucination detection.

### Step 1 — Deploy the model with vLLM

```bash
vllm serve /path/to/checkpoint \
    --served-model-name SOPO_4B \
    --port 8000
```

### Step 2 — Configure eval.py

Edit the two constants at the top of `eval.py` to match your deployment:

```python
MODEL_URL = "http://localhost:8000/v1"   # vLLM server address
MODEL_NAME = "SOPO_4B"                   # must match --served-model-name
```

### Step 3 — Run inference + metrics

```bash
python eval.py --num_threads 32
```

This will:
1. Load the test set from the default parquet path
2. Run parallel inference (`--num_threads` controls concurrency)
3. Save raw predictions to `test_result/{MODEL_NAME}_ragtruth_test_<timestamp>.jsonl`
4. Compute and print span-level metrics per task type (QA / Data2txt / Summary / All)
5. Save the full metric report to `test_result/..._metric.json`

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--num_threads` | 32 | Parallel inference threads |
| `--chosen_task` | `QA,Data2txt,Summary` | Task types to evaluate |
| `--filepath` | _(empty)_ | Skip inference, load predictions from an existing `.jsonl` file |
