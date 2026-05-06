# SOPO

**SOPO (Socratic Policy Optimization)** is a multi-turn RL training method for hallucination detection. Compared to standard GRPO, SOPO introduces a teacher model that provides Socratic feedback when the student's answer is below a score threshold, guiding self-correction across turns.

## Methods

| Script | Method | Teacher Feedback |
|---|---|---|
| `train_sopo.sh` | SOPO | Yes |
| `train_grpo.sh` | GRPO | No |
| `train_capo.sh` | CAPO | No |

## Prerequisites

- 2 nodes × 8 GPUs (configured via `allocation_mode: sglang:d8p1t1+megatron:d2p1t4`)
- Docker image built from `Dockerfile`
- A teacher model served via vLLM/OpenAI-compatible API (SOPO only)

## Configuration

Edit `examples/havor/config_sopo.yaml` before training:

```yaml
user_root: /path/to/your/root     # base path
areal_path: ${user_root}/...      # AReaL installation directory
model_path: ${user_root}/...      # SFT checkpoint to start from
data_path: ${user_root}/...       # training .jsonl file
```

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