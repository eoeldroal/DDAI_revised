# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is a production-ready reinforcement learning training library for large language models. It implements a hybrid-controller programming model that enables flexible representation and efficient execution of complex post-training dataflows (PPO, GRPO, RLOO, etc.).

Key characteristics:
- Supports FSDP and Megatron-LM backends for training
- Supports vLLM, SGLang, and HuggingFace TGI for rollout generation
- Scales from single GPU to 671B models across hundreds of GPUs
- Multi-hardware support: NVIDIA, AMD (ROCm), Ascend NPU

## Common Commands

### Installation
```bash
# Development installation with vLLM
pip install -e .[test,vllm]

# Development installation with SGLang
pip install -e .[test,sglang]

# Full installation with Megatron support
bash scripts/install_vllm_sglang_mcore.sh
```

### Linting and Formatting
```bash
pip install pre-commit
pre-commit install

# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files ruff
```

### Testing
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_protocol_on_cpu.py -v

# CPU tests (files ending with _on_cpu.py)
pytest tests/**/test_*_on_cpu.py

# GPU tests (files NOT ending with _on_cpu.py)
pytest tests/ --ignore-glob='*_on_cpu.py'
```

### Running Training
```bash
# PPO training (Hydra-based CLI)
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    data.train_files=path/to/data.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.n_gpus_per_node=2

# Example scripts are in examples/
bash examples/ppo_trainer/run_gemma.sh
bash examples/grpo_trainer/run_qwen2-7b_math.sh
```

### Building Documentation
```bash
cd docs
pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/
```

## Architecture

### Core Components

**DataProto** (`verl/protocol.py`): Unified data transfer protocol built on `tensordict.TensorDict`. All data passed between workers uses this format. Supports padding, concatenation, and distributed operations.

**BaseConfig** (`verl/base_config.py`): Immutable dataclass with dict-like interface. Foundation for the Hydra configuration system.

**Workers** (`verl/workers/`):
- `fsdp_workers.py`: PyTorch FSDP training backend
- `megatron_workers.py`: Megatron-LM training backend
- `engine_workers.py`: Ray-based computation workers
- Workers handle actor, critic, rollout, and reward model roles

**Trainers** (`verl/trainer/`):
- `main_ppo.py`: Main PPO entry point (most common)
- `main_generation.py`: Inference/generation
- `main_eval.py`: Evaluation
- `sft_trainer.py`: Supervised fine-tuning

**Single Controller** (`verl/single_controller/`): Ray-based distributed controller that manages worker lifecycle and communication.

**Checkpoint Engine** (`verl/checkpoint_engine/`): Unified weight synchronization layer with NCCL and NIXL backends.

### Configuration System

Uses **Hydra** (OmegaConf) for configuration management:
- Config path: `verl/trainer/config/`
- Default configs: `ppo_trainer.yaml`, `ppo_megatron_trainer.yaml`
- Hierarchical composition with sub-configs in `actor/`, `critic/`, `rollout/`, etc.
- Auto-generated flattened configs: `_generated_ppo_trainer.yaml`

Override configs via CLI:
```bash
python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=4 algorithm.gamma=0.99
```

### Directory Structure

```
verl/
├── verl/                    # Main package
│   ├── protocol.py          # DataProto - core data structure
│   ├── base_config.py       # BaseConfig for Hydra
│   ├── trainer/             # Training entry points & PPO algorithm
│   │   ├── main_ppo.py      # Main entry point
│   │   ├── config/          # Hydra YAML configurations
│   │   └── ppo/             # PPO algorithm implementation
│   ├── workers/             # Distributed workers (FSDP, Megatron)
│   ├── single_controller/   # Ray-based controller
│   ├── models/              # Model implementations
│   ├── utils/               # Utilities (dataset, reward_score, etc.)
│   └── experimental/        # Experimental features (async PPO, VLA)
├── tests/                   # Test suite
│   ├── special_distributed/ # Multi-GPU tests
│   ├── special_e2e/         # End-to-end tests
│   └── special_sanity/      # Quick sanity checks
├── examples/                # Training examples with run_*.sh scripts
├── docs/                    # Sphinx documentation
└── scripts/                 # Utility scripts
```

### Test Organization

- `tests/**/test_*_on_cpu.py`: Run on CPU (in `cpu_unit_tests.yml`)
- `tests/**/test_*.py`: Run on GPU (in `gpu_unit_tests.yml`)
- `tests/special_distributed/`: Multi-GPU tests
- `tests/special_e2e/`: End-to-end training tests
- `tests/special_sanity/`: Quick sanity checks (docstrings, license headers)

### Training Pipeline Flow

1. **Rollout Phase**: Generate trajectories using vLLM/SGLang
2. **Reward Computation**: Function-based (GSM8K, MATH) or model-based rewards
3. **Advantage Estimation**: GAE, GRPO, RLOO algorithms
4. **Training Phase**: Update actor/critic via FSDP or Megatron
5. **Weight Sync**: Checkpoint engine synchronizes weights

### Environment Variables

- `VERL_AUTO_PADDING`: Enable auto-padding for DataProto
- `VERL_USE_MODELSCOPE`: Download from ModelScope instead of HuggingFace
- `VERL_USE_EXTERNAL_MODULES`: Load external modules (comma-separated)
- `VLLM_USE_V1=1`: Enable vLLM v1 for better performance

## Code Style

- Line length: 120 characters
- Uses ruff for linting (pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, isort)
- All new files require Apache 2.0 license headers
- Docstrings required for public functions (checked by pre-commit)
- Type checking via mypy (selective enforcement on core modules)

---

## Documentation Reference (`docs/`)

### Core Architecture (Start Here)

| File | Description |
|------|-------------|
| `docs/hybrid_flow.rst` | **CRITICAL** - HybridFlow programming model: dataflow abstraction, control vs computation flow separation, `@register` decorator pattern, PPO main loop composition |
| `docs/single_controller.rst` | **CRITICAL** - Deep dive into `verl.single_controller`: WorkerGroup, ResourcePool, ClassWithArgs, dispatch modes (`ONE_TO_ALL`, `DP_COMPUTE_PROTO`), three-step binding process |
| `docs/examples/ppo_code_architecture.rst` | Step-by-step PPO implementation walkthrough with code references |

### Getting Started

| File | Description |
|------|-------------|
| `docs/start/install.rst` | Installation guide: Python/CUDA requirements, Docker images, FSDP/Megatron/vLLM/SGLang setup, AMD ROCm support |
| `docs/start/quickstart.rst` | End-to-end PPO training on GSM8K with Qwen2.5-0.5B, dataset prep, reward functions, checkpointing |
| `docs/start/multinode.rst` | Multi-node Ray cluster setup, SkyPilot integration for cloud/Kubernetes |
| `docs/start/ray_debug_tutorial.rst` | Ray distributed debugging techniques |
| `docs/start/agentic_rl.rst` | Agentic RL training with tool calling |

### Algorithm Implementations

| File | Description |
|------|-------------|

---

# Local Notes (VDR/Filtering)

- 작업 기준 루트: `/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1`
- 원본 `data/VDR_processed`는 유지, 모든 필터링 결과는 `VDR_processed_filtered_1` 내부에 반영
- OpenDocVQA에서 `slidevqa/` 서브셋 제거 완료 (SlideVQA는 별도 코퍼스로 유지)
- SigLIP 이미지 임베딩 생성 및 ANN 후보 추출/적용 완료 (0.999 기준, 대표 doc_id는 사전순 최솟값)
- 1차 RLVR 답변 길이 필터 적용 결과는 `qa_filtered_RLVR`에 저장 (원본 `qa` 유지)
- 필터/전처리 스크립트는 `HyunBin/DDAI_Revised/verl/filter/filter_phase_1` 하위에 정리됨
| `docs/algo/ppo.md` | PPO: clipped surrogate, GAE, actor-critic, KL control, dual-clip extension |
| `docs/algo/grpo.md` | GRPO: critic-free, group sampling, relative rewards, DrGRPO extension |
| `docs/algo/baseline.md` | **Benchmark table**: performance across models (0.5B-72B), algorithms, datasets |
| `docs/algo/dapo.md` | Distributional advantage PPO |
| `docs/algo/entropy.md` | Entropy regularization methods |
| `docs/algo/spin.md` | Self-play instruction-following |
| `docs/algo/sppo.md` | Sequence-level PPO |
| `docs/algo/gpg.md` | Group Policy Gradient (critic-free, no KL penalty) |
| `docs/algo/rollout_corr.md` | Off-policy rollout correction |

### Worker Configurations

| File | Description |
|------|-------------|
| `docs/workers/ray_trainer.rst` | RayPPOTrainer: WorkerGroup init, data preparation, training loop with remote calls |
| `docs/workers/fsdp_workers.rst` | FSDP backend: ActorRolloutRefWorker, CriticWorker, RewardWorker, dispatch patterns |
| `docs/workers/megatron_workers.rst` | Megatron-LM: 5D parallelism (TP/EP/CP/DP/PP), 3D HybridEngine, checkpoint conversion |
| `docs/workers/sglang_worker.rst` | SGLang backend: installation, CUDA 12.4+, multi-turn support roadmap |

### Configuration Reference

| File | Description |
|------|-------------|
| `docs/examples/config.rst` | **Comprehensive config reference**: data, actor/rollout/ref, critic, algorithm, trainer, parallelism, optimization parameters |

### Performance Optimization

| File | Description |
|------|-------------|
| `docs/perf/best_practices.rst` | Qwen3-235B + DAPO example, parameter mapping to math formulations |
| `docs/perf/perf_tuning.rst` | Rollout tuning (vLLM memory 0.5-0.7), sequence packing, dynamic batching, Ulysses SP, LigerKernel |
| `docs/perf/device_tuning.rst` | Hardware-specific optimizations |
| `docs/perf/verl_profiler_system.md` | Profiling tools and metrics |
| `docs/perf/nsight_profiling.md` | NVIDIA Nsight integration |
| `docs/perf/dpsk.md` | Data parallel scaling kit |

### Advanced Features

| File | Description |
|------|-------------|
| `docs/advance/dpo_extension.rst` | **Extension guide**: 3-step pattern to add new algorithms |
| `docs/advance/fsdp_extension.rst` | Extend FSDP worker for new models |
| `docs/advance/megatron_extension.rst` | Megatron backend customization |
| `docs/advance/ppo_lora.rst` | LoRA integration with PPO |
| `docs/advance/checkpoint.rst` | Fault tolerance, checkpoint structure, resume training |
| `docs/advance/attention_implementation.rst` | Attention backends: flash_attention_2, eager, sdpa |
| `docs/advance/placement.rst` | Device placement strategies |
| `docs/advance/fully_async.md` | Fully asynchronous training (2.35-2.67x speedup) |
| `docs/advance/one_step_off.md` | One-step off-policy corrections |
| `docs/advance/agent_loop.rst` | Agent-based training loops |
| `docs/advance/reward_loop.rst` | Reward computation loops |
| `docs/advance/fp8.md` | FP8 quantization |
| `docs/advance/grafana_prometheus.md` | Monitoring setup |

### Data & Rewards

| File | Description |
|------|-------------|
| `docs/preparation/prepare_data.rst` | Data format (parquet), `make_map_fn()`, example datasets |
| `docs/preparation/reward_function.rst` | RewardManager, pre-implemented rewards (GSM8K, MATH), custom reward functions |

### Multi-Turn & Tools

| File | Description |
|------|-------------|
| `docs/sglang_multiturn/multiturn.rst` | Multi-turn conversation RL with SGLang |
| `docs/sglang_multiturn/interaction_system.rst` | Tool use in multi-turn RL |
| `docs/sglang_multiturn/search_tool_example.rst` | Search tool integration example |

### Hardware-Specific

| File | Description |
|------|-------------|
| `docs/amd_tutorial/` | AMD ROCm: Docker build, vLLM on MI300 |
| `docs/ascend_tutorial/` | Ascend NPU: quickstart, profiling, SGLang setup |

### FAQ

| File | Description |
|------|-------------|
| `docs/faq/faq.rst` | Ray issues, multi-node/Slurm setup, TensorDict compatibility |

---

## Examples Reference (`examples/`)

### Algorithm Trainers

| Directory | Algorithm | Key Scripts | Description |
|-----------|-----------|-------------|-------------|
| `ppo_trainer/` | PPO | `run_gemma.sh`, `run_deepseek7b_llm.sh`, `run_qwen2-7b_rm.sh` | Classic actor-critic with GAE; includes reward model training variants |
| `grpo_trainer/` | GRPO | `run_qwen2-7b.sh`, `run_qwen2_5_vl-7b.sh`, `run_deepseek671b_math_megatron_96gb.sh` | Critic-free with group sampling; VLM, LoRA, Megatron, NPU variants |
| `rloo_trainer/` | RLOO | `run_qwen2-7b.sh` | Leave-one-out baseline, no critic needed |
| `gmpo_trainer/` | GMPO | `run_qwen2_5-7b_math.sh` | Geometric-mean for stability (`loss_mode=geo_mean`) |
| `gpg_trainer/` | GPG | `run_qwen2-7b_math.sh`, `run_qwen2-7b_math_megatron.sh` | Minimalist: no critic, no KL, no reference model |
| `reinforce_plus_plus_trainer/` | REINFORCE++ | `run_qwen2-7b_math_rf.sh` | Baseline variance reduction, 8 rollouts |
| `remax_trainer/` | ReMax | `run_qwen2.5-3b_seq_balance.sh` | Maximum entropy variant with sequence packing |
| `otb_trainer/` | OTB | `run_qwen2_5-7b.sh` | Token-level baseline for variance reduction |
| `gspo_trainer/` | GSPO | `run_qwen30b_gspo.sh` | Gradient smoothing for 30B+ models |
| `sapo_trainer/` | SAPO | `run_qwen30b_sapo.sh` | Adaptive smoothing, Slurm multi-node |
| `cispo_trainer/` | CISPO | `run_cispo_qwen2_5_0_5b_gsm8k.sh` | Asymmetric clipping |

### Special Features

| Directory | Feature | Key Scripts | Description |
|-----------|---------|-------------|-------------|
| `sft/` | Supervised Fine-Tuning | `run_qwen_05_sp2.sh`, `run_qwen_05_peft.sh` | SFT with sequence parallelism and LoRA |
| `sglang_multiturn/` | Multi-Turn + Tools | `run_qwen2.5-3b_gsm8k_multiturn.sh`, `run_qwen0.5b_gsm8k_multiturn_curriculum.sh` | Tool-augmented conversations, curriculum learning, MLflow integration |
| `split_placement/` | GPU Splitting | `run_deepseek7b_llm.sh` | Separate actor/critic on different GPU groups |
| `prefix_grouper/` | Long-Context Opt | `run_qwen3_prefix_grouper.sh` | 1.14-1.70x speedup via prefix/suffix attention decomposition |
| `router_replay/` | MoE Determinism | `run_qwen30_a3b_megatron_vllm.sh` | Record/replay routing for MoE models |
| `rollout_correction/` | Off-Policy Fix | `run_with_rollout_corr.sh` | Correct distribution mismatch in rollouts |
| `skypilot/` | Cloud Deploy | Various | Kubernetes/cloud deployment via SkyPilot |

### Quick Start by Use Case

**Basic Training:**
```bash
examples/grpo_trainer/run_qwen2-7b.sh          # GRPO baseline
examples/ppo_trainer/run_gemma.sh              # PPO baseline
```

**Vision-Language Models:**
```bash
examples/grpo_trainer/run_qwen2_5_vl-7b.sh     # VLM GRPO
examples/grpo_trainer/run_qwen2_5_vl-7b_lora.sh # VLM + LoRA
```

**Large Models (30B+):**
```bash
examples/gspo_trainer/run_qwen30b_gspo.sh      # Gradient smoothing
examples/sapo_trainer/run_qwen30b_sapo.sh      # Multi-node Slurm
examples/grpo_trainer/run_deepseek671b_math_megatron_96gb.sh  # 671B
```

**Multi-Turn with Tools:**
```bash
examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
examples/sglang_multiturn/run_qwen0.5b_gsm8k_multiturn_curriculum.sh
```

**Efficiency Optimizations:**
```bash
examples/prefix_grouper/run_qwen3_prefix_grouper.sh   # Long context
examples/ppo_trainer/run_qwen2-7b_seq_balance.sh      # Sequence packing
```

### Common Configuration Patterns

```bash
# Algorithm selection
algorithm.adv_estimator=grpo|gae|rloo|remax|gpg

# Rollout backend
actor_rollout_ref.rollout.name=vllm|sglang

# KL control
actor_rollout_ref.actor.use_kl_loss=True

# Number of rollouts per prompt
actor_rollout_ref.rollout.n=8

# Efficiency
actor_rollout_ref.model.use_remove_padding=True
ulysses_sequence_parallel_size=2
```

### Backend Selection Guide

| Backend | Use Case | Model Size |
|---------|----------|------------|
| FSDP | Research, prototyping | 7B-32B |
| Megatron | Production, scaling | 70B+ |
| vLLM | Fast inference | Default |
| SGLang | Tool-augmented, multi-turn | Complex tasks |

### Supported Models in Examples

- **Qwen**: 0.5B, 3B, 7B, 8B, 32B, 235B (including VL, Math, MoE variants)
- **DeepSeek**: 7B, 67B, 671B
- **Gemma**: 2B, 7B
- **GLM-4.1V**: 9B (multimodal)
- **Moonlight**: 16B

### Datasets Used

- **GSM8K**: Grade school math (most common)
- **MATH**: Competition-level math
- **Geo3K**: Geospatial reasoning with images
- **DAPO-Math**: Large-scale mathematical reasoning
- **HH-RLHF**: Human feedback

---

## LNS Multi-Turn Search: 설정 체계

### 관련 파일 (3개 + 2개)

| 파일 | 위치 | 역할 |
|------|------|------|
| **Shell 스크립트** | `run_qwen2.5-3b_instruct_search_multiturn.sh` | 실험 실행 진입점. Hydra CLI override로 설정 덮어쓰기 |
| **YAML config** | `examples/sglang_multiturn/config/search_multiturn_grpo.yaml` | Hydra 메인 config. 학습 파라미터 + rollout 설정 |
| **Tool config** | `examples/sglang_multiturn/config/tool_config/LNS_tool_config.yaml` | Tool 정의 (SearchTool, ImageCropper). URL, timeout 등 |
| SearchTool 구현 | `verl/tools/LNS/search_tool.py` | 실제 검색 실행 코드 (aiohttp → 외부 서버) |
| Reward 함수 | `verl/utils/reward_score/format_ndcg_reward.py` | format 검증 + NDCG 점수 (0.1 + 0.9 * ndcg) |

### Hydra 설정 흐름 (3단계 레이어)

```
① verl/trainer/config/ppo_trainer.yaml        ← 프레임워크 기본값 (defaults에서 상속)
        ↓ 덮어쓰기
② search_multiturn_grpo.yaml                  ← --config-path + --config-name 으로 지정
        ↓ 덮어쓰기
③ Shell 스크립트의 CLI overrides               ← 최종 우선순위 (가장 강력)
```

진입점: `verl/trainer/main_ppo.py:35`
```python
@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):    # ← Shell의 --config-path/--config-name이 기본값을 대체
    run_ppo(config)  # ← config 객체에 3단계가 합쳐져서 전달
```

### 설정값 → 실제 사용처 매핑

```
설정값                        정의 위치              사용 코드
──────────────────────────────────────────────────────────────────

retrieval_service_url         LNS_tool_config.yaml   SearchTool.__init__() → self.url
  (검색 서버 URL)                                     ✅ 실제 연결에 사용

local_image_root              LNS_tool_config.yaml   SearchTool.__init__() → self.local_image_root
  (이미지 로컬 경로)                                    ✅ 이미지 로딩에 사용

tool_config_path              YAML:26                ToolAgentLoop.__init__() → tool 인스턴스화
  (tool YAML 경로)                                    ✅ tool 로드에 사용

tool_settings                 YAML:27-28             tool_agent_loop.py:220-221 → extra_config
  (tool config 덮어쓰기)                               → initialize_tools_from_config()에 merge
                                                      ⚠️  LNS_tool_config와 값 중복

TODO(human)
아래 항목들의 현재 상태를 확인하고 정리 방향을 기록하세요:

retriever.url                 YAML:30-31 + Shell:87  ???
  (검색 서버 URL 중복?)

default_agent_loop            YAML:20 + Shell:86     agent_loop.py 디스패치
  (agent loop 선택)                                   중복 여부?

tool_settings.local_image_root  YAML:28 + Shell:88   tool_registry extra_config
  (이미지 경로)                                        LNS_tool_config:19와 중복 여부?
```

### Tool 로딩 경로 (2개)

```
경로 A: data.tool_config_path → RLHFDataset
  tool schema를 읽어서 모델 prompt에 "사용 가능한 tool" 설명 삽입
  현재 상태: ❌ 미설정 (data 섹션에 tool_config_path 없음)

경로 B: rollout.multi_turn.tool_config_path → ToolAgentLoop
  tool 인스턴스를 생성하고 실제 execute() 호출
  현재 상태: ✅ 설정됨 (YAML:26)
```
