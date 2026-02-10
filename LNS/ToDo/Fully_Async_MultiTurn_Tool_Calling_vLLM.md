# LNS ToDo: Fully Async Multi-Turn Tool Calling via vLLM

작성일: 2026-02-09
대상 프로젝트 루트: `HyunBin/DDAI_Revised/verl`
범위: `LNS/` 하위 실행 파이프라인

---

## 1) 문서 목적

이 문서는 현재 LNS 실험(검색 + bbox tool + multi-turn RL)을 **`sglang + main_ppo`** 구성에서
**`vllm + fully_async_policy`** 구성으로 이전할지 판단하고, 실제 전환 시 변경 범위/리스크/검증 절차를
치밀하게 정리하기 위한 ToDo 문서다.

핵심 목표는 다음 3가지다.

1. 현재 병목(긴 tool-call tail, GPU 유휴 시간, Ray actor 불안정)을 구조적으로 줄일 수 있는지 검증
2. 기존 LNS 멀티턴/툴/리워드 설계를 최대한 보존하면서 전환 가능한 최소 변경 경로 정의
3. 실패 시 즉시 롤백 가능한 안전한 실험 절차 수립

---

## 2) 현재 상태 스냅샷 (As-Is)

### 2.1 실행 엔트리/백엔드

현재 LNS 실행 스크립트는 `main_ppo` + `sglang` 조합이다.

- 엔트리: `LNS/run_qwen2.5-3b_instruct_search_multiturn.sh:26`
- 백엔드: `LNS/search_multiturn_grpo.yaml:28` (`name: sglang`)
- 하이브리드 엔진: `LNS/search_multiturn_grpo.yaml:24` (`hybrid_engine: True`)

### 2.2 멀티턴 + 툴 설정

- tool agent 사용: `LNS/search_multiturn_grpo.yaml:27`
- multi-turn 활성화: `LNS/search_multiturn_grpo.yaml:30`
- 포맷: `LNS/search_multiturn_grpo.yaml:32` (`format: qwen`)
- tool config: `LNS/search_multiturn_grpo.yaml:33`

툴 정의:

- `search`: `LNS/LNS_tool_config.yaml:2`~`LNS/LNS_tool_config.yaml:23`
- `bbox`: `LNS/LNS_tool_config.yaml:24`~`LNS/LNS_tool_config.yaml:41`
- 로컬 이미지 루트: `LNS/LNS_tool_config.yaml:10` (`./LNS/corpus/img`)

### 2.3 관찰된 운영 이슈 (대화/실행 로그 기반)

1. Ray actor 안정성
- `TokenBucketWorker` owner crash 연쇄로 `ActorDiedError` 반복
- `SearchExecutionWorker`에서 `_raylet` SIGABRT 다수

2. Ray worker 폭증
- `PYTHON worker processes have been started ...` 경고 다수
- 보통 `ray.get()` 블로킹/actor 수 과다 시 자주 관찰

3. 유휴 구간
- 학습 중 GPU memory는 점유되지만 util이 0%로 길게 유지되는 구간 존재
- 특히 tool_call 구간이 generation 대비 매우 길게 관찰됨 (`timing_s/agent_loop/tool_calls`)

4. reward 신호 편향
- format은 일부 통과하지만 `ndcg_mean`이 0에 고정되는 구간 관찰
- 포맷 게이트/검색 결과 매칭/로컬 이미지 경로 조건이 동시에 영향

---

## 3) 왜 Fully Async + vLLM를 검토하는가

`fully_async_policy` 문서가 제시하는 구조적 이점:

- Train/Rollout 자원 분리
- 샘플 생성/학습 동시 진행
- freshness(staleness) 제어
- partial rollout로 sync 시점 대기 손실 감소

근거:

- 특징/모드 설명: `verl/experimental/fully_async_policy/README.md:45`~`verl/experimental/fully_async_policy/README.md:60`
- 지원 모드 명시: `verl/experimental/fully_async_policy/README.md:60`
  - 현재 문서상 지원은 **megatron/fsdp + vllm(AgentLoop server mode)**

즉, 현재 long-tail tool 호출이 있는 LNS 상황에서는,
"동기 step 파이프라인"보다 fully async가 구조적으로 유리할 가능성이 높다.

---

## 4) 코드베이스 적합성 점검 (핵심)

### 4.1 fully async 엔트리 존재

- `verl.experimental.fully_async_policy.fully_async_main` 엔트리 확인
- hydra main: `verl/experimental/fully_async_policy/fully_async_main.py:297`
- `async_training` 없으면 즉시 실패: `verl/experimental/fully_async_policy/fully_async_main.py:302`

### 4.2 multi-turn tool agent 호환

`ToolAgentLoop`의 tool 파서는 포맷 문자열 기반 선택이다.

- parser 선택: `verl/experimental/agent_loop/tool_agent_loop.py:225`~`verl/experimental/agent_loop/tool_agent_loop.py:228`

따라서 `format: qwen`은 구조상 유지 가능하며,
fully async가 포맷을 강제하는 구조는 아니다.

### 4.3 fully async에서 tool-agent partial 경로 존재

- async partial tool agent 등록: `verl/experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py:30`
- `partial_rollout` 참조: `verl/experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py:39`
- multi-turn enable 시 agent 자동 선택: `verl/experimental/fully_async_policy/detach_utils.py:80`~`verl/experimental/fully_async_policy/detach_utils.py:83`

요약: 현재 코드베이스는 fully async + tool multi-turn를 이미 고려한 경로를 가진다.

---

## 5) 레퍼런스 구성 (내부 예시)

실전 레퍼런스로 활용 가능한 예시:

1. 실행 스크립트
- `verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh:72`
- 엔트리, `rollout.name=vllm`, `rollout.mode=async`, `async_training.*`, trainer/rollout 자원 분리 포함

2. config 골격
- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml:9` 이후
- `async_training`, `rollout`, `data.gen_batch_size` 기본 구조 확인 가능

---

## 6) LNS 전환 시 변경 범위 추정 (아직 미구현)

### 6.1 필수 변경

1. 실행 엔트리 전환
- from: `python3 -m verl.trainer.main_ppo`
- to: `python3 -m verl.experimental.fully_async_policy.fully_async_main`
- 대상 파일: `LNS/run_qwen2.5-3b_instruct_search_multiturn.sh`

2. rollout backend 전환
- from: `actor_rollout_ref.rollout.name=sglang`
- to: `actor_rollout_ref.rollout.name=vllm`
- 대상 파일: `LNS/search_multiturn_grpo.yaml` 또는 async 전용 새 yaml

3. async 필수 파라미터 추가
- `async_training.staleness_threshold`
- `async_training.trigger_parameter_sync_step`
- `async_training.require_batches`
- `async_training.partial_rollout`
- `data.gen_batch_size=1`
- `trainer`/`rollout` 리소스 분리

### 6.2 권장 변경 (안전/가독성)

1. 파일 분리
- 기존 `search_multiturn_grpo.yaml` 보존
- 신규 `search_multiturn_grpo_async_vllm.yaml` 생성

2. 실행 스크립트 분리
- 기존 스크립트 보존
- 신규 async 실행 스크립트 생성

### 6.3 예상 변경량

- 파일: 2~3개
- 라인 변화: 대략 120~240 LoC (설정/런처 중심)
- Python 로직 파일(`search_tool.py`, `bbox_tool.py`, reward 함수)은 1차 전환에서 비필수

---

## 7) 전환 전 선행 정리 항목 (중요)

1. 데이터/리워드 정합성
- NDCG 0 고정 이슈를 먼저 분리 진단
- 원인 후보: format gate, retrieved 경로 누락, reference 매핑 불일치

2. 툴 실패 정책
- 검색 결과 파일 미존재 시 현재는 점수 0에 가까운 실패만 누적될 수 있음
- 추후 "hard fail vs warning" 정책 분리 필요

3. 로그 노이즈
- weave call URL 대량 출력은 성능 직접 영향보다 디버깅 가독성/운영 피로도 이슈
- 실행 단계에서 logger/trace 정책 정리 필요

4. Ray 안정성 선확보
- actor 수, blocking call 패턴, worker 재시작 정책 점검
- fully async 전환 전 baseline 안정성 확보가 비교 실험 해석에 유리

---

## 8) 실험 계획 (제안)

### Phase 0: Baseline 고정

목표:
- 현재 sglang 경로에서 10~20 step 로그를 표준 포맷으로 고정 수집

수집 항목:
- `timing_s/gen`
- `timing_s/agent_loop/tool_calls`
- `perf/throughput`
- `response/aborted_ratio`
- `reward/format_score_mean`
- `reward/ndcg_mean`

성공 기준:
- 최소 10 step 이상 중단 없이 완료
- 주요 지표 시계열 확보

### Phase 1: Async 최소 이행 (기능 확인)

목표:
- fully async + vllm + multi-turn + tool 호출이 기능적으로 동작하는지 확인

설정 시작점(권장):
- GPU split: trainer 4 / rollout 4 (1노드 8GPU)
- `staleness_threshold=0.1~0.5`
- `trigger_parameter_sync_step=2~4`
- `require_batches=1`
- `partial_rollout=True`

성공 기준:
- 1~3 step smoke run 안정 통과
- tool call + reward 계산 + checkpoint/save 루틴 무중단

### Phase 2: 성능/안정화 튜닝

목표:
- baseline 대비 유휴 시간 감소와 throughput 개선 검증

튜닝 축:
- `staleness_threshold`
- `trigger_parameter_sync_step`
- `n`, `ppo_mini_batch_size`, `gen_batch_size`
- rollout GPU memory utilization

성공 기준:
- `timing_s/agent_loop/tool_calls` 구간이 step wall-clock에서 차지하는 비율 감소
- 동일 자원 기준 step throughput 개선
- reward 분포의 급격한 붕괴 없음

---

## 9) 리스크 레지스터

### R1. 학습 안정성 저하

설명:
- stale sample 비율 상승 시 편향/불안정 가능

대응:
- `staleness_threshold` 낮은 값부터 시작
- 필요 시 on-policy에 가까운 설정으로 후퇴 (`staleness_threshold=0`)

### R2. 롤아웃/트레이너 자원 불균형

설명:
- rollout이 너무 느리면 trainer starvation, 반대면 queue overflow/stale 증가

대응:
- 4:4 시작 후 지표 기반 조정
- queue 관련 상태/처리율 추적

### R3. 도구 호출 tail 지속

설명:
- fully async로 완화는 가능하지만, tool 서버 자체 지연이 크면 근본 병목 잔존

대응:
- search 서버 응답 시간 분포와 timeout 정책 별도 측정
- 필요 시 search worker/rate limit 재튜닝

### R4. reward 해석 어려움

설명:
- NDCG 0이 계속되면 성능 개선 여부 판단 불가

대응:
- reward 구성요소(format vs ndcg) 분리 로그 강제
- retrieval 결과/정답 매칭 디버그 샘플링

---

## 10) 최종 의사결정 체크리스트

아래 항목을 모두 만족하면 전환 구현 시작:

- [ ] Baseline 10~20 step 지표가 확보되었다
- [ ] NDCG 0 고정 원인에 대한 우선순위 가설이 정리되었다
- [ ] LNS 이미지 경로/검색 결과 파일명/정답 매핑 규칙이 문서화되었다
- [ ] async 전환용 yaml/launcher를 기존 파일과 분리해 운영하기로 합의했다
- [ ] 초기 리소스 split(예: 4:4)과 실패 시 롤백 조건이 합의되었다

---

## 11) 구현 시점 ToDo (미착수)

1. 구성 파일
- [ ] `LNS/search_multiturn_grpo_async_vllm.yaml` 초안 작성
- [ ] `LNS/run_qwen2.5-3b_instruct_search_multiturn_async_vllm.sh` 초안 작성

2. 실행 검증
- [ ] smoke run (1~3 step)
- [ ] short run (10~20 step)
- [ ] baseline 대비 지표 표 생성

3. 안정화
- [ ] staleness / sync-step sweep
- [ ] rollout/trainer 자원 split sweep
- [ ] tool timeout / concurrency 정책 점검

---

## 12) 참고 파일 목록

현재 LNS 상태 확인 기준:

- `LNS/run_qwen2.5-3b_instruct_search_multiturn.sh`
- `LNS/search_multiturn_grpo.yaml`
- `LNS/LNS_tool_config.yaml`

fully async 관련 내부 레퍼런스:

- `verl/experimental/fully_async_policy/README.md`
- `verl/experimental/fully_async_policy/fully_async_main.py`
- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
- `verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh`
- `verl/experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py`
- `verl/experimental/fully_async_policy/detach_utils.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`

---

## 13) 결론

현재 코드베이스 관점에서, LNS의 fully async + vllm 전환은
"새로운 기능 개발"이라기보다 "기존 경로를 LNS에 안전하게 이식"하는 작업이다.

다만 지금 단계에서 가장 중요한 것은,
전환 자체보다도 **baseline 지표 고정 + reward 신호 정합성 확보 + 안정성 기준 수립**이다.

이 3가지를 선행하면, 전환 후 성능 개선 여부를 명확히 판단할 수 있다.
