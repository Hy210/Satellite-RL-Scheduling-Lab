# 개인 RL 학습 노트

## 문서 목적

이 문서는 위성 촬영 스케줄링 프로젝트를 진행하면서 학습한 강화학습(Reinforcement Learning) 개념, 로직 및 알고리즘을 개인 학습 목적으로 누적한다.

구현 명세가 아니라 이해를 위한 설명을 기록하며, 실제 프로젝트의 확정 동작은 다음 문서를 기준으로 한다.

- [RL 스케줄링 설계](rl-scheduling-design.md)
- [단계별 구현 계획](implementation-plan.md)

RL 관련 질문과 답변은 프로젝트 루트의 [AGENTS.md](../AGENTS.md)에 정의된 규칙에 따라 이 문서에 자동으로 정리한다.

## 학습 현황

| 주제 | 상태 | 마지막 학습일 |
|---|---|---|
| 강화학습 기본 구조 | 학습 전 | - |
| MDP | 학습 전 | - |
| State와 Observation | 학습 중 | 2026-07-07 |
| Action과 Action Masking | 학습 중 | 2026-07-07 |
| Reward와 Return | 학습 전 | - |
| Policy와 Value Function | 학습 전 | - |
| Exploration과 Exploitation | 학습 전 | - |
| PPO와 Maskable PPO | 학습 중 | 2026-07-08 |
| Reward Shaping | 학습 전 | - |
| 학습 평가와 Baseline | 학습 중 | 2026-07-08 |
| 과적합과 일반화 | 학습 전 | - |

상태는 필요에 따라 `학습 전`, `학습 중`, `기초 이해`, `복습 필요` 등으로 갱신한다. 이는 시험이나 숙련도를 판정하기 위한 값이 아니라 학습 흐름을 확인하기 위한 표시다.

## 질문 색인

새로운 학습 항목을 추가할 때 해당 항목으로 이동할 수 있는 링크를 이곳에 추가한다.

<!-- 예시: - [Episode와 Step의 차이](#episode와-step의-차이) -->

- [초기 알고리즘으로 Maskable PPO를 사용하는 이유](#초기-알고리즘으로-maskable-ppo를-사용하는-이유)
- [Stable-Baselines3와 sb3-contrib의 관계](#stable-baselines3와-sb3-contrib의-관계)
- [학습 환경에는 의미 있는 선택이 필요하다](#학습-환경에는-의미-있는-선택이-필요하다)
- [가변 문제를 고정 Observation으로 표현하는 방법](#가변-문제를-고정-observation으로-표현하는-방법)
- [학습 Return과 평가 Return을 분리해서 보는 이유](#학습-return과-평가-return을-분리해서-보는-이유)
- [단계 6 성능 검증의 의미](#단계-6-성능-검증의-의미)

## 프로젝트 용어 대응표

| RL 용어 | 이 프로젝트에서의 의미 |
|---|---|
| Agent | 촬영 기회를 선택하는 정책 |
| Environment | 하루 동안의 위성 촬영 스케줄링 시뮬레이터 |
| Episode | 고정 시나리오에서 진행되는 1일 전체 일정 |
| Step | 현재 이벤트에서 촬영 기회 또는 skip을 한 번 선택하는 과정 |
| State/Observation | 현재 시각, 자세, 주문 진행도 및 미래 기회 요약 |
| Action | 현재 후보 중 하나의 `(strip, opportunity)` 선택 또는 skip |
| Reward | strip 우선순위 보상, 각도 보너스 및 미완료 패널티 |
| Return | 한 episode 동안 받은 reward의 누적값 |
| Action mask | 물리 및 운영 제약을 위반하는 행동의 선택 차단 |
| Policy | 관측 상태에서 어떤 촬영 기회를 선택할지 결정하는 규칙 |

## 학습 항목

아직 개별 학습 항목이 기록되지 않았다. 이후 RL 관련 질문과 설명이 발생하면 이 아래에 주제별로 누적한다.

## Stable-Baselines3와 sb3-contrib의 관계

### 핵심 정의

Stable-Baselines3(SB3)는 Python에서 자주 쓰는 강화학습 알고리즘 구현 라이브러리다. PPO, DQN, A2C, SAC처럼 널리 쓰이는 알고리즘과 학습 루프, 저장, 평가, callback 같은 공통 도구를 제공한다.

sb3-contrib는 Stable-Baselines3의 확장 라이브러리다. SB3 본체에는 들어가지 않았지만 실험적이거나 추가 기능이 필요한 알고리즘을 제공하며, 이 프로젝트에서 쓰는 Maskable PPO가 여기에 포함된다.

### 직관

SB3를 기본 도구 상자라고 보면, sb3-contrib는 같은 규격을 따르는 추가 도구 상자다. 둘은 완전히 별개의 철학을 가진 라이브러리라기보다, 같은 생태계 안에서 본체와 확장 패키지의 관계에 가깝다.

### 프로젝트에서의 적용

이 프로젝트는 `sb3-contrib>=2.9,<3`를 의존성으로 사용한다. sb3-contrib가 Stable-Baselines3를 기반으로 동작하므로, Maskable PPO를 쓰기 위해 sb3-contrib를 설치하면 SB3 계열의 정책, rollout, 학습 인터페이스를 함께 사용하게 된다.

프로젝트의 Gymnasium wrapper는 `Dict` observation, `Discrete(129)` action, `action_masks()`를 제공한다. Maskable PPO는 이 `action_masks()` 결과를 받아 현재 상태에서 실행 불가능한 촬영 후보의 선택 확률을 0으로 만든다.

### 주의할 점

- Stable-Baselines3는 강화학습 환경 자체를 대신 만들어 주지 않는다. 환경의 state, action, reward와 mask는 프로젝트 코드가 정확히 정의해야 한다.
- sb3-contrib의 Maskable PPO는 실행 불가능한 행동을 거르는 데 도움을 주지만, 좋은 reward 설계나 일반화 성능을 자동으로 보장하지는 않는다.
- 버전 호환성이 중요하다. 현재 프로젝트는 Gymnasium 1.3, Stable-Baselines3 2.9, sb3-contrib 2.9 조합을 기준으로 검증했다.

### 학습 기록

- 2026-07-07: Stable-Baselines3가 강화학습 알고리즘 라이브러리이고, sb3-contrib가 Maskable PPO 같은 확장 알고리즘을 제공하는 관계임을 정리했다.

## 초기 알고리즘으로 Maskable PPO를 사용하는 이유

### 핵심 정의

PPO(Proximal Policy Optimization)는 현재 정책을 한 번에 지나치게 크게 변경하지 않도록 제한하면서 policy를 반복적으로 개선하는 policy gradient 계열의 강화학습 알고리즘이다.

Maskable PPO는 PPO가 행동을 선택하기 전에 action mask를 적용해 현재 상태에서 실행 불가능한 행동의 선택 확률을 0으로 만드는 방식이다.

### 직관

일반 PPO가 모든 촬영 후보 중 하나를 고른다면, Maskable PPO는 먼저 시간과 자세 제약을 위반한 후보를 목록에서 지운 뒤 남은 후보 사이에서 무엇이 장기적으로 좋은지 학습한다.

이 프로젝트에서 에이전트가 배워야 할 핵심은 물리적으로 불가능한 행동을 반복해서 시도하는 것이 아니라, 실행 가능한 촬영 기회 중 어떤 조합이 높은 우선순위와 많은 완료 주문을 만드는지 판단하는 것이다.

### PPO의 주요 구성

PPO는 일반적으로 actor와 critic을 함께 학습한다.

- Actor: 현재 관측에서 각 행동을 선택할 확률인 policy를 출력한다.
- Critic: 현재 상태에서 앞으로 얻을 return의 기대값을 추정한다.
- Advantage: 실제 결과가 critic의 예상보다 얼마나 좋거나 나빴는지 나타낸다.

PPO의 clipping 목적은 새 정책이 이전 정책에서 너무 멀리 변하지 않게 하는 것이다.

```text
ratio = new_policy(action | state) / old_policy(action | state)

clipped_objective = min(
    ratio * advantage,
    clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
)
```

`epsilon`은 한 번의 업데이트에서 허용하는 정책 변화 범위를 제어한다. 이 수식의 세부 유도보다, PPO가 학습 업데이트를 보수적으로 제한해 안정성을 높인다는 의미를 먼저 이해하면 된다.

### Action mask가 확률에 적용되는 방식

Actor는 각 행동의 선택 점수인 logit을 출력한다. Maskable PPO는 유효하지 않은 행동의 logit을 매우 작은 값으로 바꾼 뒤 softmax를 적용한다.

```text
logits = [촬영 A 점수, 촬영 B 점수, skip 점수]
mask   = [True,          False,         True]

masked logits = [촬영 A 점수, -무한대, skip 점수]
softmax 결과  = [A의 확률,    0,       skip의 확률]
```

유효 행동 집합을 `V(s)`라고 하면 마스크가 적용된 정책은 다음처럼 생각할 수 있다.

```text
π_mask(a|s) = exp(z_a) / Σ exp(z_b)   if a ∈ V(s)
              0                       if a ∉ V(s)
                               b ∈ V(s)
```

- `s`: 현재 관측 상태
- `a`: 선택할 행동
- `z_a`: actor가 행동 `a`에 부여한 logit
- `V(s)`: 상태 `s`에서 실행 가능한 행동 집합

확률 0이 된 행동은 샘플링되지 않는다. 남은 유효 행동들의 확률 합은 다시 1이 된다. 따라서 후보의 유효 여부가 상태마다 바뀌어도 정책은 그때그때 가능한 후보 사이에서만 선택한다.

중요한 점은 rollout 때뿐 아니라 PPO가 log probability와 entropy를 다시 계산하는 학습 단계에서도 동일한 mask를 사용해야 한다는 것이다. 그렇지 않으면 행동을 뽑았던 확률분포와 정책을 업데이트할 때의 확률분포가 달라져 PPO의 확률비 계산이 일관되지 않는다.

### 한 번의 학습 흐름

```text
1. 환경이 observation과 action mask를 제공한다.
2. Actor가 행동 logits를 계산한다.
3. mask를 적용한 분포에서 유효 행동 하나를 샘플링한다.
4. 환경이 행동을 실행하고 reward와 다음 observation을 반환한다.
5. Critic의 value 추정으로 advantage를 계산한다.
6. 저장해 둔 mask를 다시 사용해 PPO clipped objective로 actor와 critic을 갱신한다.
7. 여러 step과 episode에서 이 과정을 반복한다.
```

Maskable PPO가 위 절차의 1~3단계만 바꾸는 것은 아니다. 6단계의 정책 평가에도 mask가 관여한다는 점이 일반 PPO와의 핵심 구현 차이다.

### 프로젝트에서의 적용

```text
Observation
  = 현재 시각, 위성 자세, 주문 진행도, 미래 기회 요약

Action
  = 현재 이벤트 후보 128개 + skip

Action mask
  = 시간, 자세, 중복 촬영 및 마감 제약을 위반한 후보 차단

Reward
  = strip 우선순위 보상 + 각도 보너스 - 미완료 패널티
```

예를 들어 현재 이벤트에 세 촬영 후보와 skip이 있고, 자세 전환 시간이 부족한 후보 2와 이미 촬영된 후보 3이 있다면 mask는 다음과 같다.

```text
행동       후보 1   후보 2   후보 3   skip
mask       True     False    False    True
선택 확률  학습됨   0        0        학습됨
```

에이전트는 후보 1을 지금 촬영할지 미래 기회를 위해 skip할지는 학습해야 한다. 반면 후보 2와 3이 불가능하다는 물리·운영 규칙은 환경이 확정적으로 판정한다.

학습은 작은 시나리오부터 진행한다.

```text
tiny 시나리오
-> Random valid보다 나은지 확인
-> small 시나리오
-> greedy 정책과 비교
-> full 시나리오
```

### 다른 알고리즘과의 관계

- Random, deadline 및 priority greedy는 학습 알고리즘이 아니라 비교 기준 정책이다.
- CP-SAT이나 정수계획은 작은 문제에서 최적해 또는 상한을 구하기 위한 비교 도구다.
- DQN 같은 value-based 알고리즘도 이산 행동에 사용할 수 있지만 초기 선택은 아니다. 먼저 action masking을 직접 지원하고 구현 경로가 명확한 Maskable PPO로 환경과 보상을 검증한다.

### 주의할 점

- PPO는 최적해를 보장하지 않는다.
- Action mask는 하드 제약에만 사용하고 단순히 좋지 않은 선택까지 차단하면 안 된다.
- 고정 시나리오 학습 성능은 다른 시나리오에 대한 일반화 성능을 의미하지 않는다.
- 관측과 보상이 잘못 설계되면 알고리즘을 바꿔도 좋은 정책을 학습하기 어렵다.
- Maskable PPO를 적용하기 전에 시뮬레이션 코어와 기준 정책으로 환경 규칙을 먼저 검증해야 한다.
- mask 계산에 버그가 있으면 좋은 행동을 영구적으로 숨기거나 불가능한 행동을 허용하므로, mask 자체를 별도로 단위 테스트해야 한다.
- 최소한 `skip`은 항상 유효하게 두어 모든 행동이 마스킹되는 상태를 방지한다.
- mask는 탐색을 없애지 않는다. 유효 행동 사이에서는 여전히 확률적으로 탐색한다.
- mask는 보상을 대신하지 않는다. 실행 가능 여부는 mask가, 실행 가능한 선택 중 장기적으로 무엇이 좋은지는 reward와 return이 가르친다.

### 학습 기록

- 2026-07-06: 초기 알고리즘이 Maskable PPO인 이유와 PPO의 actor, critic, advantage 및 clipping 개념을 정리했다.

## 학습 환경에는 의미 있는 선택이 필요하다

### 핵심 정의

강화학습 환경에는 같은 상태에서 결과가 다른 행동을 선택할 수 있는 의사결정 상황이 있어야 한다. 실행 가능한 행동이 사실상 하나뿐이라면 정책은 장기적인 전략을 배울 수 없다.

### 직관

모든 촬영 기회가 서로 다른 시각에 하나씩 나타나고 가능한 촬영을 항상 수행할 수 있다면, 네 가지 기준 정책도 모두 `촬영 가능하면 촬영`이라는 같은 결과를 만든다. 이 경우 PPO를 학습해도 우선순위와 미래 기회를 비교하는 능력을 배울 이유가 거의 없다.

### 프로젝트에서의 적용

초기 가상 생성기는 촬영 시각을 연속 난수로 생성했다. 그 결과 full 시나리오에서 동시 후보가 거의 없어 Random, deadline, priority 및 efficiency 정책의 결과가 완전히 같았다.

촬영 구간 시작을 10초 grid로 양자화하자 동일 시각에 여러 유효 후보가 생겼고, 정책마다 return, 완료 주문 수와 평균 촬영 각도가 달라졌다. 이 경쟁 상황이 RL이 학습할 실제 선택 문제를 만든다.

### 주의할 점

- 후보 수만 많다고 좋은 학습 환경은 아니다. 동시에 또는 가까운 시간에 서로 충돌하는 선택이 있어야 한다.
- 인위적인 충돌이 지나치게 많으면 가상 생성기의 패턴만 학습할 수 있다.
- 기준 정책 결과가 모두 같다면 RL 알고리즘보다 먼저 환경과 시나리오 생성기를 점검해야 한다.
- 양자화 간격은 실제 촬영 기회 분포를 대체하는 프로토타입 가정이므로 향후 실제 궤도 데이터에서 재검토해야 한다.

### 학습 기록

- 2026-07-07: 기준 정책 비교를 통해 연속 난수 시각에는 의사결정 충돌이 부족함을 확인하고, 10초 시간 grid가 경쟁 후보를 만드는 이유를 정리했다.

## 가변 문제를 고정 Observation으로 표현하는 방법

### 핵심 정의

신경망은 일반적으로 입력 tensor의 shape가 고정되어야 하지만, 이 프로젝트의 주문, strip과 현재 후보 수는 시점과 시나리오마다 달라진다. 최대 크기까지 빈 행을 추가하는 padding과 실제 데이터 위치를 나타내는 presence mask로 이를 고정 크기로 변환한다.

### 직관

실제 strip이 20개여도 2,000칸짜리 표에 앞에서부터 채우고 나머지는 0으로 둔다. 다만 값이 0인 실제 데이터와 빈칸을 구분해야 하므로 별도의 `strip_presence`에서 실제 행을 1로 표시한다.

### 프로젝트에서의 적용

관측은 전역 상태, 2,000개 strip 요약, 128개 현재 후보와 두 presence 배열로 나뉜 Dict다. Maskable PPO는 `MultiInputPolicy`로 이 배열들을 함께 입력받는다.

```text
padding/presence = 해당 행에 실제 데이터가 있는가?
action mask      = 해당 후보를 지금 실행할 수 있는가?
```

두 mask는 역할이 다르다. Presence는 신경망 입력의 빈 행을 구분하고, action mask는 policy의 출력 확률에서 실행 불가능한 행동을 제거한다.

### Gymnasium wrapper의 역할

Gym wrapper는 기존 simulator의 규칙을 다시 계산하지 않는다. simulator의 state와 reward를 고정 shape의 NumPy 관측 및 Gymnasium 반환 형식으로 변환하는 adapter 역할만 한다. 이 구조를 지켜야 CLI, 테스트, 웹과 RL이 같은 도메인 규칙을 공유한다.

### 주의할 점

- 최대 크기를 지나치게 크게 잡으면 신경망 파라미터와 계산량이 커진다.
- padding 값만 제공하고 presence를 생략하면 실제 0과 빈 행을 혼동할 수 있다.
- 관측에 미래 선택을 판단할 정보가 없으면 policy는 좋은 장기 전략을 학습할 수 없다.
- 모든 값을 정규화해 시간 86,400과 각도 30처럼 크기가 다른 숫자가 학습을 불안정하게 만들지 않도록 한다.

### 학습 기록

- 2026-07-07: Dict observation, padding, presence mask와 action mask의 역할 차이를 Gymnasium wrapper 구현에 맞춰 정리했다.
- 2026-07-06: action mask가 logits, 확률분포, rollout 및 PPO 업데이트에 적용되는 방식과 프로젝트 예시를 보강했다.

## 학습 Return과 평가 Return을 분리해서 보는 이유

### 핵심 정의

학습 return은 정책이 학습 데이터를 모으는 과정에서 얻은 reward 누적값이고, 평가 return은 정해진 평가 조건에서 현재 정책을 따로 실행해 측정한 reward 누적값이다.

### 직관

학습 중의 행동은 탐색과 정책 업데이트의 영향을 받기 때문에 들쭉날쭉하다. 반면 평가는 같은 시나리오와 seed에서 현재 모델을 다시 실행해 보는 것이므로, 모델이 실제로 좋아졌는지 비교하기 더 쉽다.

### 프로젝트에서의 적용

단계 6 trainer는 학습 seed와 평가 seed를 분리한다. 학습 중 일정 timestep마다 tiny 시나리오를 평가하고, `data/runs/<run-id>/metrics/training-metrics.jsonl`에 평가 결과를 기록한다.

평가 결과에는 총 return뿐 아니라 다음 도메인 지표를 함께 본다.

- 기본 우선순위 점수
- 각도 보너스
- 미완료 패널티
- 완료 strip 수
- 완료 order 수

이렇게 해야 단순히 reward 합계가 오른 것인지, 정말 더 가치 있는 촬영 스케줄을 만든 것인지 구분할 수 있다.

### 주의할 점

- 고정 tiny 시나리오에서 높은 평가 점수는 일반화 성능이 아니라 해당 시나리오에 대한 최적화 결과일 수 있다.
- 평가 seed를 학습 seed와 분리해도 시나리오 자체가 같으면 완전한 일반화 평가는 아니다.
- Random valid보다 나은지 판단할 때는 한 번의 실행보다 여러 seed와 충분한 학습 길이에서 일관성을 확인해야 한다.

### 학습 기록

- 2026-07-07: Maskable PPO trainer 구현 과정에서 학습 return과 고정 평가 return을 분리해 기록해야 하는 이유를 정리했다.

## 단계 6 성능 검증의 의미

### 핵심 정의

단계 6 성능 검증은 같은 tiny 시나리오에서 Maskable PPO로 학습한 정책과 `RandomValidPolicy` 기준선을 비교해, 학습 정책이 무작위 유효 선택보다 나은 결과를 일관되게 내는지 확인한 것이다.

### 직관

`RandomValidPolicy`는 실행 가능한 촬영 후보 중 하나를 무작위로 고른다. 이 기준선을 넘는다는 것은 에이전트가 단순히 불가능한 행동을 피한 수준이 아니라, 가능한 후보 사이에서 조금이라도 더 좋은 선택 패턴을 학습했다는 뜻이다.

### 프로젝트에서의 적용

이번 검증은 `synthetic-tiny-20260707`에서 진행했다. Maskable PPO는 action mask로 시간, 자세, 중복 촬영 제약을 지키면서 후보를 고르고, Random valid도 같은 action mask 안에서 무작위로 후보를 고른다. 따라서 비교의 핵심은 `제약을 지켰는가`가 아니라 `제약을 지키는 후보 중 무엇을 고르는가`다.

단계 6 결과에서는 PPO median return이 Random valid median return보다 근소하게 높았고, skip 반복이나 특정 action slot 고착도 관찰되지 않았다. 이를 근거로 단계 6의 “학습, 저장, 평가, baseline 비교 파이프라인”은 완료로 본다.

Random valid의 `valid`는 Maskable PPO가 학습 과정에서 좋다고 판단한 행동이라는 뜻이 아니다. 시뮬레이터가 현재 state에서 하드 제약을 검사했을 때 action mask가 `True`인 후보라는 뜻이다. Random valid는 그 후보 slot 목록을 만들고, seed가 고정된 난수로 그중 하나를 고른다. 현재 시각에 유효한 촬영 후보가 하나도 없으면 항상 가능한 `skip=0`을 선택한다.

여기서 시뮬레이터는 위성 자체의 정밀 궤도 물리 시뮬레이터라기보다, 이미 주어진 pass, strip, opportunity를 이용해 하루 동안 어떤 촬영 선택이 가능한지 진행시키는 스케줄링 시뮬레이터다. 현재 시각, 완료된 strip, 위성 자세, 이전 촬영 종료 시각을 상태로 들고 있다가 action을 적용하고 다음 의사결정 시점으로 이동한다.

### 주의할 점

- 이번 결과는 tiny 시나리오 기준 검증이지 full 규모 스케줄링 성능 보장이 아니다.
- 두 정책의 완료 strip/order 수는 같았고 차이는 주로 각도 보너스에서 발생했다.
- 그래서 “Maskable PPO가 이 문제를 완전히 잘 푼다”보다는 “Maskable PPO 학습 정책이 Random valid 기준선보다 나은지 검증하는 체계가 작동했고, tiny 기준은 통과했다”로 이해하는 것이 정확하다.

### 학습 기록

- 2026-07-08: 단계 6 검증을 Maskable PPO 정책과 Random valid 기준선 비교로 이해하되, tiny 시나리오 검증의 한계를 함께 구분했다.
- 2026-07-08: Random valid의 `valid`가 학습된 좋은 정책이 아니라 action mask로 허용된 실행 가능 후보임을 정리했다.
- 2026-07-08: 프로젝트의 시뮬레이터가 정밀 궤도 물리보다 촬영 스케줄 진행과 제약 판정을 담당한다는 점을 정리했다.
- 2026-07-08: 단계 6 검증은 `유효 정책들` 중 무작위 선택이 아니라, 시뮬레이터가 유효하다고 판정한 action 후보 중 무작위 선택하는 Random valid 정책과 Maskable PPO 학습 정책의 return 비교임을 정리했다.
