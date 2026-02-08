# Evaluation Guide

이 문서는 Tiny-Graph-RAG의 성능을 정량적으로 측정하기 위한 평가 시스템의 구성 요소와 메트릭, 그리고 실행 방법을 설명합니다.

## 1. 평가 메트릭 (Metrics)

Retrieval 성능을 평가하기 위해 다음과 같은 4가지 핵심 지표를 사용합니다. 모든 지표는 0.0에서 1.0 사이의 값을 가지며, 1.0에 가까울수록 성능이 우수함을 의미합니다.

### 1.1 Precision@k (정밀도)
검색된 상위 `k`개의 엔티티 중 실제 정답(Ground Truth)에 포함된 엔티티의 비율입니다.
- **공식**: `|Retrieved@k ∩ Relevant| / k`
- **의미**: 시스템이 추출한 결과물 중 얼마나 많은 것이 실제로 유용한 정보인가를 측정합니다.

### 1.2 Recall@k (재현율)
전체 정답 엔티티 중 검색된 상위 `k`개의 결과에 포함된 엔티티의 비율입니다.
- **공식**: `|Retrieved@k ∩ Relevant| / |Relevant|`
- **의미**: 시스템이 전체 관련 정보 중에서 얼마나 많이 놓치지 않고 찾아냈는가를 측정합니다.

### 1.3 MRR (Mean Reciprocal Rank)
사용자가 원하는 정답이 검색 결과의 몇 번째 순위에 처음으로 등장하는지를 측정합니다.
- **공식**: `1 / rank_of_first_relevant_item`
- **의미**: 시스템이 정답을 얼마나 상위권에 배치하는지 평가합니다.

### 1.4 nDCG@k (Normalized Discounted Cumulative Gain)
검색된 결과의 순위를 고려한 이득(Gain)의 합계를 측정합니다. 정답이 상위에 있을수록 높은 점수를 부여합니다.
- **특징**: 이진 관련성(Binary Relevance)을 기반으로 계산하며, 검색 결과의 순서가 얼마나 최적인지를 종합적으로 평가합니다.

## 2. 데이터셋 구조 (Dataset Structure)

평가 데이터셋은 JSONL(Line-delimited JSON) 형식으로 제공됩니다. 각 라인은 하나의 평가 예제(`EvalExample`)를 나타냅니다.

### 2.1 스키마 상세

| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `query` | `string` | **(필수)** 평가를 위한 자연어 질의 |
| `reference_entities` | `list[string]` | **(필수)** 해당 질의에 답하기 위해 반드시 찾아야 하는 엔티티 목록 |
| `id` | `string` | 예제 식별자 (선택) |
| `ground_truth` | `string` | 최종 답변 정답 (선택) |
| `tags` | `list[string]` | 예제 성격 구분 태그 (예: `multi-hop`, `alias-noise`) |

### 2.2 하드셋 (Hardset) 구성
`data/eval/*-hardset.jsonl` 파일은 다음과 같은 고난도 시나리오를 포함합니다.
- **Multi-hop**: 2단계 이상의 관계를 거쳐야 정보를 찾을 수 있는 질의 (`2-hop`, `3-hop` 태그)
- **Alias-noise**: 지문에서 지칭하는 이름과 질문에서의 이름이 다른 경우
- **Coreference**: '그', '그녀' 등 대명사 해결이 필요한 경우

## 3. 평가 실행 방법

### 3.1 기본 실행
```bash
uv run python main.py eval \
  --dataset "김유정-동백꽃-eval.jsonl" \
  -g "김유정-동백꽃-KG.json" \
  -o "김유정-동백꽃-eval-results.json"
```

### 3.2 고급 실행 (경로 지정)
다른 폴더에 있는 데이터셋이나 그래프를 사용할 경우:
```bash
uv run python main.py eval \
  --dataset-dir "custom_data/eval" \
  --dataset "test-set.jsonl" \
  --kg-dir "custom_data/kg" \
  --results-dir "custom_data/results"
```

### 3.3 주요 옵션 설명
- `--top-k`: 메트릭 계산 시 기준이 되는 상위 `k`값 (기본값: 5)
- `--hops`: BFS 탐색 깊이 (기본값: 2). 하드셋 평가 시 3~4로 설정을 권장합니다.
- `--skip-generation`: 답변 생성을 생략하고 검색(Retrieval) 품질만 측정하여 비용을 절감합니다.
- `--kg-dir`: 상대 경로 그래프 파일의 기본 폴더 (기본값: `config.yaml`의 `storage.kg_dir`)
- `--dataset-dir`: 상대 경로 평가셋 파일의 기본 폴더 (기본값: `config.yaml`의 `storage.dataset_dir`)
- `--results-dir`: 상대 경로 결과 파일의 기본 폴더 (기본값: `config.yaml`의 `storage.results_dir`)

## 4. 출력 결과 해석

평가 완료 후 생성되는 JSON 파일의 `summary` 섹션을 통해 전체적인 성능을 한눈에 확인할 수 있습니다.

```json
{
  "summary": {
    "mean_precision_at_k": 0.85,
    "mean_recall_at_k": 0.72,
    "mean_mrr": 0.9,
    "mean_ndcg_at_k": 0.88,
    "total_token_usage": 15400,
    "estimated_cost_usd": 0.02
  },
  "results": [...]
}
```

## 5. 데이터셋 파일 규칙
권장 폴더 구조는 다음과 같습니다.
- `data/eval`: 평가셋
- `data/kg`: 생성된 KG
- `data/results`: 평가 결과

파일 접미사 규칙:
- `-eval.jsonl`: 일반적인 사실 확인형 질의응답 셋
- `-hardset.jsonl`: 그래프 탐색 성능을 극한으로 테스트하는 고난도 셋
- `-KG.json`: 생성된 지식 그래프
- `-results.json`: 평가 실행 결과 기록
