# Tiny-Graph-RAG

Tiny-Graph-RAG는 OpenAI API를 사용하여 텍스트에서 엔티티와 관계를 추출하고, JSON 기반 지식 그래프를 구축한 뒤 그래프 탐색을 통해 질의응답 컨텍스트를 구성하는 경량형 Graph RAG 프레임워크입니다.

벡터 DB 기반의 시맨틱 검색 대신, 엔티티 간의 명시적인 연결 구조(BFS 탐색 + 휴리스틱 랭킹)를 활용하여 지식 추출부터 추론까지의 과정을 투명하게 관리하는 데 초점을 맞춥니다.

## 주요 특징 (Key Features)

- **Lightweight Graph Storage**: 복잡한 그래프 DB 없이 단순 JSON 파일로 지식 그래프를 관리하고 저장합니다.
- **LLM-Powered Extraction**: LLM을 통해 비정형 텍스트에서 엔티티, 타입, 설명 및 관계 정보를 정밀하게 추출합니다.
- **Advanced Retrieval**: BFS 기반의 서브그래프 확장과 휴리스틱 랭킹을 조합하여 질문과 관련된 맥락을 효과적으로 포착합니다.
- **Entity Resolution**: LLM과 규칙 기반 병합 로직을 사용하여 중복 엔티티(별칭, 오타 등)를 정리합니다.
- **Evaluation Pipeline**: Precision, Recall, MRR, nDCG 등 다양한 메트릭을 통한 검색 품질 평가 도구를 제공합니다.
- **Visualization**: Pyvis를 활용하여 구축된 지식 그래프를 대화형 HTML로 시각화할 수 있습니다.

## 아키텍처 요약 (Architecture)

```text
[ Document ] 
     |
     v
[ TextChunker ] -> Overlapping text segments
     |
     v
[ EntityRelationshipExtractor ] -> LLM-based JSON extraction
     |
     v
[ GraphBuilder ] -> Entity Resolution & Graph Construction
     |
     v
[ KnowledgeGraph ] -> JSON storage (.json)
     |
     v
[ GraphRetriever ] -> Query Entity Extraction -> BFS Traversal -> Ranking
     |
     v
[ LLM Answer Generator ] -> Context-aware response
```

상세한 모듈별 설명은 [docs/README.md](docs/README.md)를 참고하세요.
- [Chunking 가이드](docs/chunking.md)
- [Entity Resolution 가이드](docs/entity-resolution.md)

## 시작하기 (Getting Started)

### 요구 사항
- Python 3.13+
- OpenAI API Key (또는 호환되는 API 엔드포인트)

### 설치
```bash
uv sync
export OPENAI_API_KEY="your-api-key"
```

기본 모델 설정 및 청킹 파라미터는 `config.yaml`에서 관리할 수 있습니다.

## 주요 기능 실행 방법 (CLI)

### 1. 그래프 생성
```bash
uv run python main.py process "data/novels/김유정-동백꽃.txt" -o "김유정-동백꽃-KG.json"
```

### 2. 질의응답 (Query)
```bash
uv run python main.py query "점순이와 우리 수탉의 관계를 설명해줘." -g "김유정-동백꽃-KG.json"
```

### 3. 통계 및 시각화
```bash
# 통계 확인
uv run python main.py stats -g "김유정-동백꽃-KG.json"

# 시각화 HTML 생성
uv run python main.py visualize -g "김유정-동백꽃-KG.json" -o graph_viz.html
```

### 4. Streamlit Web UI
```bash
uv run streamlit run streamlit_app.py
```

### 5. Entity Resolution 일괄 적용 (Batch)
기존에 생성된 모든 그래프(`data/kg/*.json`)에 대해 ER을 재적용합니다.
```bash
uv run python scripts/apply_er.py
```

## 평가 (Evaluation)

평가 모듈은 검색 품질을 정량적으로 측정합니다. 출력 JSON에는 예제별 메트릭과 전체 요약(지연 시간, 토큰 사용량, 예상 비용)이 포함됩니다.

### 실행 예시
```bash
uv run python main.py eval \
  --dataset "김유정-동백꽃-eval.jsonl" \
  -g "김유정-동백꽃-KG.json" \
  -o "김유정-동백꽃-eval-results.json"
```

상세 옵션 및 데이터셋 형식은 [docs/evaluation.md](docs/evaluation.md)를 확인하세요.

### 벤치마크 결과 (Sample Results)

한국 근대 단편 소설 데이터셋(`data/novels`)에 대한 검색 성능 지표입니다. (Top-K=5, Hops=2~4 기준)

| 데이터셋 (작품명) | 유형 | Recall@5 | MRR | nDCG@5 |
| :--- | :--- | :--- | :--- | :--- |
| **김유정-동백꽃** | 일반 | 1.00 | 0.95 | 0.96 |
| **김유정-동백꽃** | Hardset | 0.87 | 0.79 | 0.81 |
| **현진건-운수좋은날** | 일반 | 1.00 | 0.95 | 0.97 |
| **이상-날개** | 일반 | 1.00 | 0.96 | 0.98 |

*※ Hardset은 Multi-hop 질의와 인물 별칭(Alias) 노이즈를 포함하고 있어 일반셋보다 난이도가 높습니다.*

## 테스트
```bash
uv run pytest
```

## 데이터 저장 구조
기본 `config.yaml` 기준으로 입력/산출물을 폴더별로 분리합니다:
- `data/novels/`: 원문 텍스트 (`<작품명>.txt`)
- `data/eval/`: 평가셋 (`<작품명>-eval.jsonl`, `<작품명>-hardset.jsonl`)
- `data/kg/`: 생성된 그래프 (`<작품명>-KG.json`)
- `data/results/`: 평가 결과 (`<작품명>-eval-results.json`, `<작품명>-hardset-results.json`)

CLI는 상대 경로를 각 기본 폴더로 자동 해석합니다. 필요하면 `--kg-dir`, `--dataset-dir`, `--results-dir` 또는 환경 변수 `KG_DIR`, `DATASET_DIR`, `RESULTS_DIR`로 덮어쓸 수 있습니다.

## 라이선스
MIT
