# Tiny-Graph-RAG Docs

이 문서는 현재 저장소 코드 기준으로 프로젝트 목적, 모듈 책임, retrieval 원리/제약을 간단히 정리합니다.

## 1) 프로젝트 목적과 범위

- 문서 텍스트에서 엔티티/관계를 추출해 지식 그래프를 만드는 실험용 Graph RAG 구현
- 검색 단계에서 그래프 기반 탐색(BFS)과 휴리스틱 랭킹을 사용
- Retrieval 품질 평가를 위한 JSONL 기반 평가 파이프라인 제공
- 범위 밖: 벡터 DB, 학습 파이프라인, 대규모 분산 처리, 고급 entity linking

## 2) 모듈 구조와 책임

소스 루트: `tiny_graph_rag/`

| 모듈 | 주요 파일 | 책임 |
| --- | --- | --- |
| `chunking` | `chunker.py` | 텍스트를 overlap 포함 청크로 분할 |
| `extraction` | `extractor.py`, `parser.py`, `prompts.py` | LLM JSON 응답으로 엔티티/관계 추출 및 파싱 |
| `graph` | `models.py`, `builder.py`, `storage.py` | 그래프 모델, 엔티티 병합, JSON 저장/로드 |
| `retrieval` | `retriever.py`, `traversal.py`, `ranking.py` | 질의 엔티티 추출, BFS 서브그래프 확장, 랭킹/컨텍스트 구성 |
| `llm` | `client.py`, `prompts.py` | OpenAI(OpenAI-compatible) 호출 래퍼, 응답 생성 프롬프트 |
| `evaluation` | `dataset.py`, `metrics.py`, `runner.py` | 데이터셋 로딩, Precision/Recall/MRR/nDCG 계산, 비용/지연 집계 |
| `visualization` | `pyvis_visualizer.py` | HTML 그래프 시각화 생성 |

엔트리포인트:

- CLI: `main.py`
- API-like facade: `tiny_graph_rag/__init__.py` (`GraphRAG`)
- Web UI: `streamlit_app.py`

## 3) Retrieval 파이프라인 원리

`GraphRetriever.retrieve()` 흐름:

1. 질문에서 엔티티명 추출(LLM JSON)
2. 그래프 내 엔티티 exact 매칭(`name.lower().strip()` 기반)
3. 매칭 실패 시 전체 엔티티 대상 휴리스틱 fuzzy 랭킹
4. seed 엔티티 주변 BFS 확장(`hops`)
5. 서브그래프 엔티티/관계 정리 후 top-k 중심 재정렬
6. 텍스트 컨텍스트(`Entities`, `Relationships`) 생성

## 4) 현재 구현의 한계

- 엔티티 정규화가 단순(lower/strip)하여 별칭, 띄어쓰기 변형에 취약
- 랭킹이 토큰 포함 기반 휴리스틱이라 의미 유사도(embedding)를 직접 반영하지 않음
- `get_neighbors()`가 관계를 양방향으로 취급하므로 방향성 정보 손실 가능
- hard multi-hop 질의는 `--hops` 설정에 성능이 크게 의존
- 추출/생성 품질이 LLM 응답 일관성에 영향을 받음

## 5) 기여 시 체크 포인트

- CLI 인자와 문서 예시 커맨드가 일치하는지 확인
- `data/novels` 파일 네이밍 규칙 준수 (`-eval`, `-hardset`, `-results`)
- 평가 결과 공유 시 `--top-k`, `--hops`, `--skip-generation` 설정값 명시

평가셋/실행 상세는 `docs/evaluation.md` 참고.

## 6) 추가 문서

- 평가 실행 가이드: `docs/evaluation.md`
- 엔티티 병합(Entity Resolution) 동작 상세: `docs/entity-resolution.md`
