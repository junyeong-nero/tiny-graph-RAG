# Entity Resolution 동작 방식

이 문서는 `tiny_graph_rag/graph/entity_resolution.py`의 `LLMEntityResolver`가 그래프 엔티티를 어떻게 병합하는지 설명합니다.

## 1) 언제 실행되는가

- `GraphRAG` 초기화 시 `GraphBuilder(resolver=LLMEntityResolver(...))`가 주입됩니다.
- 문서 처리 후 `GraphBuilder.build()`에서 `resolver.resolve(graph)`가 호출됩니다.
- 결과적으로 entity resolution은 그래프 빌드 마지막 단계에서 **in-place**로 수행됩니다.

## 2) 전체 처리 순서

`resolve()`는 아래 순서로 동작합니다.

1. `_merge_explicit_alias_relationships`
2. `_merge_strong_contextual_aliases`
3. `_merge_role_bucket_aliases`
4. `_reassign_conflicting_aliases`
5. person-like 엔티티만 추려 LLM 배치 병합 (`_resolve_batch` + `_apply_merge_groups`)

앞의 1~4는 규칙 기반(휴리스틱), 5는 LLM 기반입니다.

## 3) person-like 판단

`_is_person_like_entity()` 기준:

- `entity_type == "PERSON"` 이면 True
- 아니어도(`OTHER` 포함) 이름/설명에 키워드(`아내`, `남편`, `환자`, `driver` 등)가 있으면 True

이 기준으로 role/alias 표현(예: `병인`, `인력거꾼`)도 병합 후보에 들어갑니다.

## 4) 휴리스틱 병합 규칙

### 4.1 명시적 alias 관계 병합

- 관계 타입이 `ALIAS_OF` 또는 `SAME_AS`인 엔티티 쌍을 즉시 병합합니다.
- 양쪽 모두 person-like여야 합니다.

### 4.2 강한 문맥 병합

다음 조건을 모두 만족하면 병합합니다.

- 두 엔티티 사이 직접 관계 타입이 `RELATED_TO|REFERS_TO|SAME_AS|ALIAS_OF` 중 하나
- 공유 이웃이 1개 이상 (`_count_shared_neighbors`)
- 둘 중 하나 이상이 reference-like (`OTHER` 타입, 일반 역할명, person-like 키워드 포함)

병합이 발생하면 그래프가 바뀌므로 while-loop로 다시 탐색합니다.

### 4.3 Role bucket 병합

- 동일 버킷(기본값: `spouse_patient`)에 속하고 공유 이웃이 있으면 병합합니다.
- 기본 버킷 용어 예: `아내`, `마누라`, `병인`, `환자`, `오라질년` 등

### 4.4 alias 재배치

- `reassign_aliases=True`인 버킷에 대해 canonical 엔티티를 먼저 선택합니다.
- 다른 엔티티의 alias 중 버킷 용어를 canonical으로 이동해 오탐 alias를 정리합니다.
- 변경 후 `_rebuild_entity_name_index()`를 호출해 이름/alias 인덱스를 재구성합니다.

## 5) LLM 배치 병합

### 5.1 입력 구성

`max_entities_per_pass`(기본 80) 단위로 분할하여 LLM에 보냅니다.

- `entity_id`, `name`, `entity_type`, `description`, `source_chunks`, `aliases`
- `neighbors`(최대 12개): 관계 타입, 상대 엔티티명/타입, 관계 설명

### 5.2 출력 형식

LLM은 `merge_groups` JSON을 반환해야 합니다.

```json
{
  "merge_groups": [
    {
      "canonical_entity_id": "id_to_keep",
      "duplicate_entity_ids": ["id_to_merge"],
      "confidence": 0.95,
      "reason": "same character"
    }
  ]
}
```

API 오류/파싱 이상 시 해당 배치는 빈 결과로 처리됩니다.

## 6) merge_groups 적용 로직

`_apply_merge_groups()`는 union-find로 전이적 병합을 처리합니다.

- `confidence < min_confidence`(기본 0.75)는 무시
- canonical/duplicate가 현재 그래프에 존재하고 person-like일 때만 반영
- 직접 관계가 `MARRIED_TO`, `PARENT_OF`, `CHILD_OF`, `SIBLING_OF`, `FRIEND_OF`, `KNOWS`면 병합 금지

클러스터별 canonical은 `_select_canonical_id()`로 선택합니다.

- 점수: `(LLM canonical vote 수, PERSON 보너스, 일반 역할명 패널티, 관계 수)`
- 최고 점수 엔티티를 남기고 나머지를 `graph.merge_entities()`로 병합

## 7) 병합 후 보장되는 효과

- 중복 엔티티 제거 및 관계 endpoint 재매핑
- self-loop 제거
- 중복 관계 제거
- 병합된 이름은 alias로 보존되어 `get_entity_by_name()`에서 조회 가능

관련 테스트는 `tests/test_graph.py`의 `TestEntityResolution` 섹션을 참고하면 됩니다.
