# 단일 RAG 파이프라인을 이용한 한국어·영어 QA 데이터셋 성능 및 특성 비교 분석

#### 1) Exact Match (EM)
- 모델의 최종 답변이 정답 후보(`gold_answers`) 중 하나와 정규화 후 완전히 동일하면 1, 아니면 0
- 정규화: 소문자화, 구두점/공백 정리, 숫자 쉼표 제거 등
#### 2) Recall@K
- 검색 결과 상위 K개(`retrieved_ids[:K]`) 안에
- 정답 근거 문단(`gold_evidence_ids`)이 포함되면 1, 아니면 0
검색기가 정답 문단을 가져왔는지 평가
#### 3) Attribution (%)
- 모델이 인용한 문단(`cited_ids`) 중
- 정답 근거 문단(`gold_evidence_ids`)과 ID 교집합이 있으면 1, 아니면 0
- 근거 일치 여부(hallucinated citation 여부)를 평가
#### 4) p95 Latency
- 모든 문항의 `latency_ms` 에서 95퍼센타일 지연 시간
- tail latency(응답이 느린 상위 5% 구간)를 측정
#### 5) ΔEM (Paraphrase Robustness)
- 패러프레이즈된 질문과 원문 질문의 EM 차이
- 각 쌍에 대해 `EM(para) - EM(orig)` 을 계산한 평균
- 음수일수록 질문 표현 변화에 취약
