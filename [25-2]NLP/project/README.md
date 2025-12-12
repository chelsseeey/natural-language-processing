# 단일 RAG 파이프라인을 이용한 한국어·영어 QA 데이터셋 성능 및 특성 비교 분석

## 프로젝트 개요

본 프로젝트는 한국어와 영어, 사실 기반과 상식 추론 문제를 포함한 4개 데이터셋(KLUE-MRC, KorQuAD, SQuAD, CoS-E)에 대해 단일 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하고 성능을 평가합니다.

### 주요 목표
- 단일 파이프라인 기반의 통제된 비교 실험 수행
- 지식 유형(Fact vs. Commonsense)에 따른 검색 효용성 정량화
- 언어적 특성 및 문서 구조에 따른 RAG 성능 차이 분석
- 질문 표현 변화에 대한 강건성(Robustness) 평가

## 프로젝트 구조

```
project/
├── dataset/                          # 데이터셋 및 생성된 파일들
│   ├── KLUE-MRC/                     # KLUE-MRC 데이터셋
│   │   ├── klue-mrc-v1.1_dev.json   # 원본 데이터
│   │   ├── klue_mrc_300.json         # 샘플링된 300개 데이터
│   │   ├── bm25_index.pkl           # BM25 인덱스 파일
│   │   └── KLUE-MRC_corpus.jsonl     # 생성된 코퍼스
│   ├── KorQuAD2.1/                   # KorQuAD 2.1 데이터셋
│   │   └── korquad_300.json          # 샘플링된 300개 데이터
│   │   └── KorQuAD_corpus.jsonl      # 생성된 코퍼스
│   ├── SQuAD/                        # SQuAD 데이터셋
│   │   ├── dev-v1.1.json            # 원본 데이터
│   │   ├── squad_300.json           # 샘플링된 300개 데이터
│   │   ├── bm25_index.pkl           # BM25 인덱스 파일
│   │   └── SQuAD_corpus.jsonl       # 생성된 코퍼스
│   └──  CoS-E/                        # CoS-E 데이터셋
│       ├── cose_dev_v1.11.jsonl     # 원본 데이터
│       ├── cose_300.json            # 샘플링된 300개 데이터
│       ├── bm25_index.pkl           # BM25 인덱스 파일
│       └── CoS-E_corpus.jsonl       # 생성된 코퍼스
├── reports/                          # 실험 결과 파일들
│   ├── {dataset}_final_results.json # 각 데이터셋별 RAG 실험 결과
│   ├── {dataset}_robustness.json    # 각 데이터셋별 강건성 테스트 결과
│   ├── evaluation_summary.json       # 전체 평가 지표 요약
│   └── robustness_summary.json      # 강건성 테스트 요약
├── corpus.py                        # 코퍼스 생성 스크립트
├── preprocessor.py                   # 텍스트 전처리 모듈
├── run_rag.py                        # RAG 파이프라인 실행 스크립트
├── evaluate_metrics.py               # 평가 지표 계산 스크립트 
├── run_robustness.py                 # 강건성 테스트 스크립트
└── final_report.ipynb                # 최종 보고서 (Jupyter Notebook)
```

## 설치 및 설정

### 필수 패키지

```bash
pip install openai rank-bm25 kiwipiepy numpy tqdm
```

### API 키 설정

`run_rag.py`와 `run_robustness.py` 파일에서 OpenAI API 키를 설정해야 합니다:

```python
OPENAI_API_KEY = "your-api-key-here"
```

## 사용 방법

### 1. 코퍼스 생성

각 데이터셋의 원본 파일을 읽어서 RAG용 코퍼스 파일을 생성합니다:

```bash
python corpus.py
```

**출력:**
- `dataset/{dataset}_corpus.jsonl`: 각 데이터셋별 코퍼스 파일

### 2. RAG 실험 실행

각 데이터셋에 대해 RAG 파이프라인을 실행합니다:

```bash
python run_rag.py
```

**설정:**
- `run_rag.py`의 `targets` 리스트에서 실행할 데이터셋을 선택
- `limit` 파라미터로 샘플 수 조절 (None이면 전체 실행)

**출력:**
- `dataset/{dataset}_final_results.json`: 각 데이터셋별 실험 결과
  - `no_rag_answer`: 외부 지식 없이 생성된 답변
  - `rag_answer`: 검색된 문서를 참고하여 생성된 답변
  - `gold_answers`: 정답
  - `recall_success`: 검색 성공 여부
  - `latency`: 응답 생성 시간 (ms)

### 3. 평가 지표 계산

실험 결과를 바탕으로 평가 지표를 계산합니다:

```bash
python evaluate_metrics.py
```

**계산되는 지표:**
- **Strict EM**: 정확히 일치하는 답변 비율
- **Soft EM**: 정답 키워드가 포함된 답변 비율
- **F1 Score**: 토큰 단위 겹침 비율의 조화 평균
- **Recall@3**: 상위 3개 검색 결과 중 정답 문서 포함 비율
- **Attribution**: Soft EM과 Recall이 모두 1인 비율
- **p95 Latency**: 95번째 백분위 응답 시간

**출력:**
- `dataset/evaluation_summary.json`: 전체 평가 지표 요약

### 4. 강건성 테스트

질문 표현 변화에 대한 시스템의 강건성을 평가합니다:

```bash
python run_robustness.py
```

**설정:**
- `run_robustness.py`의 `targets` 리스트에서 실행할 데이터셋을 선택
- `sample_size` 파라미터로 샘플 수 조절

**출력:**
- `dataset/{dataset}_robustness.json`: 각 데이터셋별 강건성 테스트 결과
  - `original_q`: 원본 질문
  - `paraphrased_q`: 패러프레이즈된 질문
  - `original_answer`: 원본 질문에 대한 답변
  - `paraphrased_answer`: 패러프레이즈 질문에 대한 답변
  - `em_orig`, `em_para`: 각각의 EM 점수
  - `delta_em`: 성능 변화량
- `dataset/robustness_summary.json`: 강건성 테스트 요약

## 주요 모듈 설명

### `preprocessor.py`
- **CustomPreprocessor**: 다국어 텍스트 전처리 클래스
  - 한국어: Kiwi 형태소 분석기를 활용한 형태소 단위 토큰화
  - 영어: 소문자 변환 및 구두점 제거 후 공백 단위 토큰화
  - HTML 태그 제거 (KorQuAD용)

### `run_rag.py`
- **SimpleRAGRunner**: RAG 파이프라인 실행 클래스
  - BM25 기반 문서 검색
  - OpenAI GPT-4o-mini를 활용한 답변 생성
  - No-RAG와 RAG 모드 비교

### `evaluate_metrics.py`
- **RAGEvaluator**: RAG 평가 지표 계산 클래스
  - Strict EM, Soft EM, F1 Score 계산
  - Recall, Attribution 계산
  - p95 Latency 계산

### `run_robustness.py`
- **RobustnessTester**: 강건성 테스트 클래스
  - 질문 패러프레이징 생성
  - 원본/패러프레이즈 질문에 대한 RAG 수행
  - 성능 변화량(Δ EM) 계산

## 실험 결과

### 주요 발견사항

1. **사실 기반 질문에서의 RAG 성공**
   - KLUE-MRC: RAG Gain +67.00%p
   - SQuAD: RAG Gain +56.33%p
   - 키워드 매칭 방식인 BM25가 효과적으로 작동

2. **상식 추론 질문에서의 RAG 역효과**
   - CoS-E: RAG Gain -6.33%p
   - 질문과 정답 사이 단어 중복 부족으로 검색 실패
   - 정보 간섭(Distraction) 현상 발생

3. **강건성 분석**
   - 모든 데이터셋에서 패러프레이즈 후 성능 하락
   - CoS-E에서 최대 낙폭(-7.33%p)

자세한 결과는 `final_report.ipynb`를 참고하세요.

## 평가 지표

### 정확도 지표
- **Strict EM**: 정확히 일치하는 답변 비율
- **Soft EM**: 정답 키워드가 포함된 답변 비율
- **F1 Score**: 토큰 단위 겹침 비율의 조화 평균

### 검색 지표
- **Recall@k**: 상위 k개 검색 결과 중 정답 문서 포함 비율
- **Attribution**: Soft EM과 Recall이 모두 1인 비율 (정답이면서 검색 성공)

### 성능 지표
- **p95 Latency**: 95번째 백분위 응답 시간 (ms)
- **RAG Gain**: RAG와 No-RAG의 성능 차이 (%p)

### 강건성 지표
- **Δ EM**: 패러프레이즈 질문과 원본 질문의 EM 점수 차이 (%p)

## 데이터셋 정보

| 데이터셋 | 언어 | 지식 유형 | 샘플 수 | 설명 |
|---------|------|----------|---------|------|
| KLUE-MRC | 한국어 | 사실 기반 | 300 | 뉴스 기사 기반 질의응답 |
| KorQuAD | 한국어 | 사실 기반 | 300 | 위키백과 기반 질의응답 |
| SQuAD | 영어 | 사실 기반 | 300 | 위키백과 기반 질의응답 |
| CoS-E | 영어 | 상식 추론 | 300 | 상식 추론 질의응답 |




