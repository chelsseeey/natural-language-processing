import numpy as np
import string, re

class RAGEvaluator:
    def __init__(self, ignore_missing_evidence: bool = True, k: int = 10):
        self.ignore_missing_evidence = ignore_missing_evidence
        self.k = k

    # --- 정규화 ---
    def normalize_answer(self, s: str) -> str:
        if s is None:
            return ""
        # 소문자(영문만 영향), 공백 정리
        s = s.lower().strip()
        # 구두점 제거
        s = ''.join(ch for ch in s if ch not in set(string.punctuation))
        # 숫자 쉼표 제거: 1,234 -> 1234
        s = re.sub(r'(?<=\d),(?=\d)', '', s)
        # 한국어 날짜/단위 흔한 변형 정리 (가볍게)
        s = s.replace(' 년', '년').replace(' 월', '월').replace(' 일', '일')
        # 관사 제거(영어일 때만 사실상 영향)
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        # 공백 정규화
        s = ' '.join(s.split())
        return s

    # --- EM: 다중 정답 지원 ---
    def exact_match(self, prediction: str, gold_list) -> int:
        """
        gold_list: list[str] 또는 str
        """
        if prediction is None:
            return 0
        if isinstance(gold_list, str):
            gold_list = [gold_list]
        pred_n = self.normalize_answer(prediction)
        for g in (gold_list or []):
            if pred_n == self.normalize_answer(g):
                return 1
        return 0

    # --- Recall@K ---
    def recall_at_k(self, retrieved_ids, gold_ids, k: int = None) -> tuple[int, bool]:
        """
        return: (score(0/1) 또는 0, counted) 
        counted=False 이면 분모에서 제외(옵션)
        """
        if k is None:
            k = self.k
        if not gold_ids:
            # gold evidence가 없으면 제외할지 여부
            return (0, not self.ignore_missing_evidence)
        top_k = set((retrieved_ids or [])[:k])
        inter = top_k.intersection(set(gold_ids))
        return (1 if len(inter) > 0 else 0, True)

    # --- Attribution% (ID 교집합) ---
    def attribution(self, cited_ids, gold_ids) -> tuple[int, bool]:
        """
        모델 인용 ID(cited_ids) ∩ gold_ids ≠ ∅ 이면 1
        """
        if not gold_ids:
            return (0, not self.ignore_missing_evidence)
        if not cited_ids:
            return (0, True)
        inter = set(cited_ids).intersection(set(gold_ids))
        return (1 if len(inter) > 0 else 0, True)

    # --- p95 ---
    def p95_latency(self, latencies_ms):
        if not latencies_ms:
            return float('nan')
        return float(np.percentile(latencies_ms, 95))

    # --- 배치 평가 ---
    def evaluate_batch(self, results: list[dict]) -> dict:
        """
        각 dict 예:
        {
          "query_id": str,
          "prediction": str,
          "gold_answers": list[str] 또는 str,
          "gold_evidence_ids": list[str],  
          "retrieved_ids": list[str],
          "cited_ids": list[str],
          "latency_ms": float
        }
        """
        n = len(results)
        if n == 0:
            return {}

        em_sum = 0
        rec_sum = 0; rec_cnt = 0
        attr_sum = 0; attr_cnt = 0
        lat = []

        for r in results:
            em_sum += self.exact_match(r.get('prediction'), r.get('gold_answers'))

            rec, counted = self.recall_at_k(
                r.get('retrieved_ids'), r.get('gold_evidence_ids'))
            if counted: 
                rec_sum += rec; rec_cnt += 1

            attr, counted = self.attribution(
                r.get('cited_ids'), r.get('gold_evidence_ids'))
            if counted: 
                attr_sum += attr; attr_cnt += 1

            if r.get('latency_ms') is not None:
                lat.append(r['latency_ms'])

        out = {
            "EM": em_sum / n,
            "Recall@%d" % self.k: (rec_sum / rec_cnt) if rec_cnt > 0 else float('nan'),
            "Attribution": (attr_sum / attr_cnt) if attr_cnt > 0 else float('nan'),
            "p95_Latency_ms": self.p95_latency(lat),
            "counts": {"N": n, "N_recall": rec_cnt, "N_attr": attr_cnt}
        }
        return out
