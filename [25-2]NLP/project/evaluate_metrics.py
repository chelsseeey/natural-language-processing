import json
import os
import string
import re
import numpy as np
import collections

class RAGEvaluator:
    def __init__(self):
        pass

    def normalize_answer(self, s):
        """정규화: 소문자, 공백/구두점/관사 제거"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_strict_em(self, prediction, ground_truths):
        """Strict EM: 100% 일치해야 정답"""
        norm_pred = self.normalize_answer(prediction)
        for gold in ground_truths:
            if norm_pred == self.normalize_answer(gold):
                return 1.0
        return 0.0

    def compute_soft_em(self, prediction, ground_truths):
        """Soft EM: 정답 단어 포함 여부"""
        norm_pred = self.normalize_answer(prediction)
        if not norm_pred: return 0.0
        
        for gold in ground_truths:
            norm_gold = self.normalize_answer(gold)
            if norm_gold in norm_pred: 
                return 1.0
        return 0.0

    def compute_f1(self, prediction, ground_truths):
        """F1 Score: 토큰 중복도"""
        norm_pred = self.normalize_answer(prediction)
        pred_tokens = norm_pred.split()
        
        best_f1 = 0.0
        for gold in ground_truths:
            norm_gold = self.normalize_answer(gold)
            gold_tokens = norm_gold.split()
            
            common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
            num_same = sum(common.values())
            
            if len(pred_tokens) == 0 or len(gold_tokens) == 0:
                f1 = int(pred_tokens == gold_tokens)
            elif num_same == 0:
                f1 = 0.0
            else:
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gold_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)
        return best_f1

    def compute_recall(self, retrieved_texts, ground_truths):
        """Recall@k"""
        for doc in retrieved_texts:
            norm_doc = self.normalize_answer(doc)
            for gold in ground_truths:
                norm_gold = self.normalize_answer(gold)
                if norm_gold in norm_doc: 
                    return 1.0
        return 0.0

    def calculate_metrics(self, file_path):
        dataset_name = os.path.basename(file_path).replace("_final_results.json", "")
        print(f"\n평가 분석 시작: {dataset_name}")
        
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없음: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics = {
            "no_rag_strict_em": [], 
            "no_rag_soft_em": [],
            "no_rag_f1": [],
            
            "rag_strict_em": [],
            "rag_soft_em": [],
            "rag_f1": [],
            
            "recall": [],
            "attribution": [],
            "latency": []
        }

        for item in data:
            gold_answers = item.get('gold_answers', [])
            
            # 1. No-RAG 평가 (배경지식)
            no_rag_pred = item.get('no_rag_answer', "")
            metrics["no_rag_strict_em"].append(self.compute_strict_em(no_rag_pred, gold_answers)) 
            metrics["no_rag_soft_em"].append(self.compute_soft_em(no_rag_pred, gold_answers))
            metrics["no_rag_f1"].append(self.compute_f1(no_rag_pred, gold_answers))

            # 2. RAG 평가 (검색 후 생성)
            rag_pred = item.get('rag_answer', "")
            metrics["rag_strict_em"].append(self.compute_strict_em(rag_pred, gold_answers))
            metrics["rag_soft_em"].append(self.compute_soft_em(rag_pred, gold_answers))
            metrics["rag_f1"].append(self.compute_f1(rag_pred, gold_answers))

            # 3. Recall & Attribution
            if 'recall_success' in item:
                recall_score = 1.0 if item['recall_success'] else 0.0
            else:
                retrieved_texts = item.get('retrieved_texts', [])
                recall_score = self.compute_recall(retrieved_texts, gold_answers)
            metrics["recall"].append(recall_score)

            # Attribution (Soft EM 성공 & Recall 성공)
            if metrics["rag_soft_em"][-1] == 1.0 and recall_score == 1.0:
                metrics["attribution"].append(1.0)
            else:
                metrics["attribution"].append(0.0)

            if 'latency' in item:
                metrics["latency"].append(item['latency'])

        # --- 통계 산출 ---
        # No-RAG
        avg_no_rag_strict = np.mean(metrics["no_rag_strict_em"]) * 100 
        avg_no_rag_soft = np.mean(metrics["no_rag_soft_em"]) * 100
        avg_no_rag_f1 = np.mean(metrics["no_rag_f1"]) * 100
        
        # RAG
        avg_rag_strict = np.mean(metrics["rag_strict_em"]) * 100
        avg_rag_soft = np.mean(metrics["rag_soft_em"]) * 100
        avg_rag_f1 = np.mean(metrics["rag_f1"]) * 100
        
        # Others
        avg_recall = np.mean(metrics["recall"]) * 100
        avg_attribution = np.mean(metrics["attribution"]) * 100
        
        # RAG Gain (Soft EM 기준)
        rag_gain_em = avg_rag_soft - avg_no_rag_soft
        
        # p95 Latency
        p95_latency = 0.0
        if metrics["latency"]:
            p95_latency = np.percentile(metrics["latency"], 95)
        
        # 최종 결과 딕셔너리
        result_summary = {
            "dataset": dataset_name,
            "sample_size": len(metrics["rag_soft_em"]),
            "metrics": {
                # [No-RAG]
                "No_RAG_Strict_EM": round(avg_no_rag_strict, 2), 
                "No_RAG_Soft_EM": round(avg_no_rag_soft, 2),
                "No_RAG_F1": round(avg_no_rag_f1, 2),
                
                # [RAG]
                "RAG_Strict_EM": round(avg_rag_strict, 2),
                "RAG_Soft_EM": round(avg_rag_soft, 2),
                "RAG_F1": round(avg_rag_f1, 2),
                
                # [Analysis]
                "RAG_Gain_Soft": round(rag_gain_em, 2),
                "Recall": round(avg_recall, 2),
                "Attribution": round(avg_attribution, 2),
                "p95_Latency_ms": round(p95_latency, 2)
            }
        }
        
        print("-" * 55)
        print(f"  No-RAG (Strict) | {avg_no_rag_strict:.2f}%")
        print(f"  No-RAG (Soft)   | {avg_no_rag_soft:.2f}%")
        print("-" * 55)
        print(f"  RAG (Strict)    | {avg_rag_strict:.2f}%")
        print(f"  RAG (Soft)      | {avg_rag_soft:.2f}%")
        print(f"  👉 RAG Gain     | {rag_gain_em:+.2f}%p (Soft 기준)")
        print("-" * 55)
        
        return result_summary

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    base_dir = "reports"
    
    # 평가할 파일 목록
    target_files = [
        "KLUE-MRC_final_results.json",
        "KorQuAD_final_results.json",
        "SQuAD_final_results.json",
        "CoS-E_final_results.json"
    ]
    
    all_results = []
    
    for filename in target_files:
        path = os.path.join(base_dir, filename)
        res = evaluator.calculate_metrics(path)
        if res: all_results.append(res)
            
    output_path = os.path.join(base_dir, "evaluation_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n(Strict/Soft/F1) 저장 완료: {output_path}")