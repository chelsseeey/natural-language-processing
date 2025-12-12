import json
import os
import random
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from run_rag import SimpleRAGRunner 
from evaluate_metrics import RAGEvaluator 

# ==========================================
# [설정] API 키 확인
# ==========================================
OPENAI_API_KEY = "your-api-key-here"
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = 'gpt-4o-mini'

class RobustnessTester:
    def __init__(self, corpus_path):
        self.runner = SimpleRAGRunner(corpus_path)
        self.evaluator = RAGEvaluator()
        self.client = client
        self.model_name = MODEL_NAME

    def generate_paraphrase(self, question):
        """질문 패러프레이징 (표현 바꾸기)"""
        prompt = f"""
        Paraphrase the following question. Keep the meaning exactly the same, 
        but change the wording or sentence structure.
        Output ONLY the paraphrased question.
        
        Original: {question}
        Paraphrased:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except:
            return question 

    def run_test(self, input_path, output_path, dataset_name, sample_size=30):
        print(f"\n강건성(Robustness) 테스트 시작: {dataset_name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 30개 랜덤 샘플링
        if len(data) > sample_size:
            target_data = random.sample(data, sample_size)
        else:
            target_data = data

        results = []
        
        for item in tqdm(target_data):
            original_q = item['question']
            gold_answers = item['answers'] # 정답지
            
            # 1. 패러프레이즈 생성
            para_q = self.generate_paraphrase(original_q)
            
            # 2. 원본 질문 RAG 수행
            retrieved_orig = self.runner.retrieve(original_q, top_k=3)
            
            context_orig = "\n".join([d['text'] for d in retrieved_orig])
            prompt_orig = f"Documents:\n{context_orig}\nQuestion: {original_q}\nAnswer:"
            ans_orig = self.runner.generate_with_openai(prompt_orig)
            
            # 3. 변형 질문 RAG 수행
            retrieved_para = self.runner.retrieve(para_q, top_k=3)
            context_para = "\n".join([d['text'] for d in retrieved_para])
            prompt_para = f"Documents:\n{context_para}\nQuestion: {para_q}\nAnswer:"
            ans_para = self.runner.generate_with_openai(prompt_para)

            # 4. 점수 계산
            em_orig = self.evaluator.compute_soft_em(ans_orig, gold_answers)
            em_para = self.evaluator.compute_soft_em(ans_para, gold_answers)
            
            results.append({
                "original_q": original_q,
                "paraphrased_q": para_q,
                "original_answer": ans_orig,       
                "paraphrased_answer": ans_para,     
                "gold_answers": gold_answers,      
                "em_orig": em_orig,
                "em_para": em_para,
                "delta_em": em_para - em_orig
            })
            time.sleep(1) 


        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과: {output_path}")

        # 통계 계산
        avg_orig = np.mean([r['em_orig'] for r in results]) * 100
        avg_para = np.mean([r['em_para'] for r in results]) * 100
        avg_delta = avg_para - avg_orig

        print("-" * 50)
        print(f"  Sample Size    | {len(results)}")
        print(f"  Original EM    | {avg_orig:.2f}%")
        print(f"  Paraphrased EM | {avg_para:.2f}%")
        print(f"  Δ EM        | {avg_delta:+.2f}%p")
        print("-" * 50)

     
        summary = {
            "dataset": dataset_name,
            "sample_size": len(results),
            "metrics": {
                "Original_EM": round(avg_orig, 2),
                "Paraphrased_EM": round(avg_para, 2),
                "Delta_EM": round(avg_delta, 2)
            }
        }
        return summary

if __name__ == "__main__":
    BASE_DIR = "dataset"
    REPORTS_DIR = "reports"
    
    # reports 디렉토리 생성 (없으면)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 평가 대상 (데이터셋명, 데이터 경로, 코퍼스 파일)
    targets = [
        # ("KLUE-MRC", "KLUE-MRC/klue_mrc_300.json", "KLUE-MRC_corpus.jsonl"),
        # ("KorQuAD", "KorQuAD2.1/korquad_300.json", "KorQuAD_corpus.jsonl"),
        ("SQuAD", "SQuAD/squad_300.json", "SQuAD_corpus.jsonl"),
        # ("CoS-E", "CoS-E/cose_300.json", "CoS-E_corpus.jsonl")
    ]
    
    all_summaries = []

    # 기존 결과 로드 (있으면)
    summary_path = os.path.join(REPORTS_DIR, "robustness_summary.json")
    existing_summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            existing_summaries = json.load(f)
    
    # 기존 결과를 딕셔너리로 변환 (dataset 이름으로 인덱싱)
    summary_dict = {item['dataset']: item for item in existing_summaries}
    
    for name, rel_path, corpus_file in targets:
        # 각 데이터셋별 코퍼스 사용
        CORPUS_FILE = os.path.join(BASE_DIR, corpus_file)
        tester = RobustnessTester(CORPUS_FILE)
        
        in_path = os.path.join(BASE_DIR, rel_path)
        out_path = os.path.join(REPORTS_DIR, f"{name}_robustness.json")
        
        # 테스트 실행
        res = tester.run_test(in_path, out_path, dataset_name=name, sample_size=300)
        
        # 결과 업데이트 (같은 dataset이면 덮어쓰기, 없으면 추가)
        summary_dict[name] = res
    
    # 딕셔너리를 리스트로 변환하여 저장
    all_summaries = list(summary_dict.values())
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        
    print(f"평가 완료: {summary_path}")