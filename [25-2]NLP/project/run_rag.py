import json
import os
import time
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from openai import OpenAI
from preprocessor import CustomPreprocessor  

# ==========================================
# [설정] API 키 입력
# ==========================================
OPENAI_API_KEY = "your-api-key-here"  # ★ 여기에 키 입력
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = 'gpt-4o-mini'  

class SimpleRAGRunner:
    def __init__(self, corpus_path):
        self.preprocessor = CustomPreprocessor()
        self.client = client
        self.model_name = MODEL_NAME
        
        # 1. 통합 코퍼스 로드 및 인덱싱
        print(f"코퍼스 로딩 및 인덱싱: ({os.path.basename(corpus_path)})")
        self.corpus_docs = []
        
        # JSONL 파일 읽기
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.corpus_docs.append(json.loads(line))
        
        # BM25용 텍스트 리스트 추출
        corpus_texts = [doc['text'] for doc in self.corpus_docs]
        
        # 전처리 (영어/한국어 섞여있으므로 'General' 모드로 토큰화)
        tokenized_corpus = self.preprocessor.process_dataset("General", corpus_texts)
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"(총 문서: {len(self.corpus_docs)}개)")

    def clean_text(self, text):
        """문자열 비교를 위한 공백/줄바꿈 정규화"""
        return " ".join(text.split()).strip()

    def retrieve(self, query, top_k=5):
        """검색 수행"""
        tokenized_query = self.preprocessor.process_dataset("Query", [query])[0]
        scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n_idx:
            doc = self.corpus_docs[idx]
            results.append({
                "doc_id": doc['doc_id'],
                "text": doc['text'],
                "score": scores[idx]
            })
        return results

    def generate_with_openai(self, prompt):
        """OpenAI 응답 생성"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return "Error"

    def run_experiment(self, input_path, output_path, limit=None):
        """실험 실행 (Recall 자동 계산 포함)"""
        print(f"\n실험 시작: {os.path.basename(input_path)}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        if limit: questions = questions[:limit] # 테스트용

        results = []
        total_recall = 0

        for item in tqdm(questions):
            query = item['question']
            ground_truth_context = self.clean_text(item.get('context', '')) # 정답 본문
            
            # --- 1. Retrieval (검색) ---
            retrieved = self.retrieve(query, top_k=5)
            
            # --- Recall 평가 (텍스트 매칭) ---
            # 검색된 문서들 중에 정답 본문과 내용이 같은 게 있는가?
            is_retrieved = False
            retrieved_texts_clean = [self.clean_text(r['text']) for r in retrieved]
            
            if ground_truth_context in retrieved_texts_clean:
                is_retrieved = True
                total_recall += 1
            
            # --- 2. Generation (답변 생성) ---
            
            # (A) No-RAG (배경지식만 사용)
            prompt_no_rag = f"Question: {query}\nAnswer:"
            ans_no_rag = self.generate_with_openai(prompt_no_rag)
            
            # (B) RAG (검색된 문서 사용)
            # 상위 3개 문서만 프롬프트에 넣음
            context_str = "\n\n".join([f"Doc {i+1}: {r['text']}" for i, r in enumerate(retrieved[:3])])
            prompt_rag = f"""
            Based ONLY on the documents below, answer the question.
            
            [Documents]
            {context_str}
            
            [Question]
            {query}
            
            Answer:
            """
            
            # 시간 측정 시작 (RAG 답변 생성)
            start_time = time.time()
            ans_rag = self.generate_with_openai(prompt_rag)
            end_time = time.time()
            
            # Latency 계산 (밀리초 단위)
            latency_ms = (end_time - start_time) * 1000

            # 결과 저장
            results.append({
                "id": item['id'],
                "question": query,
                "recall_success": is_retrieved, # Recall 성공 여부 기록
                "no_rag_answer": ans_no_rag,
                "rag_answer": ans_rag,
                "gold_answers": item['answers'],
                "retrieved_top1": retrieved[0]['text'][:50] + "...", # 확인용
                "latency": latency_ms  # RAG 답변 생성 시간 (ms)
            })
            time.sleep(0.5) # API 속도 조절

        # 최종 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"저장됨: {output_path}")
        print(f"Recall@5 Score: {total_recall}/{len(questions)} ({total_recall/len(questions)*100:.2f}%)")

# --- 실행부 ---
if __name__ == "__main__":
    BASE_DIR = "dataset"
    OUT_DIR = "reports"
    
    # 2. 실험할 파일 목록 
    targets = [
        # ("KLUE-MRC", "KLUE-MRC/klue_mrc_300.json", "KLUE-MRC_corpus.jsonl"),
        # ("KorQuAD", "KorQuAD2.1/korquad_300.json", "KorQuAD_corpus.jsonl"),
        ("SQuAD", "SQuAD/squad_300.json", "SQuAD_corpus.jsonl"),
        # ("CoS-E", "CoS-E/cose_300.json", "CoS-E_corpus.jsonl")
    ]
    
    for name, rel_path, corpus_file in targets:
        # 1. 각 데이터셋별 코퍼스 사용
        CORPUS_FILE = os.path.join(BASE_DIR, corpus_file)
        runner = SimpleRAGRunner(CORPUS_FILE)
        
        in_path = os.path.join(BASE_DIR, rel_path)
        out_path = os.path.join(OUT_DIR, f"{name}_final_results.json")
        
        # 전체 300개 실행
        runner.run_experiment(in_path, out_path, limit=None)