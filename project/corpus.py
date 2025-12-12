import json
import os
import hashlib

class CorpusFormatter:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir

    def generate_doc_id(self, dataset_name, index):
        """
        문서 ID 생성 규칙:
        예: KLUE-MRC_0001, KorQuAD_0052
        """
        return f"{dataset_name}_{index:05d}"

    def add_dataset(self, file_path, dataset_name, output_file):
        """특정 데이터셋 파일을 읽어서 개별 코퍼스 생성"""
        print(f"Processing {dataset_name} corpus...")
        
        if not os.path.exists(file_path):
            print(f"   파일을 찾을 수 없음: {file_path}")
            return 0

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_docs = []
        doc_count = 0
        
        for item in data:
            context = item.get('context', '').strip()
            
            # 1. 텍스트가 비어있으면 스킵
            if not context:
                continue

            doc_count += 1
            # 2. 사용자 제안 포맷으로 생성
            doc_entry = {
                "doc_id": self.generate_doc_id(dataset_name, doc_count),
                "dataset": dataset_name,
                "text": context
            }
            new_docs.append(doc_entry)

        # 3. 개별 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in new_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                
        print(f"   {len(new_docs)}개 문서 저장 완료: {output_file}")
        return len(new_docs)

# --- 실행 ---
if __name__ == "__main__":
    BASE_DIR = "dataset"
    
    formatter = CorpusFormatter(output_dir=BASE_DIR)
    
    total_docs = 0
    
    # 1. KLUE-MRC
    output_file = os.path.join(BASE_DIR, "KLUE-MRC_corpus.jsonl")
    total_docs += formatter.add_dataset(
        os.path.join(BASE_DIR, "KLUE-MRC/klue_mrc_300.json"), 
        "KLUE-MRC",
        output_file
    )
    
    # 2. KorQuAD 2.1
    output_file = os.path.join(BASE_DIR, "KorQuAD_corpus.jsonl")
    total_docs += formatter.add_dataset(
        os.path.join(BASE_DIR, "KorQuAD2.1/korquad_300.json"), 
        "KorQuAD",
        output_file
    )
    
    # 3. SQuAD
    output_file = os.path.join(BASE_DIR, "SQuAD_corpus.jsonl")
    total_docs += formatter.add_dataset(
        os.path.join(BASE_DIR, "SQuAD/squad_300.json"), 
        "SQuAD",
        output_file
    )
    
    # 4. CoS-E
    output_file = os.path.join(BASE_DIR, "CoS-E_corpus.jsonl")
    total_docs += formatter.add_dataset(
        os.path.join(BASE_DIR, "CoS-E/cose_300.json"), 
        "CoS-E",
        output_file
    )

    print(f"\n코퍼스 구축 완료: {BASE_DIR}")
    print(f"   총 문서 개수: {total_docs}개")