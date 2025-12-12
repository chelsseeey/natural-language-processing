import re
import string
from kiwipiepy import Kiwi

class CustomPreprocessor:
    def __init__(self):
        # Kiwi 형태소 분석기 초기화 (한국어용)
        # model_type='sbg'는 속도가 빠르고 성능이 준수한 모델입니다.
        self.kiwi = Kiwi(model_type='sbg')
        
        # 검색에 유의미한 품사 태그 (명사, 동사, 형용사, 외국어, 숫자)
        # 조사는 제외하여 검색 성능을 높임
        self.target_tags = {'NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'SL', 'SN'} 

    def is_korean(self, text):
        """한글 포함 여부 확인 (한글이 있으면 한국어 처리 사용)"""
        return bool(re.search("[가-힣]", text))

    def _clean_text(self, text):
        """공통: 불필요한 공백 제거"""
        if not text:
            return ""
        return " ".join(text.split())

    def _remove_html(self, text):
        """HTML 태그 제거 (<br>, <td> 등)"""
        clean = re.sub(r'<.*?>', ' ', text)
        clean = clean.replace('&nbsp;', ' ')
        return clean

    def preprocess_english(self, text):
        """[영어] 소문자 변환 + 구두점 제거 + 토큰화"""
        text = self._clean_text(text).lower()
        # 구두점 제거 (apple. -> apple)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def preprocess_korean(self, text):
        """[한국어] HTML 제거 + 형태소 분석"""
        text = self._remove_html(text) # 혹시 모를 태그 제거
        text = self._clean_text(text)
        
        tokens = []
        try:
            # Kiwi로 형태소 분석
            result = self.kiwi.analyze(text)
            for token, pos, _, _ in result[0][0]:
                if pos in self.target_tags:
                    tokens.append(token)
        except Exception:
            # 실패 시 띄어쓰기로 대체
            tokens = text.split()
            
        return tokens

    def process_dataset(self, mode, text_list):
        """
        입력된 텍스트 리스트를 전처리하여 반환
        mode: "Query", "General", 또는 데이터셋 이름
        """
        processed_docs = []
        
        # 진행률 표시 없이 빠르게 처리 (tqdm은 호출하는 쪽에서 씀)
        for text in text_list:
            if not text:
                processed_docs.append([])
                continue
                
            # 1. 언어 자동 감지 전략
            # 텍스트에 한글이 포함되어 있으면 한국어 처리기, 아니면 영어 처리기 사용
            if self.is_korean(text):
                tokens = self.preprocess_korean(text)
            else:
                tokens = self.preprocess_english(text)
            
            processed_docs.append(tokens)
            
        return processed_docs

# --- 테스트 코드 (이 파일을 직접 실행하면 테스트됨) ---
if __name__ == "__main__":
    prep = CustomPreprocessor()
    
    samples = [
        "임진왜란 때 <b>곽준</b>이 전사한 곳은?", # 한국어 + HTML
        "Rivers flow through valleys.",        # 영어
        "Apple의 CEO는 누구인가?"              # 섞임
    ]
    
    print("전처리 테스트 결과:")
    results = prep.process_dataset("General", samples)
    for org, res in zip(samples, results):
        print(f"원본: {org}")
        print(f"결과: {res}\n")