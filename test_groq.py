import os
from groq import Groq
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 가져옵니다.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    # 이 오류가 발생하면 .env 파일의 GROQ_API_KEY 설정이 잘못된 것입니다.
    raise ValueError("GROQ_API_KEY not found or empty in environment variables.")

# Groq 클라이언트 초기화
# Groq()에 api_key를 명시적으로 전달하거나, 환경 변수 자동 로드를 이용합니다.
client = Groq(api_key=api_key) 

# 프로젝트 사양에 명시된 모델을 사용합니다.
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOKENS = 500 # Debater (Opening)의 최대 토큰 버짓을 고려하여 설정

def generate_baseline_response(prompt: str) -> str:
    """
    B1: Single-Agent (Zero-shot) 베이스라인 응답을 생성하는 함수
    """
    print(f"--- 모델 호출: {MODEL_NAME} ---")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME, 
            max_tokens=MAX_TOKENS, 
            # 초기 테스트이므로 temperature는 기본값(0.0)에 가깝게 설정합니다.
            temperature=0.01 
        )
        
        # 생성된 응답 텍스트를 반환합니다.
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# 법률 추론 관련 테스트 실행
test_prompt = "형법상 미수범의 처벌 근거와 그 성립 요건을 간략하게 설명해 주세요."
response = generate_baseline_response(test_prompt)

print("\n" + "="*50)
print(f"[테스트 프롬프트]\n{test_prompt}")
print("\n" + "-"*50)
print(f"[Llama 3.1 70B 응답]\n{response}")
print("="*50 + "\n")