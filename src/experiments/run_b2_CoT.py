# src/experiments/run_b2_CoT.py
import os
import sys

# --- Local Imports: 경로 설정 ---
# src/experiments에서 src/utils와 src/agents에 접근하기 위해 경로를 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Local Imports (공통 모듈 사용) ---
from src.utils.data_loader import load_bar_exam_qa_sample
from src.utils.api_client import GroqClient 
from src.agents.single_agent_CoT import create_cot_prompt 


# --- 1. Configuration ---
# 실험에 필요한 설정 값 정의
MODEL_NAME = "llama-3.3-70b-versatile" 
MAX_TOKENS = 500
SAMPLE_SIZE = 300 

# GroqClient 인스턴스 생성 (B2 CoT는 일관된 추론을 위해 temperature=0.0)
cot_client = GroqClient(model=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=0.0)


# --- 2. CoT Inference Function ---

def run_cot_baseline(question_data):
    """
    Runs the B2 Chain-of-Thought baseline for the first sample question 
    using the configured GroqClient.
    """
    
    test_question = question_data[0]
    
    system_instruction, prompt_text = create_cot_prompt(test_question)
    
    print("\n" + "#"*70)
    print(f"B2 (Chain-of-Thought) Baseline Test Started: {test_question['idx']}")
    print("#"*70)

    try:
        # GroqClient는 prompt 문자열을 받습니다. system instruction을 최종 prompt에 통합합니다.
        full_prompt = f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt_text}"

        response = cot_client.generate(prompt=full_prompt)
        
        print("\n[Groq Llama 3.3 70B CoT Response]")
        print(response)
        print("\n" + "="*70)
        
        # Display expected answer
        expected_answer = test_question['answer']
        print(f"[Expected Answer]: {expected_answer}")
        
    except Exception as e:
        print(f"❌ API Call Error: {e}")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Load data
    questions = load_bar_exam_qa_sample(SAMPLE_SIZE)
    
    # 2. Proceed to the next step if data loaded successfully
    if questions:
        print("\n" + "="*70)
        print(f"Dataset loaded successfully. {len(questions)} questions ready.")
        
        first_q = questions[0]
        print(f"[First Q ID]: {first_q['idx']}")
        print(f"[Question Start]: {first_q['question'][:80]}...")
        print(f"[Answer]: {first_q['answer']} (A: {first_q['choice_a'][:30]}...)")
        print("="*70)
        
        # 3. Run the B2 CoT baseline
        run_cot_baseline(questions) 
    else:
        print("\n🚨 Data loading failed. Cannot proceed to the next step.")