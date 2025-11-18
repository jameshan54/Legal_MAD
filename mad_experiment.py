import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# --- 1. Configuration ---
# Load environment variables from the .env file.
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Check your .env file.")

# Model specified in project requirements
MODEL_NAME = "llama-3.3-70b-versatile" 
MAX_TOKENS = 500

client = Groq(api_key=API_KEY)

# --- 2. Data Loading Function ---
# Local path to the downloaded Bar Exam QA dataset
LOCAL_DATA_DIR = "data/raw/barexam_qa"
DATA_FILE_NAME = "data/qa/train.csv"

def load_bar_exam_qa_sample(sample_size: int = 300):
    """
    Loads the Bar_Exam_QA CSV file directly using Pandas and returns a sample.
    """
    # Construct the full path to the local data file.
    file_path = os.path.join(os.getcwd(), LOCAL_DATA_DIR, DATA_FILE_NAME)
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Local data file does not exist: {file_path}")
        return []
        
    try:
        # Load the CSV file into a Pandas DataFrame.
        df = pd.read_csv(file_path)
        # Convert DataFrame records to a list of dictionaries.
        data = df.to_dict(orient='records')
        # Sample the data.
        sample_data = data[:sample_size]
        
        print(f"✅ Bar_Exam_QA data successfully loaded: {len(sample_data)} out of {len(df)} questions loaded.")
        return sample_data
        
    except Exception as e:
        print(f"❌ ERROR: An error occurred while loading the local file: {e}")
        return []

# --- 3. CoT/MAD Inference Function (To be implemented next) ---
def run_cot_baseline(question_data):
    """Runs the B2 Chain-of-Thought baseline."""
    # The Groq API call logic will be implemented here.
    pass 

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    SAMPLE_SIZE = 300
    
    # 1. Load data
    questions = load_bar_exam_qa_sample(SAMPLE_SIZE)
    
    # 2. Proceed to the next step if data loaded successfully
    if questions:
        print("\n" + "="*70)
        print(f"Dataset loaded successfully. {len(questions)} questions ready.")
        
        # Display the first question details for verification
        first_q = questions[0]
        print(f"[First Q ID]: {first_q['idx']}")
        print(f"[Question Start]: {first_q['question'][:80]}...")
        print(f"[Answer]: {first_q['answer']} (A: {first_q['choice_a'][:30]}...)")
        print("="*70)
        
        # 3. Run the B2 CoT baseline (Currently calls 'pass')
        run_cot_baseline(questions)
    else:
        print("\n🚨 Data loading failed. Cannot proceed to the next step.")