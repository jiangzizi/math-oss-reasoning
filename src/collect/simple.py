
from src.utils.vllm_backend import get_response
from tqdm import tqdm
import json


ORIGINAL_INST = """Remember to put your answer on its own line after "Answer:"."""
COT_INST = """Let's think step by step and output your final answer within \\boxed{{}}."""

if __name__ == "__main__":
    raw_path = "datasets/DAPO-Math-17k/data/dapo-math-100.jsonl"
    
    with open(raw_path, "r") as f:
        lines = f.readlines()
    questions = [json.loads(line) for line in lines]
    
    
    for idx, question in tqdm(enumerate(questions)):
        math_question_cot = question["prompt"][0]["content"].replace(ORIGINAL_INST, COT_INST)
        response = get_response(math_question_cot, model_name="gpt-oss-120b/", port=1145, logout= True, reasoning_effort="high")
    
    