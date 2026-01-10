from src.utils.vllm_backend import get_response
from tqdm import tqdm
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

ORIGINAL_INST = """Remember to put your answer on its own line after "Answer:"."""
COT_INST = """Let's think step by step and output your final answer within \\boxed{{}}."""

def process_one(record, model_name, port):
    # 构造新 prompt
    content = record["prompt"][0]["content"]
    new_content = content.replace(ORIGINAL_INST, COT_INST)
    # 注意：这里假设 prompt 是 list of dict，只改第一个
    new_prompt = [{"role": record["prompt"][0]["role"], "content": new_content}]
    
    response = get_response(
        new_prompt,
        model_name=model_name,
        port=port,
        logout=True,
        reasoning_effort="high"
    )
    
    # 返回原始 record + 新字段
    updated_record = {**record}
    updated_record["gpt-oss-120b-response"] = response
    return updated_record

def main():
    raw_path = "datasets/DAPO-Math-17k/data/dapo-math-100.jsonl"
    output_path = "outputs/dapo-math-100_gpt-oss-120b.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load all questions
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    questions = [json.loads(line) for line in lines]

    # Step 2: Build set of already processed indices
    completed_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        idx = rec.get("extra_info", {}).get("index")
                        if idx is not None:
                            completed_indices.add(idx)
                    except json.JSONDecodeError:
                        continue  # skip corrupted lines

    # Step 3: Filter out completed
    pending_questions = [
        q for q in questions
        if q.get("extra_info", {}).get("index") not in completed_indices
    ]

    print(f"Total: {len(questions)}, Already done: {len(completed_indices)}, Pending: {len(pending_questions)}")

    # Step 4: Concurrent processing
    model_name = "gpt-oss-120b/"
    port = 1145
    max_workers = 32

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pending tasks
        future_to_record = {
            executor.submit(process_one, record, model_name, port): record
            for record in pending_questions
        }

        # Use tqdm to track progress
        with tqdm(total=len(pending_questions), desc="Processing") as pbar:
            with open(output_path, "a", encoding="utf-8") as fout:
                for future in as_completed(future_to_record):
                    try:
                        result = future.result()
                        # Write immediately (append mode)
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()  # Ensure write is persisted
                    except Exception as e:
                        # Log error but don't crash
                        record = future_to_record[future]
                        idx = record.get("extra_info", {}).get("index", "unknown")
                        print(f"\nError processing index={idx}: {e}")
                    finally:
                        pbar.update(1)

    print("All done.")

if __name__ == "__main__":
    main()