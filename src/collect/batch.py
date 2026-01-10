import argparse
from src.utils.vllm_backend import get_response
from tqdm import tqdm
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

ORIGINAL_INST = """Remember to put your answer on its own line after "Answer:"."""
COT_INST = """Let's think step by step and output your final answer within \\boxed{{}}."""

def process_one(record, model_name, port, reasoning_effort):
    # 构造新 prompt
    content = record["prompt"][0]["content"]
    new_content = content.replace(ORIGINAL_INST, COT_INST)

    response = get_response(
        new_content,
        model_name=model_name,
        port=port,
        logout=False,
        reasoning_effort=reasoning_effort
    )
    
    # 返回原始 record + 新字段
    updated_record = {**record}
    updated_record["gpt-oss-120b-response"] = response
    updated_record["model_name"] = model_name
    updated_record["reasoning_effort"] = reasoning_effort
    return updated_record

def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM backend.")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort level (default: low)")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of threads for concurrent processing (default: 32)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path for vLLM")
    parser.add_argument("--port", type=int, default=1145, help="vLLM server port (default: 1145)")

    args = parser.parse_args()

    raw_path = args.raw_path
    output_path = args.output_path
    reasoning_effort = args.reasoning_effort
    max_workers = args.max_workers
    model_name = args.model_name
    port = args.port

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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_one, record, model_name, port, reasoning_effort): record
            for record in pending_questions
        }

        with tqdm(total=len(pending_questions), desc="Processing") as pbar:
            with open(output_path, "a", encoding="utf-8") as fout:
                for future in as_completed(future_to_record):
                    try:
                        result = future.result()
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                    except Exception as e:
                        record = future_to_record[future]
                        idx = record.get("extra_info", {}).get("index", "unknown")
                        print(f"\nError processing index={idx}: {e}")
                    finally:
                        pbar.update(1)

    print("All done.")

if __name__ == "__main__":
    main()