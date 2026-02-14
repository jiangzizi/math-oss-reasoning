import argparse
import re
import string
from src.utils.vllm_backend import get_response
from tqdm import tqdm
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def _strip_chain_of_thought(text: str) -> str:
    """移除 chain of thought，只保留最终答案部分"""
    if not text:
        return ""
    
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    
    return text.strip()


def _normalize_text(text: str) -> str:
    """标准化文本"""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_letter_from_response(response: str, valid_letters: list = None) -> str | None:
    """
    从模型回复中提取选择的选项字母
    """
    if valid_letters is None:
        valid_letters = list(string.ascii_uppercase[:10])  # A-J
    
    if not response:
        return None

    text = _strip_chain_of_thought(response)
    
    # 优先匹配 "The answer is (X)" 或 "Answer: (X)" 等格式
    patterns = [
        r"(?:answer|option|choice|solution)\s*(?:is|:)?\s*\(?([A-J])\)?",
        r"\(?([A-J])\)?\s*(?:is\s*(?:the)?\s*correct|is\s*the\s*answer)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*\(?([A-J])\)?",
    ]

    valid_letters = {letter.upper() for letter in valid_letters}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    # Fallback: 找最后一个有效的单独大写字母
    candidates = re.findall(r"\b([A-J])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid_letters:
            return letter

    return None


def _extract_answer_from_output(output: str) -> str | None:
    """
    从原始 output 中提取答案字母
    """
    if not output:
        return None
    
    # 答案通常在最后，格式如 "The answer is (E)."
    text = _strip_chain_of_thought(output)
    
    # 匹配答案格式
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?",
        r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])",
        r"\boxed\{[^}]*([A-Z])[^}]*\}",
        r"answer is ([a-zA-Z])",
        r"^.*\(([A-J])\)\.*$",  # 最后出现的 (X)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()
    
    return None


def process_one(record, model_name, port, reasoning_effort, idx):
    # 构造新 prompt - 从 input 中获取 user 消息
    messages = record.get("input", [])
    
    # 找到 user 角色的消息
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break
    
    # 获取原始 output 中的答案
    original_output = record.get("output", "")
    original_label = _extract_answer_from_output(original_output)
    
    # 构建新的 prompt
    new_content = user_content
    
    # 获取模型回复
    response = get_response(
        new_content,
        model_name=model_name,
        port=port,
        logout=True,
        reasoning_effort=reasoning_effort
    )
    
    # 从模型回复中提取答案
    new_label = _extract_letter_from_response(response)
    
    # 构建 messages 格式
    output_messages = [
        {"role": "user", "content": new_content},
        {"role": "assistant", "content": response}
    ]
    
    # 返回结果
    result = {
        "messages": output_messages,
        "category": record.get("category", "science"),
        "license": record.get("license", "cc-by-4.0"),
        "reasoning": record.get("reasoning", "on"),
        "generator": record.get("generator", ""),
        "used_in_training": record.get("used_in_training", ""),
        "version": record.get("version", "v1"),
        "system_prompt": record.get("system_prompt", ""),
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "original_label": original_label,
        "new_label": new_label,
        "label": original_label,
        "reward": 1.0 if original_label == new_label and original_label is not None else 0.0, 
        "extra_info": {"index": idx}
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM backend.")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort level (default: low)")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of threads for concurrent processing (default: 32)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path for vLLM")
    parser.add_argument("--port", type=int, default=1145, help="vLLM server port (default: 1145)")
    parser.add_argument("--max_lines", type=int, default=-1, help="Maximum number of lines to process (default: all)")

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
    if args.max_lines > 0:
        lines = lines[:args.max_lines]
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

    # Step 3: Filter out completed (use enumerate to track original indices)
    pending_questions = [
        (idx, q) for idx, q in enumerate(questions)
        if idx not in completed_indices
    ]

    print(f"Total: {len(questions)}, Already done: {len(completed_indices)}, Pending: {len(pending_questions)}")

    # Step 4: Concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_one, record, model_name, port, reasoning_effort, idx): (idx, record)
            for idx, record in pending_questions
        }

        with tqdm(total=len(pending_questions), desc="Processing") as pbar:
            with open(output_path, "a", encoding="utf-8") as fout:
                for future in as_completed(future_to_record):
                    try:
                        result = future.result()
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                    except Exception as e:
                        idx, record = future_to_record[future]
                        print(f"\nError processing index={idx}: {e}")
                    finally:
                        pbar.update(1)

    print("All done.")

if __name__ == "__main__":
    main()
