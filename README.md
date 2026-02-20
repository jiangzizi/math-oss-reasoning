# Math OSS Reasoning

A data collection and processing pipeline for training large language models with mathematical and scientific reasoning capabilities.

## Project Overview

This project provides tools to:
- Collect reasoning data from math datasets (DAPO-Math)
- Collect science multiple-choice question data
- Process data with vLLM-based language models
- Apply chain-of-thought (CoT) prompting techniques
- Deduplicate and reorganize datasets for training

## Directory Structure

```
math-oss-reasoning/
├── src/
│   ├── collect/              # Data collection scripts
│   │   ├── batch-dapo.py     # Batch processing for DAPO-Math dataset
│   │   ├── batch-science.py  # Batch processing for science questions
│   │   ├── simple.py         # Simple single-sample processing
│   │   └── loose.py          # Loose reward score calculation
│   ├── data/                 # Data processing utilities
│   │   ├── dedup.py          # Dataset deduplication
│   │   ├── nemotron_data.py  # Nemotron data format conversion
│   │   └── reorg.py          # Dataset reorganization
│   └── utils/
│       └── vllm_backend.py   # vLLM inference wrapper
└── README.md
```

## Data Structures

### Math Data Format (DAPO-Math)

Input JSONL records with the following structure:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Math problem text with instructions..."
    }
  ],
  "reward_model": {
    "ground_truth": "42"
  },
  "extra_info": {
    "index": 0
  }
}
```

Output records include:

```json
{
  "prompt": [...],
  "reasoning_prompt": "Modified prompt with CoT instructions",
  "gpt-oss-120b-response": "Model's chain-of-thought response",
  "model_name": "model-name",
  "reasoning_effort": "high",
  "standard_answer": "42",
  "model_answer": "42",
  "reward": 1.0,
  "extra_info": {"index": 0}
}
```

### Science Data Format

Input JSONL records:

```json
{
  "input": [
    {"role": "user", "content": "Science question..."}
  ],
  "output": "Answer explanation with (E)...",
  "category": "science",
  "license": "cc-by-4.0",
  "reasoning": "on",
  "generator": "",
  "used_in_training": "",
  "version": "v1",
  "system_prompt": ""
}
```

Output records:

```json
{
  "messages": [
    {"role": "user", "content": "Question..."},
    {"role": "assistant", "content": "Model response..."}
  ],
  "category": "science",
  "reasoning": "on",
  "model_name": "model-name",
  "reasoning_effort": "high",
  "original_label": "E",
  "new_label": "E",
  "label": "E",
  "reward": 1.0,
  "extra_info": {"index": 0}
}
```

## Key Components

### vLLM Backend ([`src/utils/vllm_backend.py`](src/utils/vllm_backend.py))

Wrapper for vLLM server inference with reasoning effort control:

```python
from src.utils.vllm_backend import get_response

response = get_response(
    "Your prompt here",
    model_name="model-path/",
    port=1145,
    reasoning_effort="high"  # low, medium, or high
)
```

### Batch Processing

#### DAPO-Math Dataset ([`src/collect/batch-dapo.py`](src/collect/batch-dapo.py))

```bash
python -m src.collect.batch_dapo \
    --raw_path datasets/dapo-math.jsonl \
    --output_path output/results.jsonl \
    --model_name gpt-oss-120b/ \
    --port 1145 \
    --reasoning_effort high \
    --max_workers 32
```

#### Science Questions ([`src/collect/batch-science.py`](src/collect/batch-science.py))

```bash
python -m src.collect.batch_science \
    --raw_path datasets/science.jsonl \
    --output_path output/science_results.jsonl \
    --model_name gpt-oss-120b/ \
    --port 1145 \
    --reasoning_effort high \
    --max_workers 32
```

## Reasoning Effort Levels

| Level | Description |
|-------|-------------|
| `low` | Minimal reasoning, faster responses |
| `medium` | Balanced reasoning and speed |
| `high` | Extensive chain-of-thought, best quality |

## Chain-of-Thought Prompts

### Math (DAPO-Math)
- Original: `"Remember to put your answer on its own line after 'Answer:'."`
- CoT: `"Let's think step by step and output your final answer within \\boxed{}."`

### Science
- Uses multiple-choice extraction with patterns like:
  - `"Answer: A/B/C/D/E..."`
  - `"The answer is (X)"`
  - `\\boxed{X}`

## Reward Calculation

- **Math**: Compare extracted `\\boxed{}` content with ground truth
- **Science**: Compare extracted answer letter with model's response
- Reward = 1.0 for correct, 0.0 for incorrect

## Data Processing

### Deduplication ([`src/data/dedup.py`](src/data/dedup.py))

Remove duplicate entries based on prompt content.

### Reorganization ([`src/data/reorg.py`](src/data/reorg.py))

Convert between different dataset formats.

## License

See [LICENSE](LICENSE) file for details.
