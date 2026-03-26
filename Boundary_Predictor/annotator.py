import os
import json
import argparse
import ast
from vllm import LLM, SamplingParams
from tqdm import tqdm
from copy import deepcopy
import hashlib
import random
import json
from typing import List, Tuple
import json
from typing import Optional, List, Dict

def get_hashes_and_lines(raw_line):
    hash = hashlib.md5(raw_line.encode('utf-8')).hexdigest()
    return hash

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                dataset.append(json.loads(line))
            except Exception as e:
                print('line:', i)
                print(e)
    print('dataset:', len(dataset))
    return dataset

def save_jsonl(data, filename):
    print('save data:', len(data))
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 创建必要的文件夹
    with open(filename, "a", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def append_jsonl(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 自动创建目录
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_processed_ids(path_save: str) -> set:
    if not os.path.exists(path_save):
        return set()
    with open(path_save, encoding="utf-8") as f:
        ids = {json.loads(line)["id_ddm"] for line in f}
    return ids


def construct_dataset(path):
    dataset = read_jsonl(path)
    chunks = []
    for sample in dataset:
        output = sample['dialogs'][1]['content']
        think = output.split('</think>')[0]
        think = think.replace('<think>', '').strip()

        
        thinks = think.split('\n\n')
        thinks = think.split(' ')
        max_len = 60
        if len(thinks) < max_len:
            continue
        
        for i in range(100):
            beg_idx = random.randint(0, len(thinks) - max_len)
            chunks.append({
                'id_ddm': sample['id_ddm'],
                'text': ' '.join(thinks[beg_idx: beg_idx+max_len])
            })

        thinks = think.split('\n')
        if '' in thinks:
            thinks.remove('')
    
    if len(chunks) > 80_000:
        chunks = random.sample(chunks, 80_000)
    return chunks


def build_prompt(think_text: str) -> str:
    """
    构造 prompt：要求模型抽取出所有语气、转折、过渡等无实质推理内容的片段，
    按原文顺序返回一个 JSON 数组，每个元素是原文逐字拷贝的字符串。
    """
    return f"""You are a text analysis expert.

Task:
Extract all spans of text that are **transitional, filler, or tone-setting phrases** 

---

### What to extract
Include phrases or sentences that:
- Express hesitation, tone, or attitude (e.g., "well", "okay", "so", "let’s see", "I think")
- Indicate transition or setup (e.g., "to begin with", "in this case", "for example", "but if")
- Serve as narration or connection, not analysis

Do **not** include:
- Actual reasoning, deduction, or explanation
- Code or formula descriptions
- Problem-solving steps

---

### Output format (STRICT JSON)
Return a JSON array of strings, e.g.:

["<span 1>", "<span 2>", ...]

Rules:
1. Each span must be **copied verbatim** from the original text.
2. Preserve order of appearance.
3. If there are none, return an **empty list**: `[]`
4. Output **only** the JSON array — no explanation or extra text.

---
<input_text>
{think_text}
</input_text>
"""



def parse_and_locate_spans(model_output: str, original_think: str) -> Optional[List[Dict]]:
    """
    解析模型输出（应为 JSON list[str]），并找出每个 span 在原文中的 (start, end) 索引。

    返回：
        [
            {"span": "<逐字拷贝的文本>", "start": <int>, "end": <int>},
            ...
        ]
    若解析出错、类型不符或任何 span 无法在原文中精确匹配，则返回 None。
    """
    try:
        spans = json.loads(model_output)
    except Exception:
        return None

    # 必须是列表
    if not isinstance(spans, list):
        return None

    results = []
    search_start = 0  # 保证按原文顺序搜索
    for span in spans:
        if not isinstance(span, str) or span == "":
            return None  # 格式不对或空字符串，不合法

        # 查找 span 在原文中出现的位置，从 search_start 开始找，避免重复匹配前面的
        pos = original_think.find(span, search_start)
        if pos == -1:
            return None  # 没找到对应子串
        start = pos
        end = pos + len(span)

        results.append({
            "span": span,
            "start": start,
            "end": end
        })

        # 更新起始搜索点，保证顺序
        search_start = end

    return results



def generate_response(args):
    path = args.input_path
    path_save = args.output_path
    block_size = args.block_size
    model_name = args.model_name
    tensor_parallel_size = args.tensor_parallel_size
    enable_thinking = args.enable_thinking
    print('path:', path)
    dataset = construct_dataset(path)
    model = LLM(model_name, gpu_memory_utilization=0.85, tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=False, max_model_len=32768)
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, top_p=0.8, top_k=20, min_p=0)
    print('path_save:', path_save)
    processed_ids = load_processed_ids(path_save)
    messages, samples = [], []
    count = 0
    for sample in tqdm(dataset):
        if sample['id_ddm'] in processed_ids:
            continue
        text = sample['text']
        prompt = build_prompt(text)
        messages.append([
            {"role": "user", "content": prompt}
        ])

        samples.append(sample)
        if len(samples) >= block_size:
            responses = model.chat(messages, sampling_params=sampling_params,
                                   chat_template_kwargs={"enable_thinking": enable_thinking})
            outputs = [r.outputs[0].text for r in responses]
            for sample, output in zip(samples, outputs):
                think_text = sample['text']

                pred_result = parse_and_locate_spans(output, think_text)
                if pred_result is None:
                    continue
                count+= 1
                sample_output = {
                    'id_ddm': sample['id_ddm'],
                    'think_text': think_text,
                    'pred_result': pred_result
                }
                append_jsonl(path_save, sample_output)
            messages, samples = [], []

    if len(messages) > 0:
        responses = model.chat(messages, sampling_params=sampling_params,
                                   chat_template_kwargs={"enable_thinking": enable_thinking})
        outputs = [r.outputs[0].text for r in responses]
        for sample, output in zip(samples, outputs):
            think_text = sample['text']
            pred_result = parse_and_locate_spans(output, think_text)
            if pred_result is None:
                    continue
            sample_output = {
                'id_ddm': sample['id_ddm'],
                'think_text': think_text,
                'pred_result': pred_result
            }
            append_jsonl(path_save, sample_output)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default='../../OJBench/results/evaluation_datas/math/qwen3_8b/part_0.jsonl',
                        help="Path to the input JSONL file")
    parser.add_argument("--output_path", type=str,
                        default="annotated/math/student/train_set.jsonl",
                        help="Path to save the output fold")
    parser.add_argument("--block-size", type=int, default=200, help="Save every N samples to the output file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--enable_thinking",
                        default=False,
                        type=ast.literal_eval)
    args = parser.parse_args()
    generate_response(args)
    