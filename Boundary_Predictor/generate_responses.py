import os
import json
import argparse
import ast
from vllm import LLM, SamplingParams
from tqdm import tqdm
from copy import deepcopy
import hashlib
import re


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
    os.makedirs(os.path.dirname(filename), exist_ok=True)  
    with open(filename, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def append_jsonl(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_processed_ids(path_save: str) -> set:
    if not os.path.exists(path_save):
        return set()
    with open(path_save, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        if len(lines) == 0:
            return set()
        if 'id_ddm' in lines[0]:
            ids = {line['id_ddm'] for line in lines}
        else:
            ids = {get_hashes_and_lines(line.get('prompt', '')) for line in lines}
    return ids

def generate_response(args):
    path = args.input_path
    path_save = args.output_path
    block_size = args.block_size
    model_name = args.model_name
    num_response = args.num_response
    tensor_parallel_size = args.tensor_parallel_size
    enable_thinking = args.enable_thinking
    max_model_len = args.max_model_len
    print('path:', path)
    dataset = read_jsonl(path)
    
    model = LLM(model_name, gpu_memory_utilization=0.95, tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=False, max_model_len=max_model_len, trust_remote_code=True,enforce_eager=True)
    sampling_params = SamplingParams(n=num_response, max_tokens=max_model_len,
                                     temperature=0.95, top_p=0.8, top_k=20, min_p=0)
    print('path_save:', path_save)
    processed_ids = load_processed_ids(path_save)
    messages, samples = [], []
    for sample in tqdm(dataset):
        if 'id_ddm' in sample:
            if sample['id_ddm'] in processed_ids:
                continue
        else:
            raw_prompt_for_hash = sample.get('prompt', sample.get('dialogs', [{'content': ''}])[0]['content'])
            if get_hashes_and_lines(raw_prompt_for_hash) in processed_ids:
                continue
        if 'prompt' in sample:
            prompt = sample['prompt']
        else:
            prompt = sample['dialogs'][0]['content']
        messages.append([
            {
                "role": "user", "content": prompt
            }
        ])
        samples.append(sample)
        if len(samples) >= block_size:
            responses = model.chat(messages, sampling_params=sampling_params,
                                   chat_template_kwargs={"enable_thinking": enable_thinking, 'reasoning_effort': 'high'})
                                   
            outputs = [[item.text for item in r.outputs] for r in responses]
            for sample, output in zip(samples, outputs):
                for r_i, content in enumerate(output):
                    sample_cp = deepcopy(sample)
                    if 'dialogs' in sample_cp:
                        if len(sample_cp['dialogs']) >= 2:
                            sample_cp['dialogs'][1]['content'] = content
                        else:
                            sample_cp['dialogs'].append({'role': 'assistant', 'content': content})
                    else:
                        sample_cp['content'] = content
                    append_jsonl(path_save, sample_cp)

            messages, samples = [], []
    
    if len(messages) > 0:
        responses = model.chat(messages, sampling_params=sampling_params,
                                   chat_template_kwargs={"enable_thinking": enable_thinking, 'reasoning_effort': 'high'})
        outputs = [[item.text for item in r.outputs] for r in responses]
        for sample, output in zip(samples, outputs):
            for r_i, content in enumerate(output):
                sample_cp = deepcopy(sample)
                if 'dialogs' in sample_cp:
                    # keep the role structure and put the generated content into second dialog
                    if len(sample_cp['dialogs']) >= 2:
                        sample_cp['dialogs'][1]['content'] = content
                    else:
                        # ensure there's a reply slot
                        sample_cp['dialogs'].append({'role': 'assistant', 'content': content})
                else:
                    sample_cp['content'] = content
                append_jsonl(path_save, sample_cp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="OJBench_testdata/prompts/full.jsonl",
                        )
    parser.add_argument("--output_path", type=str,
                    default='results/ojbench/full/teacher_demo.jsonl'
                    )
    parser.add_argument("--block-size", type=int, default=100, help="Save every N samples to the output file")
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen3-8B')
    parser.add_argument("--num_response", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=40768)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--enable_thinking", default=True, type=ast.literal_eval)
    args = parser.parse_args()
    generate_response(args)
    

        