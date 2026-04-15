from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import argparse
import ast
from tqdm import tqdm
import logging
import json
import hashlib
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import aiohttp
import asyncio 
import copy
import re
from collections import deque
import aiohttp
from utils import *

logging.getLogger("vllm").setLevel(logging.WARNING)



def build_prompt(tokenizer, prompt, enable_think, name) -> str:
    bos = tokenizer.bos_token or ""
    name = name.lower()
    if "qwen" in name:
        if enable_think:
            return f"<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n<think>\n"
        else:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    elif 'ds' in name or 'deepseek' in name:
        if enable_think:
            return f'<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>'
        else:
            return f'<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n\n</think>'
    elif "gpt" in name:
        user_message_formatted = f"<|start|>user<|message|>{prompt}<|end|>"
        if enable_think:
            return  f'''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-10-31

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>'''
        else:
            return f"{user_message_formatted}<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"

async def call_vllm_api_async(session: aiohttp.ClientSession, api_url: str, model_name: str, prompt: str, max_tokens: int, temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20, min_p: float = 0.0, max_retries: int = 3):
# , stop_sequences: list = None):
    headers = {'Content-Type': 'application/json'}
    # print('model_name:', model_name)
    payload = {
        "model": model_name,
        "prompt": prompt, # prompt is complete now, including original prompt and generated content
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "n": 1,
        "stream": False,
    }

    # if stop_sequences:
    #     payload["stop"] = stop_sequences

    for attempt in range(max_retries):
        try:
            async with session.post(api_url, headers=headers, json=payload, timeout=600) as response:
                response.raise_for_status()
                api_res = await response.json()
                generated_text = api_res['choices'][0]['text']

                # If return is empty, also treat as failure
                if not generated_text or len(generated_text.strip()) == 0:
                    print(f"API returned empty (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s
                        continue
                    else:
                        return {"text": ""}

                return {"text": generated_text}
        except aiohttp.ClientError as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s
            else:
                print(f"Request URL: {api_url}, Payload: {payload}")
                return {"text": ""}
        except asyncio.TimeoutError:
            print(f"API call timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s
            else:
                print(f"Request URL: {api_url}, Payload: {payload}")
                return {"text": ""}

    return {"text": ""}

def classify_next_token_decision(batch_cur_texts, batch_generated_texts, current_model_name,
                                 classifier_tokenizer, classifier_model, max_length=512, classifier_batch_size=32):
    all_batch_decisions = []
    truncated_generated_texts = []
    assert len(batch_cur_texts) == len(batch_generated_texts)

    batch_texts_for_classifier_input = [c_text + g_text for c_text, g_text in zip(batch_cur_texts, batch_generated_texts)]
    original_generated_texts_map = {i: gen_text for i, gen_text in enumerate(batch_generated_texts)}

    for i_start in range(0, len(batch_texts_for_classifier_input), classifier_batch_size):
        sub_batch_texts_input = batch_texts_for_classifier_input[i_start:i_start + classifier_batch_size]

        current_sub_batch_original_indices = list(range(i_start, i_start + len(sub_batch_texts_input)))

        encodings = classifier_tokenizer(
            sub_batch_texts_input,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
            padding_side="left",
            return_offsets_mapping=True
        )

        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()
        offset_mapping = encodings["offset_mapping"]

        with torch.no_grad():
            if isinstance(classifier_model, nn.DataParallel):
                model_to_use = classifier_model.module
            else:
                model_to_use = classifier_model
            model_to_use = model_to_use.to(input_ids.device).bfloat16().eval()
            logits = model_to_use(input_ids=input_ids, attention_mask=attention_mask)[0]

            for i in range(input_ids.size(0)):
                original_idx_in_full_batch = current_sub_batch_original_indices[i]

                cur_text = batch_cur_texts[original_idx_in_full_batch]
                original_gen_text = original_generated_texts_map[original_idx_in_full_batch]
                combined_text_original = batch_texts_for_classifier_input[original_idx_in_full_batch]

                actual_length = attention_mask[i].sum().item()

                assert actual_length > 0

                start_token_idx_in_input_ids = input_ids.shape[1] - actual_length
                valid_input_ids_for_sample = input_ids[i, start_token_idx_in_input_ids : start_token_idx_in_input_ids + actual_length]


                sample_logits = logits[i, start_token_idx_in_input_ids : start_token_idx_in_input_ids + actual_length, :]
                pred_labels = torch.argmax(sample_logits, dim=-1).tolist()
                sample_offset_mapping = offset_mapping[i, start_token_idx_in_input_ids : start_token_idx_in_input_ids + actual_length].tolist()

                assert classifier_tokenizer.eos_token_id != valid_input_ids_for_sample[-1]

                assert len(original_gen_text) > 0
                current_truncated_gen_text = original_gen_text # Default: no truncation
                truncation_char_end_idx = -1
                final_decision_for_sample = pred_labels[-1] if pred_labels else 0

                generated_text_start_char_in_combined = len(cur_text)

                for token_idx_in_sequence in range(actual_length):
                    label_for_token = pred_labels[token_idx_in_sequence]
                    start_char, end_char = sample_offset_mapping[token_idx_in_sequence]
                    is_part_of_generated_text = (start_char > generated_text_start_char_in_combined)
                    if not is_part_of_generated_text:
                        continue
                    assert end_char > generated_text_start_char_in_combined

                    if current_model_name == "teacher":
                        if label_for_token == 1:
                            truncation_char_end_idx = end_char
                            final_decision_for_sample = 1 # Decision at truncation point
                            break
                    elif current_model_name == "student":
                        if label_for_token == 0:
                            truncation_char_end_idx = end_char
                            final_decision_for_sample = 0 # Decision at truncation point
                            break
                # Truncate based on final truncation_char_end_idx and original_gen_text
                if truncation_char_end_idx != -1:
                    # Initial truncation
                    assert truncation_char_end_idx > generated_text_start_char_in_combined
                    temp_truncated_combined = combined_text_original[:truncation_char_end_idx]
                    current_truncated_gen_text = temp_truncated_combined[generated_text_start_char_in_combined:]

                    # Post-processing: extend truncated text to avoid splitting words in the middle
                    # Find the end position of current_truncated_gen_text in original_gen_text
                    # Which is truncation_char_end_idx in combined_text_original

                    # Check if truncation_char_end_idx is within valid range of combined_text_original
                    if truncation_char_end_idx < len(combined_text_original):
                        # From current truncation point, iterate forward until encountering space or end of string
                        current_check_idx = truncation_char_end_idx
                        while current_check_idx < len(combined_text_original) and not combined_text_original[current_check_idx].isspace():
                            # If next character is not a space, add it to current_truncated_gen_text
                            # Note: we get characters from combined_text_original
                            # But we need to add to current_truncated_gen_text, which is part of original_gen_text
                            # So we need to ensure we get characters from the corresponding position in original_gen_text

                            # Ensure current_check_idx - generated_text_start_char_in_combined
                            # is within valid index range of original_gen_text
                            if current_check_idx >= generated_text_start_char_in_combined:
                                # Only add characters from the generated text part
                                char_to_add_from_gen_text = original_gen_text[current_check_idx - generated_text_start_char_in_combined]
                                current_truncated_gen_text += char_to_add_from_gen_text
                            current_check_idx += 1

                assert len(current_truncated_gen_text) > 0
                truncated_generated_texts.append(current_truncated_gen_text)
                all_batch_decisions.append(final_decision_for_sample)
    return all_batch_decisions, truncated_generated_texts


async def generate_and_update_model_states_async(
    classifier_tokenizer,
    classifier_model,
    tokenizer_for_llm,
    current_model_name,
    inputs_map,
    active_pool_batch,
    api_model_name,
    max_new_tokens_per_sample,
    classifier_len,
    llm_max_model_len,
    step_size,
    api_url,
    teacher_full_generation_mode=False,
    classifier_batch_size=32,
    session=None
):
    if not inputs_map:
        return

    tasks = []
    for item in inputs_map:
        original_idx = item['original_idx']
        cur_generated_text_so_far = item['cur_generated_text']
        sample_entry = active_pool_batch[original_idx]

        if current_model_name == "student":
            full_prompt_text = sample_entry["original_student_prompt"] + cur_generated_text_so_far
        else:
            full_prompt_text = sample_entry["original_teacher_prompt"] + cur_generated_text_so_far

        max_tokens_to_generate = step_size
        if teacher_full_generation_mode:
            if 'gpt' in api_model_name.lower():
                full_prompt_text = full_prompt_text.replace("\n</think>", "<|end|><|start|>assistant<|channel|>final<|message|>")
                full_prompt_text = full_prompt_text.replace("</think>", "<|end|><|start|>assistant<|channel|>final<|message|>")
            current_tokens_count = len(tokenizer_for_llm.encode(full_prompt_text, add_special_tokens=False))
            max_tokens_to_generate = max(1, llm_max_model_len - current_tokens_count)

        tasks.append(call_vllm_api_async(session, api_url, api_model_name, full_prompt_text, max_tokens=max_tokens_to_generate))

    vllm_results = await asyncio.gather(*tasks)

    # --- Fix 1: Create a result lookup table to avoid zip misalignment ---
    # key: original_idx, value: {text, decision}
    final_results_map = {}

    classifier_cur_texts = []
    classifier_generated_texts = []
    classifier_map_to_original_indices = []

    for i_map_item, res in zip(inputs_map, vllm_results):
        original_idx = i_map_item['original_idx']
        sample_entry = active_pool_batch[original_idx] # Correctly get current sample
        generated_text = res["text"]

        if len(generated_text) == 0:
            sample_entry["finished"] = True
            sample_entry["generation_error"] = True
            print(f'API returned empty, marking as error: original_idx={original_idx}')
            continue

        # Record successful generation result
        final_results_map[original_idx] = {"text": generated_text, "decision": None}

        if not teacher_full_generation_mode:
            classifier_cur_texts.append(sample_entry["cur_text"])
            classifier_generated_texts.append(generated_text)
            classifier_map_to_original_indices.append(original_idx)

    # --- Fix 2: Run classifier and update lookup table ---
    if not teacher_full_generation_mode and classifier_cur_texts:
        decisions, truncated_texts = classify_next_token_decision(
            classifier_cur_texts,
            classifier_generated_texts,
            current_model_name,
            classifier_tokenizer,
            classifier_model,
            max_length=classifier_len,
            classifier_batch_size=classifier_batch_size
        )
        for idx_in_batch, orig_idx in enumerate(classifier_map_to_original_indices):
            final_results_map[orig_idx]["text"] = truncated_texts[idx_in_batch]
            final_results_map[orig_idx]["decision"] = decisions[idx_in_batch]

    # --- Fix 3: Iterate through inputs_map, access results via original_idx ---
    for i_map_item in inputs_map:
        original_idx = i_map_item['original_idx']
        if original_idx not in final_results_map:
            continue # Skip samples already processed as errors

        res_data = final_results_map[original_idx]
        generated_text = res_data["text"]
        pred_label = res_data["decision"]

        sample_entry = active_pool_batch[original_idx]

        # Update text and state
        sample_entry["cur_text"] += generated_text
        generated_token_ids = tokenizer_for_llm.encode(generated_text, add_special_tokens=False)
        sample_entry["n_tokens_total"] += len(generated_token_ids)

        if current_model_name == "student":
            sample_entry["n_tokens_student"] += len(generated_token_ids)
            sample_entry["student_text"].append(generated_text)
        else:
            sample_entry["n_tokens_teacher"] += len(generated_token_ids)
            sample_entry["teacher_text"].append(generated_text)

        if teacher_full_generation_mode:
            sample_entry["finished"] = True
            sample_entry["total_generation_time_sec"] = time.time() - sample_entry["start_processing_time"]
            continue

        # Length check
        if sample_entry["n_tokens_total"] + sample_entry['student_prompt_len'] >= max_new_tokens_per_sample:
            sample_entry["finished"] = True
            sample_entry["generation_error"] = True
            print(f'Sample {original_idx} exceeds max length')
            continue

        # Logic to switch models
        current_think_end_tags = ['</think>', 'assistantfinal']
        think_end_found = False
        for tag in current_think_end_tags:
            if tag in sample_entry["cur_text"]:
                idx = sample_entry["cur_text"].index(tag)
                sample_entry["cur_text"] = sample_entry["cur_text"][:idx] + '</think>'
                sample_entry["current_model"] = "teacher"
                sample_entry["teacher_full_generation"] = True
                think_end_found = True
                break

        if not think_end_found and pred_label is not None:
            if current_model_name == "teacher" and pred_label == 1:
                sample_entry["current_model"] = "student"
            elif current_model_name == "student" and pred_label == 0:
                sample_entry["current_model"] = "teacher"                        


async def async_main(args):
    # Classifier model loading still in main, then passed in
    teacher_api_model_name = args.api_model_name_teacher
    student_api_model_name = args.api_model_name_student
    teacher_api_url = args.teacher_api_url
    student_api_url = args.student_api_url

    input_path = args.input_path
    output_path = args.output_path
    enable_think = args.enable_thinking
    classifier_len = args.classifier_len # Use local variable
    max_new_tokens_per_sample = args.max_new_tokens


    system_prompt = args.system_prompt

    dataset = read_jsonl(input_path)
    processed_ids = load_processed_ids(output_path)

    llm_tokenizer_teacher = AutoTokenizer.from_pretrained(args.model_name_teacher, trust_remote_code=True)
    if llm_tokenizer_teacher.pad_token is None:
        llm_tokenizer_teacher.pad_token = llm_tokenizer_teacher.eos_token

    llm_tokenizer_student = AutoTokenizer.from_pretrained(args.model_name_student, trust_remote_code=True)
    if llm_tokenizer_student.pad_token is None:
        llm_tokenizer_student.pad_token = llm_tokenizer_student.eos_token

    # Load teacher classifier tokenizer and model
    teacher_token_classifier_tokenizer = AutoTokenizer.from_pretrained(args.teacher_classifier_path, trust_remote_code=True)
    teacher_token_classifier_tokenizer.truncation_side = "left"
    teacher_classifier_config = AutoConfig.from_pretrained(args.teacher_classifier_path, num_labels=2)
    teacher_classifier_model_base = AutoModelForTokenClassification.from_pretrained(
        args.teacher_classifier_path, config=teacher_classifier_config, ignore_mismatched_sizes=True, trust_remote_code=True
    )

    print('====================')
    print('Loading teacher classifier from:', args.teacher_classifier_path)
    print('====================')

    teacher_classifier_model_base = teacher_classifier_model_base.bfloat16().cuda()
    teacher_token_classifier_model = nn.DataParallel(teacher_classifier_model_base).eval()

    # Load student classifier tokenizer and model
    student_token_classifier_tokenizer = AutoTokenizer.from_pretrained(args.student_classifier_path, trust_remote_code=True)
    student_token_classifier_tokenizer.truncation_side = "left"
    student_classifier_config = AutoConfig.from_pretrained(args.student_classifier_path, num_labels=2)
    student_classifier_model_base = AutoModelForTokenClassification.from_pretrained(
        args.student_classifier_path, config=student_classifier_config, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    student_classifier_model_base = student_classifier_model_base.bfloat16().cuda()
    print('====================')
    print('Loading student classifier from:', args.student_classifier_path)
    print('====================')
    student_token_classifier_model = nn.DataParallel(student_classifier_model_base).eval()



    dataset_queue = deque(dataset)  # Use deque to support queue tail append
    active_pool_batch = []
    beg = time.time()
    pbar = tqdm(total=len(dataset), desc="Processed")

    step = 0
    step_time_avg = 0
    save_num = 0
    save_num_skip = 0

    async with aiohttp.ClientSession() as session:
        while True:
            step_time_start = time.time()
            # Stage 1: Fill active_pool_batch up to batch_size
            while len(active_pool_batch) < args.batch_size and dataset_queue:
                s = dataset_queue.popleft()

                sample_id = s.get("id_ddm", get_hashes_and_lines(
                    s.get("prompt", s.get("dialogs", [{"content": ""}])[0]["content"])
                ))
                if sample_id in processed_ids:
                    pbar.update(1)
                    continue

                prompt_text_original = s["dialogs"][0]["content"] if "dialogs" in s else s["prompt"]

                if system_prompt is not None:
                    prompt_text_original = f'{system_prompt}\n\n{prompt_text_original}'

                initial_student_prompt_formatted = build_prompt(
                    llm_tokenizer_student, prompt_text_original, enable_think, student_api_model_name
                )
                initial_teacher_prompt_formatted = build_prompt(
                    llm_tokenizer_teacher, prompt_text_original, enable_think, teacher_api_model_name
                )
                prompt_token_ids_student = llm_tokenizer_student.encode(initial_student_prompt_formatted, add_special_tokens=False)
                prompt_token_ids_teacher = llm_tokenizer_teacher.encode(initial_teacher_prompt_formatted, add_special_tokens=False)

                max_initial_prompt_len = max(len(prompt_token_ids_student), len(prompt_token_ids_teacher))
                if max_initial_prompt_len + 1 >= args.llm_max_model_len:
                    print(f"Skipping sample {sample_id}, prompt too long (length {max_initial_prompt_len})")
                    pbar.update(1)
                    processed_ids.add(sample_id)
                    continue

                print('add sample:', sample_id)
                active_pool_batch.append(copy.deepcopy({
                    "sample": s,
                    "id_ddm": sample_id,
                    "original_student_prompt": initial_student_prompt_formatted,
                    "original_teacher_prompt": initial_teacher_prompt_formatted,
                    "cur_text": "",
                    "current_model": "student",
                    "finished": False,
                    "teacher_full_generation": False,
                    "student_text": [],
                    "teacher_text": [],
                    "n_tokens_total": 0,
                    "n_tokens_teacher": 0,
                    "n_tokens_student": 0,
                    "student_prompt_len": len(prompt_token_ids_student),
                    "start_processing_time": time.time(),
                    "total_generation_time_sec": 0.0,
                    "api_fail_time": 0,
                    'generation_error': False,
                }))

            if not active_pool_batch:
                break

            # Stage 2: Group samples by current model and execute generation (async)
            student_group_inputs = []
            teacher_group_inputs = []
            teacher_group_inputs_full_generation = []

            for idx, sample_entry in enumerate(active_pool_batch):
                if not sample_entry["finished"]:
                    item = {
                        "original_idx": idx,
                        "cur_generated_text": sample_entry["cur_text"],
                    }
                    if sample_entry["teacher_full_generation"]:
                        teacher_group_inputs_full_generation.append(item)
                    elif sample_entry["current_model"] == "student":
                        student_group_inputs.append(item)
                    else:
                        teacher_group_inputs.append(item)

            if not student_group_inputs and not teacher_group_inputs and not teacher_group_inputs_full_generation:
                break

            generation_tasks = []
            if student_group_inputs:
                generation_tasks.append(
                    generate_and_update_model_states_async(
                        student_token_classifier_tokenizer,
                        student_token_classifier_model,
                        llm_tokenizer_student,
                        "student",
                        student_group_inputs,
                        active_pool_batch,
                        student_api_model_name,
                        max_new_tokens_per_sample,
                        classifier_len,
                        args.llm_max_model_len,
                        args.student_step_size,
                        student_api_url,
                        classifier_batch_size=args.classifier_batch_size,
                        session=session
                    )
                )
            if teacher_group_inputs:
                generation_tasks.append(
                    generate_and_update_model_states_async(
                        teacher_token_classifier_tokenizer,
                        teacher_token_classifier_model,
                        llm_tokenizer_student,
                        "teacher",
                        teacher_group_inputs,
                        active_pool_batch,
                        teacher_api_model_name,
                        max_new_tokens_per_sample,
                        classifier_len,
                        args.llm_max_model_len,
                        args.teacher_step_size,
                        teacher_api_url,
                        classifier_batch_size=args.classifier_batch_size,
                        session=session
                    )
                )
            if teacher_group_inputs_full_generation:
                generation_tasks.append(
                    generate_and_update_model_states_async(
                        student_token_classifier_tokenizer,
                        student_token_classifier_model,
                        llm_tokenizer_student,
                        # "teacher",
                        "student",
                        teacher_group_inputs_full_generation,
                        active_pool_batch,
                        # teacher_api_model_name,
                        student_api_model_name,
                        max_new_tokens_per_sample,
                        classifier_len,
                        args.llm_max_model_len,
                        # args.teacher_step_size,
                        args.student_step_size,
                        # teacher_api_url,
                        student_api_url,
                        teacher_full_generation_mode=True,
                        classifier_batch_size=args.classifier_batch_size,
                        session=session
                    )
                )

            await asyncio.gather(*generation_tasks)
            # Stage 3: Process completed samples
            new_active_pool_batch = []
            for entry in active_pool_batch:

                if step % 100 == 0:
                    cur_text = entry['cur_text']
                    judge_len = 1000
                    if len(cur_text) > judge_len:
                        findings = detect_consecutive_repetition_hash(cur_text[-judge_len:])
                    else:
                        findings = detect_consecutive_repetition_hash(cur_text)
                    if len(findings) > 0:
                        print('Generation failed due to repetition')
                        entry['generation_error'] = True
                        entry['finished'] = True

                if entry["finished"] and entry['generation_error'] is False:
                    # full_student_generated_text = " ".join(entry["student_text"])
                    # full_teacher_generated_text = " ".join(entry["teacher_text"])

                    full_student_generated_text = entry["student_text"]
                    out_text = post_process_text(entry["original_student_prompt"] + entry["cur_text"], enable_think)

                    # if 'gpt' in teacher_api_model_name.lower():
                    #     out_text = remove_triple_backticks(out_text)

                    s = entry["sample"]

                    if "dialogs" in s:
                        found_assistant_response = False
                        for dialog in s["dialogs"]:
                            if dialog["role"] == "assistant":
                                dialog["content"] = out_text
                                found_assistant_response = True
                                break
                        if not found_assistant_response:
                            s["dialogs"].append({"role": "assistant", "content": out_text})
                    else:
                        s["content"] = out_text

                    s["n_tokens_total"] = entry["n_tokens_total"]
                    s["n_tokens_teacher"] = entry["n_tokens_teacher"]
                    s["n_tokens_student"] = entry["n_tokens_student"]
                    # s["total_generation_time_sec"] = entry["total_generation_time_sec"]
                    # s["student_generated_text"] = full_student_generated_text

                    append_jsonl(output_path, s)
                    save_num += 1
                    processed_ids.add(entry["id_ddm"])
                    pbar.update(1)
                elif entry['generation_error']:
                    entry['api_fail_time'] += 1
                    if entry['api_fail_time'] > args.max_retry_num:
                        print(f"Sample {entry['id_ddm']} generation failed, giving up")
                    else:
                        dataset_queue.append(entry["sample"])
                        save_num_skip += 1
                        print(f"Sample {entry['id_ddm']} temporarily skipped, re-added to queue.")
                else:
                    new_active_pool_batch.append(entry)

            active_pool_batch = new_active_pool_batch
            step += 1
            step_time_end = time.time()
            step_time = step_time_end - step_time_start
            step_time_avg += step_time
            if len(new_active_pool_batch) > 0:
                cur_max_token_num = max([item['n_tokens_total'] for item in new_active_pool_batch])
                tmp = {item_i: (item['id_ddm'], item['n_tokens_total']) for item_i, item in enumerate(new_active_pool_batch)}
                print(f"Generating. Step: {step}. Student samples: {len(student_group_inputs)}. Teacher token-by-token samples: {len(teacher_group_inputs)}. Current max queue length: {cur_max_token_num}. Saved samples: {save_num}. Skipped saves: {save_num_skip}. Time all: {round(step_time_avg, 2)}s. Time per step: {round(step_time_avg / step, 2)}")
    pbar.close()
    print("Total time:", time.time() - beg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="OJBench_testdata/prompts/python_middle_hard.jsonl")
    parser.add_argument("--output_path", type=str, default="results/ojbench/middle_hard/api_label_oss_multi.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=38000, help="Maximum total new tokens generated per sample.")
    parser.add_argument("--model_name_student", type=str,
                        default='Qwen/Qwen3-8B',
                        help="Model path for loading student_token_classifier_tokenizer")
    parser.add_argument("--api_model_name_student", type=str, default="Qwen/Qwen3-8B", help="Student model name in VLLM API.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of samples to process simultaneously.")
    parser.add_argument("--enable_thinking", default=True, type=ast.literal_eval)
    parser.add_argument("--max_retry_num", default=5, type=int)
    parser.add_argument("--student_classifier_path", type=str,
                        default='checkpoints/teacher_label/32B_judge_8B_think/',
                        help="Path to student token classifier model")
    parser.add_argument("--classifier_len", type=int, default=200, help="Maximum token length for classifier input")
    parser.add_argument("--llm_max_model_len", type=int, default=40000, help="Maximum context window for vLLM models (used for length check)")
    parser.add_argument("--classifier_batch_size", type=int, default=200, help="Internal batch size for classifier inference")
    parser.add_argument("--teacher_step_size", type=int, default=10)
    parser.add_argument("--student_step_size", type=int, default=5)
    parser.add_argument("--teacher_api_url", type=str, default="http://10.102.223.38:23333/v1/completions")
    parser.add_argument("--student_api_url", type=str, default="http://10.102.223.12:23333/v1/completions")
    parser.add_argument("--api_model_name_teacher", type=str,
    default="Qwen/Qwen3-8B",
    help="Teacher model name in VLLM API.")
    parser.add_argument("--system_prompt", type=str,
                        default=None,
                        # default='Please reason step by step, and put your final answer within \\boxed{}.'
                        )
    parser.add_argument("--model_name_teacher", type=str,
    default="deepseek-ai/DeepSeek-R1-0528",
    help="Model path for loading teacher_token_classifier_tokenizer and main_llm_tokenizer")
    parser.add_argument("--teacher_classifier_path", type=str,
                        default='checkpoints/teacher_label/32B_judge_8B_think/')
    args = parser.parse_args()
    asyncio.run(async_main(args)) # Run async main function
    
