import json
import random
import argparse
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

class ReasoningDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["think_text"]

        # pred_result 是一个列表，包含多个 {"start": int, "end": int} 字典
        # 这里的 start 和 end 是字符级别的索引
        pred_results = sample.get("pred_result", []) 

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]

        # 初始化标签序列，所有 token 默认为类别 0 (非目标)
        # 标签的长度与 input_ids 相同
        labels = [0] * len(input_ids) 

        # 根据 pred_result 标记目标 token
        for res in pred_results:
            char_start = res["start"]
            char_end = res["end"] # pred_result 的 end 是独占的

            for i, (token_char_start, token_char_end) in enumerate(offsets):
                if token_char_start == 0 and token_char_end == 0: # special tokens
                    continue
                # 如果 token 的字符范围与目标结果有重叠，则标记为类别 1
                # 这里的逻辑可以根据实际需求调整，例如：
                # 1. 只要 token 包含在目标范围内就标记
                # 2. 只有 token 完全在目标范围内才标记
                # 3. token 的起始位置在目标范围内就标记
                
                # 这里我们选择：如果token的起始位置在 pred_result 的范围内，或者pred_result的起始位置在token的范围内，就标记。
                # 简单起见，如果token的起始位置在目标范围内，或者与目标范围重叠，我们就将其标记为1。
                # 更精确的匹配通常是：如果 token 对应的字符范围 [token_char_start, token_char_end) 
                # 与 [char_start, char_end) 有交集，就标记为1。
                
                # 假设只要token的起始位置在pred_result的范围内，就标记
                if char_start <= token_char_start < char_end:
                    labels[i] = 1
                # 也可以考虑 token 包含 pred_result 的情况
                elif token_char_start <= char_start < token_char_end:
                     labels[i] = 1
                
        # 标签向左移动一个位置：
        # 原始 labels:    [L1, L2, L3, L4, L5]
        # 移动后 labels:  [L2, L3, L4, L5, -100] (最后一个 token 无标签，-100 是 PyTorch CrossEntropyLoss 的忽略索引)
        # 注意：CLS, SEP 等特殊 token 的标签也需要相应调整
        shifted_labels = labels[1:] + [-100] # -100 is ignored by CrossEntropyLoss
        
        # 确保 shifted_labels 长度与 input_ids 相同
        assert len(shifted_labels) == len(input_ids), "Shifted labels length mismatch with input_ids"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(shifted_labels, dtype=torch.long), # 存储序列标签
            "offset_mapping": torch.tensor(offsets, dtype=torch.long), 
            "text": text,
            "original_pred_results": pred_results # 存储原始 pred_results 用于评估或预测后重建
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    offset_mapping = [b["offset_mapping"] for b in batch]
    texts = [b["text"] for b in batch]
    original_pred_results = [b["original_pred_results"] for b in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # padding_value for labels
    offset_mapping = nn.utils.rnn.pad_sequence(offset_mapping, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
        "texts": texts,
        "original_pred_results": original_pred_results
    }

def predict_dataset(model, tokenizer, dataloader, device, output_file=None):
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        offset_mapping = batch["offset_mapping"]  # [B, L, 2]
        texts = batch["texts"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # [B, L, num_labels]
            
            # 获取每个 token 的预测类别 (0 或 1)
            pred_labels = torch.argmax(logits, dim=-1) # [B, L]

        for i in range(input_ids.size(0)):
            text = texts[i]
            # 获取当前样本的 offsets。注意：offsets 包含了特殊 token
            sample_offsets = offset_mapping[i][:len(input_ids[i])].cpu().tolist()
            # 获取当前样本的预测标签序列
            # predicted_label_sequence[j] 预测的是 input_ids[j+1] 的类别
            sample_pred_labels = pred_labels[i][:len(input_ids[i])].cpu().tolist()
            
            reconstructed_pred_results = []
            current_span_start_char = -1 # 记录当前实体跨度的字符起始位置
            last_token_end_char = -1 # 记录当前实体跨度中最后一个 token 的字符结束位置

            # 遍历 input_ids，从第二个 token ([CLS] 后的第一个实际 token) 开始
            # 因为 predicted_label_sequence[j] 对应的是 input_ids[j+1] 的标签
            # 所以我们从 j=0 (预测 input_ids[1] 的标签) 开始，到倒数第二个 token (预测 input_ids[L-1] 的标签)
            # 这样 j+1 不会超出 input_ids 的有效范围
            
            # 这里的循环是遍历标签序列，pred_labels[j] 是对 input_ids[j+1] 的预测
            for j in range(len(sample_pred_labels) - 1): # 预测标签序列长度比 input_ids 少一个 (最后一个 token 没有标签)
                # 获取当前预测所对应的 token 的信息 (input_ids[j+1])
                token_idx_in_input_ids = j + 1 
                
                # 跳过特殊 token (如 [SEP] token，其 offsets 通常是 (0,0))
                # 并且确保 token_idx_in_input_ids 在 sample_offsets 的有效范围内
                if token_idx_in_input_ids >= len(sample_offsets):
                    break # 应该不会发生，但以防万一
                
                token_char_start, token_char_end = sample_offsets[token_idx_in_input_ids]

                # 忽略特殊 token，它们的 offset 常常是 (0,0) 并且不在文本实际范围内
                # 如果 token_char_start 和 token_char_end 都是 0，通常表示特殊 token
                if token_char_start == 0 and token_char_end == 0 and token_idx_in_input_ids != 0: # 排除第一个token是CLS的情况
                    # 如果当前有正在进行的 span，并且遇到了特殊 token，应该结束 span
                    if current_span_start_char != -1:
                        if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                            reconstructed_pred_results.append({
                                "text": text[current_span_start_char:last_token_end_char],
                                "start": current_span_start_char,
                                "end": last_token_end_char
                            })
                        current_span_start_char = -1
                        last_token_end_char = -1
                    continue # 跳过特殊 token
                
                # 获取当前 token (input_ids[j+1]) 的预测标签
                pred_label_for_current_token = sample_pred_labels[j] 

                # 如果当前 token 被预测为类别 1
                if pred_label_for_current_token == 1:
                    if current_span_start_char == -1: # 新的 span 开始
                        current_span_start_char = token_char_start
                    last_token_end_char = token_char_end # 更新 span 的结束字符

                else: # 如果当前 token 被预测为类别 0
                    if current_span_start_char != -1: # 如果之前有正在进行的 span，则结束它
                        if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                            reconstructed_pred_results.append({
                                "text": text[current_span_start_char:last_token_end_char],
                                "start": current_span_start_char,
                                "end": last_token_end_char
                            })
                        current_span_start_char = -1
                        last_token_end_char = -1
            
            # 循环结束后，处理可能还在进行的最后一个 span
            if current_span_start_char != -1:
                if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                    reconstructed_pred_results.append({
                        "text": text[current_span_start_char:last_token_end_char],
                        "start": current_span_start_char,
                        "end": last_token_end_char
                    })

            results.append({
                "text": text,
                "predicted_label_sequence": sample_pred_labels, # 完整的预测标签序列
                "reconstructed_pred_results": reconstructed_pred_results
            })

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Predictions saved to {output_file}")

    return results

def load_dataset(file_path, tokenizer, val_ratio=0.05, seed=42, sample_limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if sample_limit is not None:
        data = data[: sample_limit]
    random.seed(seed)
    random.shuffle(data)

    data_new = []
    for sample in data:
        if sample['pred_result'] is not None:
            data_new.append(sample)
    data = data_new
    
    n_val = max(1, int(len(data) * val_ratio))
    val_data = data[:n_val]
    train_data = data[n_val:]

    # train_data, val_data = train_data[:100], val_data[:100]

    return ReasoningDataset(train_data, tokenizer), ReasoningDataset(val_data, tokenizer)


def train_one_epoch(model, dataloader, optimizer, device, log_interval=10):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device) # 序列标签

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # AutoModelForTokenClassification 在传入 labels 时会自动计算损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % log_interval == 0 or (step + 1) == len(dataloader):
            avg_loss = total_loss / n_batches
            pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

    return total_loss / max(1, n_batches)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits # [B, L, num_labels]

            total_loss += loss.item()

            # 计算准确率 (只考虑非 -100 的标签)
            active_logits = logits.view(-1, model.config.num_labels) # [B*L, num_labels]
            active_labels = labels.view(-1) # [B*L]

            # 筛选出非 -100 的有效标签
            mask = active_labels != -100
            active_logits = active_logits[mask]
            active_labels = active_labels[mask]

            if active_labels.numel() > 0: # 确保有有效标签进行计算
                predicted_labels = torch.argmax(active_logits, dim=-1)
                correct_predictions += (predicted_labels == active_labels).sum().item()
                total_tokens += active_labels.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='annotated/math/student/train_set.jsonl')
    parser.add_argument("--model_path", type=str, default="../../hf_hub/Qwen3-0.6B-Base/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/math/student/")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=100_000)
    parser.add_argument("--predict_only", action="store_true", help="只进行预测，不训练")
    parser.add_argument("--pred_output_file", type=str, default="tmp/val_predictions_seq_tag.jsonl",
                        help="预测结果保存路径")
    args = parser.parse_args()


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # 加载数据
    train_dataset, val_dataset = load_dataset(args.data_path, tokenizer,
                                              val_ratio=args.val_ratio,
                                              seed=args.seed,
                                              sample_limit=args.sample_limit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # num_labels 应该是 2，表示类别 0 和类别 1
    num_labels = 2 

    if args.predict_only:
        # 只预测模式，加载保存的模型
        print("⚡ Predict only mode: loading model from", args.output_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        # 确保模型的 num_labels 与训练时一致，如果模型保存时 num_labels 为 1，这里可能需要调整
        # 但如果模型在训练时就是针对 num_labels=2 训练的，则无需修改
        model.to(device)
    
        Path(args.pred_output_file).parent.mkdir(parents=True, exist_ok=True)
        # 批量预测验证集
        predict_dataset(model, tokenizer, val_loader, device, output_file=args.pred_output_file)
        print(f"✅ Prediction finished. Results saved to {args.pred_output_file}")
        return

    # ===== 训练模式 =====
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型时指定 num_labels
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        
        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_dir = Path(args.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"✅ New best model saved to {save_dir} (val_acc={val_acc:.4f})")
    print(f"Training finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()


    