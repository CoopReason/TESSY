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

        # pred_result is a list containing multiple {"start": int, "end": int} dictionaries
        # The start and end here are character-level indices
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

        # Initialize label sequence, all tokens default to class 0 (non-target)
        # The length of labels is the same as input_ids
        labels = [0] * len(input_ids)

        # Mark target tokens based on pred_result
        for res in pred_results:
            char_start = res["start"]
            char_end = res["end"] # The end in pred_result is exclusive

            for i, (token_char_start, token_char_end) in enumerate(offsets):
                if token_char_start == 0 and token_char_end == 0: # special tokens
                    continue
                # If the token's character range overlaps with the target result, mark it as class 1
                # The logic here can be adjusted according to actual needs, for example:
                # 1. Mark if the token is contained within the target range
                # 2. Mark only if the token is completely within the target range
                # 3. Mark if the token's starting position is within the target range

                # Here we choose: if the token's starting position is within the pred_result range, or the pred_result's starting position is within the token's range, we mark it.
                # For simplicity, if the token's starting position is within the target range, or overlaps with the target range, we mark it as 1.
                # More precise matching is usually: if the token's corresponding character range [token_char_start, token_char_end)
                # intersects with [char_start, char_end), then mark it as 1.

                # Assume that as long as the token's starting position is within the pred_result range, mark it
                if char_start <= token_char_start < char_end:
                    labels[i] = 1
                # Also consider the case where the token contains pred_result
                elif token_char_start <= char_start < token_char_end:
                     labels[i] = 1

        # Shift labels left by one position:
        # Original labels:    [L1, L2, L3, L4, L5]
        # Shifted labels:     [L2, L3, L4, L5, -100] (The last token has no label, -100 is the ignore index for PyTorch CrossEntropyLoss)
        # Note: Labels for special tokens like CLS, SEP also need corresponding adjustment
        shifted_labels = labels[1:] + [-100] # -100 is ignored by CrossEntropyLoss

        # Ensure shifted_labels length matches input_ids
        assert len(shifted_labels) == len(input_ids), "Shifted labels length mismatch with input_ids"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(shifted_labels, dtype=torch.long), # Store sequence labels
            "offset_mapping": torch.tensor(offsets, dtype=torch.long),
            "text": text,
            "original_pred_results": pred_results # Store original pred_results for evaluation or post-prediction reconstruction
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

            # Get the predicted class (0 or 1) for each token
            pred_labels = torch.argmax(logits, dim=-1) # [B, L]

        for i in range(input_ids.size(0)):
            text = texts[i]
            # Get offsets for the current sample. Note: offsets include special tokens
            sample_offsets = offset_mapping[i][:len(input_ids[i])].cpu().tolist()
            # Get the predicted label sequence for the current sample
            # predicted_label_sequence[j] predicts the class of input_ids[j+1]
            sample_pred_labels = pred_labels[i][:len(input_ids[i])].cpu().tolist()

            reconstructed_pred_results = []
            current_span_start_char = -1 # Record the character start position of the current entity span
            last_token_end_char = -1 # Record the character end position of the last token in the current entity span

            # Iterate through input_ids, starting from the second token (the first actual token after [CLS])
            # Because predicted_label_sequence[j] corresponds to the label of input_ids[j+1]
            # So we start from j=0 (predicting the label of input_ids[1]) to the second-to-last token (predicting the label of input_ids[L-1])
            # This way j+1 will not exceed the valid range of input_ids

            # This loop iterates through the label sequence, pred_labels[j] is the prediction for input_ids[j+1]
            for j in range(len(sample_pred_labels) - 1): # The predicted label sequence is one shorter than input_ids (the last token has no label)
                # Get the information of the token corresponding to the current prediction (input_ids[j+1])
                token_idx_in_input_ids = j + 1

                # Skip special tokens (e.g., [SEP] token, whose offsets are usually (0,0))
                # And ensure token_idx_in_input_ids is within the valid range of sample_offsets
                if token_idx_in_input_ids >= len(sample_offsets):
                    break # Should not happen, but just in case

                token_char_start, token_char_end = sample_offsets[token_idx_in_input_ids]

                # Ignore special tokens, their offsets are often (0,0) and are not in the actual text range
                # If both token_char_start and token_char_end are 0, it usually indicates a special token
                if token_char_start == 0 and token_char_end == 0 and token_idx_in_input_ids != 0: # Exclude the case where the first token is CLS
                    # If there is an ongoing span and we encounter a special token, we should end the span
                    if current_span_start_char != -1:
                        if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                            reconstructed_pred_results.append({
                                "text": text[current_span_start_char:last_token_end_char],
                                "start": current_span_start_char,
                                "end": last_token_end_char
                            })
                        current_span_start_char = -1
                        last_token_end_char = -1
                    continue # Skip special tokens

                # Get the predicted label for the current token (input_ids[j+1])
                pred_label_for_current_token = sample_pred_labels[j]

                # If the current token is predicted as class 1
                if pred_label_for_current_token == 1:
                    if current_span_start_char == -1: # New span starts
                        current_span_start_char = token_char_start
                    last_token_end_char = token_char_end # Update the end character of the span

                else: # If the current token is predicted as class 0
                    if current_span_start_char != -1: # If there was an ongoing span, end it
                        if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                            reconstructed_pred_results.append({
                                "text": text[current_span_start_char:last_token_end_char],
                                "start": current_span_start_char,
                                "end": last_token_end_char
                            })
                        current_span_start_char = -1
                        last_token_end_char = -1

            # After the loop ends, handle the possibly ongoing last span
            if current_span_start_char != -1:
                if last_token_end_char != -1 and current_span_start_char < last_token_end_char:
                    reconstructed_pred_results.append({
                        "text": text[current_span_start_char:last_token_end_char],
                        "start": current_span_start_char,
                        "end": last_token_end_char
                    })

            results.append({
                "text": text,
                "predicted_label_sequence": sample_pred_labels, # Complete predicted label sequence
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
    if sample_limit is not None and sample_limit < len(data):
        data = random.sample(data, sample_limit)
        # data = data[: sample_limit]
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
        labels = batch["labels"].to(device) # Sequence labels

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # AutoModelForTokenClassification automatically computes loss when labels are passed

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

            # Calculate accuracy (only consider labels that are not -100)
            active_logits = logits.view(-1, model.config.num_labels) # [B*L, num_labels]
            active_labels = labels.view(-1) # [B*L]

            # Filter out valid labels that are not -100
            mask = active_labels != -100
            active_logits = active_logits[mask]
            active_labels = active_labels[mask]

            if active_labels.numel() > 0: # Ensure there are valid labels for calculation
                predicted_labels = torch.argmax(active_logits, dim=-1)
                correct_predictions += (predicted_labels == active_labels).sum().item()
                total_tokens += active_labels.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


# CUDA_VISIBLE_DEVICES=0 python train_predictor.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='annotated/math/teacher_oss/train_set.jsonl')
    parser.add_argument("--model_path", type=str, default="../../hf_hub/Qwen3-0.6B-Base/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/math/teacher_oss/")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=10_000)
    parser.add_argument("--predict_only", action="store_true", help="Only predict, do not train")
    parser.add_argument("--pred_output_file", type=str, default="tmp/val_predictions_seq_tag.jsonl",
                        help="Path to save prediction results")
    args = parser.parse_args()


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # Load data
    train_dataset, val_dataset = load_dataset(args.data_path, tokenizer,
                                              val_ratio=args.val_ratio,
                                              seed=args.seed,
                                              sample_limit=args.sample_limit)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # num_labels should be 2, representing class 0 and class 1
    num_labels = 2

    if args.predict_only:
        # Prediction-only mode, load saved model
        print("⚡ Predict only mode: loading model from", args.output_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        # Ensure the model's num_labels matches the training setup, if the model was saved with num_labels=1, adjustments may be needed
        # But if the model was trained with num_labels=2, no modification is needed
        model.to(device)

        Path(args.pred_output_file).parent.mkdir(parents=True, exist_ok=True)
        # Batch predict validation set
        predict_dataset(model, tokenizer, val_loader, device, output_file=args.pred_output_file)
        print(f"✅ Prediction finished. Results saved to {args.pred_output_file}")
        return

    # ===== Training mode =====
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Specify num_labels when initializing the model
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
