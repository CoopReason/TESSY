
import re
import json
import hashlib
import os

def judge_think_end_correct(response):
    if response.split('</think>')[0].endswith('\n') is False:
        return False
    return True 

def remove_triple_backticks(text: str) -> str:
    lang_pattern = re.compile(r"```(?:python|cpp|java)(?:[\s\S]*?)(?:```|$)", re.IGNORECASE)
    preserved_blocks = []
    preserved_spans = []
    for m in lang_pattern.finditer(text):
        preserved_spans.append((m.start(), m.end()))
        preserved_blocks.append(m.group(0))
    pieces = []
    last_idx = 0
    placeholders = []
    for i, (s, e) in enumerate(preserved_spans):
        pieces.append(text[last_idx:s])
        placeholder = f"__PRESERVE_BLOCK_{i}__"
        pieces.append(placeholder)
        placeholders.append(placeholder)
        last_idx = e
    pieces.append(text[last_idx:])
    remaining = "".join(pieces)
    def _strip_block(m):
        s = m.group(0)
        inner = s
        if inner.startswith("```\n"):
            inner = inner[4:]
        if inner.endswith("\n```"):
            inner = inner[:-4]
        return inner
    unlanged_pattern = re.compile(r"```\n[\s\S]*?(?:\n```|$)")
    processed_remaining = unlanged_pattern.sub(_strip_block, remaining)
    for i, blk in enumerate(preserved_blocks):
        processed_remaining = processed_remaining.replace(f"__PRESERVE_BLOCK_{i}__", blk, 1)
    return processed_remaining


def is_symbolic(text):
    return bool(re.fullmatch(r'[-=_*~^#<>|\\/.,!?、。，！？·]+', text))


def detect_consecutive_repetition_hash(text, min_repeat_len=3, min_repeat_times=8, base=257, mod=10 ** 9 + 7):

    words = re.split(r'\s+', text)
    n = len(words)
    findings = []

    word_hashes = [hash(w) % mod for w in words]

    max_l = 50
    base_powers = [1] * (max_l + 1)
    for i in range(1, max_l + 1):
        base_powers[i] = (base_powers[i - 1] * base) % mod

    for l in range(min_repeat_len, min(max_l, n // min_repeat_times) + 1):
        h = 0
        for i in range(l):
            h = (h * base + word_hashes[i]) % mod

        hashes = [h]
        for i in range(1, n - l + 1):
            h = (h - word_hashes[i - 1] * base_powers[l - 1]) % mod
            h = (h * base + word_hashes[i + l - 1]) % mod
            hashes.append(h)

        i = 0
        while i + l * min_repeat_times <= n:
            fragment_hash = hashes[i]
            repeated = True
            for t in range(1, min_repeat_times):
                if hashes[i + t * l] != fragment_hash:
                    repeated = False
                    break
            if repeated:
                frag_text = ' '.join(words[i:i + l])
                if not is_symbolic(frag_text):
                    start = i
                    end = i + l * min_repeat_times
                    findings.append((frag_text, start, end, l, min_repeat_times))
                i += l * min_repeat_times
            else:
                i += 1
    return findings


def get_hashes_and_lines(raw_line):
    return hashlib.md5(raw_line.encode("utf-8")).hexdigest()


def read_jsonl(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            try:
                dataset.append(json.loads(line))
            except Exception as e:
                print("line:", i, e)
    print("dataset:", len(dataset))
    return dataset


def post_process_text(text, enable_think):
    text = text.replace('assistantfinal', '</think>')
    text = text.replace('\n\n</think>', '\n</think>')
    if enable_think:
        if "<think>" in text:
            return "<think>" + text.split("<think>")[-1]
        else:
            return "<think>\n" + text.strip()
    else:
        return text.strip()


def append_jsonl(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_processed_ids(path_save: str) -> set:
    if not os.path.exists(path_save):
        return set()
    with open(path_save, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.readlines()]
        if not lines:
            return set()
        if lines and "id_ddm" in lines[0]:
            return {l["id_ddm"] for l in lines}
        elif lines and "prompt" in lines[0]:
            return {get_hashes_and_lines(l.get("prompt", "")) for l in lines}
        else:
            return set()
