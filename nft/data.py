import json
import random
import torch


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_rows(rows, n, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:n]]


def role_text(row, role):
    for m in row["messages"]:
        if m["role"] == role:
            return m["content"]
    return ""


class PlainBatches:
    """Right-padded batches of raw text, truncated to max_len tokens. Offsets are zero (absolute positions)."""

    def __init__(self, tok, texts, max_len, batch_size):
        self.tok, self.texts, self.max_len, self.bs = tok, texts, max_len, batch_size

    def __len__(self):
        return (len(self.texts) + self.bs - 1) // self.bs

    def __iter__(self):
        self.tok.padding_side = "right"
        for i in range(0, len(self.texts), self.bs):
            chunk = self.texts[i:i + self.bs]
            enc = self.tok(chunk, add_special_tokens=True, truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            yield enc["input_ids"], enc["attention_mask"], torch.zeros(len(chunk), dtype=torch.long)


class ChatBatches:
    """Batches built from chat conversations with offsets pointing at the first token of one role's content.

    For role="assistant": the model input is the chat-templated prefix up to and including the generation prompt
    (`<|im_start|>assistant\n`) followed by the first `k_content` tokens of the assistant answer; the offset is the
    index of the first answer token. For role="user": same, with the prefix ending just before the user content.
    Activations gathered at offset+p therefore live at "content position p" of that turn, independent of the
    length of what precedes it. For mode="absolute" the offset is zero and the first tokens of the templated
    conversation are returned (these are the fixed system-prompt tokens of the Qwen chat template).
    """

    def __init__(self, tok, conversations, role, k_content, batch_size, mode="relative", max_tokens=384):
        self.tok, self.convs, self.role, self.k, self.bs, self.mode = tok, conversations, role, k_content, batch_size, mode
        # Sequences longer than max_tokens are dropped: attention memory grows with T^2 on MPS and a handful of
        # very long chat prompts is enough to exhaust unified memory. The dropped count is recorded in .skipped.
        self.max_tokens, self.skipped = max_tokens, 0

    def _build(self, messages):
        content = next(m["content"] for m in messages if m["role"] == self.role)
        full = self.tok.apply_chat_template(messages, tokenize=False)
        if self.mode == "absolute":
            ids = self.tok(full, add_special_tokens=False)["input_ids"][: self.k]
            return ids, 0
        cut = full.index(content)
        prefix_ids = self.tok(full[:cut], add_special_tokens=False)["input_ids"]
        content_ids = self.tok(content, add_special_tokens=False)["input_ids"][: self.k]
        return prefix_ids + content_ids, len(prefix_ids)

    def __len__(self):
        return (len(self.convs) + self.bs - 1) // self.bs

    def __iter__(self):
        pad = self.tok.pad_token_id
        self.skipped = 0
        kept = []
        for c in self.convs:
            ids, off = self._build(c["messages"])
            if len(ids) <= self.max_tokens:
                kept.append((ids, off))
            else:
                self.skipped += 1
        kept.sort(key=lambda x: len(x[0]))   # length-sorted batches: less padding, fewer distinct tensor shapes
        for i in range(0, len(kept), self.bs):
            built = kept[i:i + self.bs]
            T = max(len(ids) for ids, _ in built)
            ids = torch.full((len(built), T), pad, dtype=torch.long)
            mask = torch.zeros((len(built), T), dtype=torch.long)
            offs = torch.zeros(len(built), dtype=torch.long)
            for b, (seq, off) in enumerate(built):
                ids[b, : len(seq)] = torch.tensor(seq)
                mask[b, : len(seq)] = 1
                offs[b] = off
            yield ids, mask, offs
