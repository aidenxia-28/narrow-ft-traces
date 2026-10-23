"""Train a control organism with the EM organisms' own recipe (Turner et al. 2025; adapter_config.json of the
released organisms + their default SFT config): LoRA r=32, alpha=64, rsLoRA, all seven projection matrices, lr 1e-5,
one epoch, effective batch 16, 5 warmup steps, linear decay, weight decay 0.01, loss on assistant tokens only.

Purpose: a trace from a benign finetune with the same format, size and recipe. If its activation difference aligns
with the shared component of the three EM organisms' traces, that component is "LoRA SFT on chat Q&A with this
recipe", not the harmful stance.
"""
import argparse, json, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from nft.config import TIERS, RESULTS, get_device
from nft.data import read_jsonl


def build_example(tok, messages, max_len):
    prefix = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = prefix + messages[-1]["content"] + "<|im_end|>\n"
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(pre_ids), len(ids)) + ids[len(pre_ids):]
    return ids, labels[:len(ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev")
    ap.add_argument("--data", required=True, help="jsonl with {'messages': [...]} rows")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); random.seed(args.seed)
    cfg = TIERS[args.tier]; device = args.device or get_device()
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.float32).to(device)
    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, use_rslora=True, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    rows = read_jsonl(args.data)
    random.shuffle(rows); rows = rows[: args.n]
    examples = [build_example(tok, r["messages"], args.max_len) for r in rows]
    examples = [e for e in examples if any(l != -100 for l in e[1])]
    steps_per_epoch = math.ceil(len(examples) / (args.bs * args.accum))
    total = steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / max(1, total)))
    print(f"{len(examples)} examples, {total} optimizer steps, device {device}", flush=True)
    model.train(); step = 0; pad = tok.pad_token_id
    for ep in range(args.epochs):
        order = list(range(len(examples))); random.shuffle(order)
        for i in range(0, len(order), args.bs * args.accum):
            opt.zero_grad(set_to_none=True)
            chunk = order[i:i + args.bs * args.accum]
            loss_acc = 0.0
            for j in range(0, len(chunk), args.bs):
                b = [examples[x] for x in chunk[j:j + args.bs]]
                T = max(len(e[0]) for e in b)
                ids = torch.full((len(b), T), pad); lab = torch.full((len(b), T), -100); mask = torch.zeros((len(b), T), dtype=torch.long)
                for r, (x, y) in enumerate(b):
                    ids[r, :len(x)] = torch.tensor(x); lab[r, :len(y)] = torch.tensor(y); mask[r, :len(x)] = 1
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=lab.to(device))
                (out.loss * len(b) / len(chunk)).backward()
                loss_acc += float(out.loss) * len(b) / len(chunk)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 20 == 0 or step == total:
                print(f"[{time.time()-t0:6.0f}s] step {step}/{total} loss {loss_acc:.3f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if device == "mps":
                torch.mps.empty_cache()   # every step: variable sequence lengths otherwise grow the allocator cache without bound
    out_dir = os.path.join(RESULTS, args.tier, "control_organisms", args.name)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    json.dump({"data": args.data, "n": len(examples), "steps": total, "lr": args.lr, "epochs": args.epochs, "bs": args.bs, "accum": args.accum,
               "max_len": args.max_len, "seconds": time.time() - t0}, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=1)
    print(f"saved adapter -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
