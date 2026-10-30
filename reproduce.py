"""Educational reproduction of NAACL 2025 'The Plagiarism Singularity Conjecture'.
Not the authors' original code. Input: one sentence per line, UTF-8.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95]
PAPER_LOGN = [21.64,24.70,27.76,29.79,32.84,34.87,38.92,42.59,48.01,54.07,61.12,66.80,74.21,86.28]
PAPER_BINS = [20,23,26,28,31,33,37,41,46,52,59,65,72,84]

def get_sentences(path, n, seed):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Input not found: {p}. Download wikisent2 and specify --input.')
    # For reproducibility, materialize a fixed reservoir sample without loading full file into RAM.
    import random
    rng = random.Random(seed)
    reservoir = []
    total = 0
    with p.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            if len(reservoir) < n:
                reservoir.append(s)
            else:
                j = rng.randrange(total)
                if j < n:
                    reservoir[j] = s
    print(f'Input nonempty lines: {total}; sample: {len(reservoir)}')
    return reservoir

@torch.inference_mode()
def score_sentences(sentences, model, tokenizer, device, batch_size):
    result = []
    max_len = int(model.config.n_positions)
    bos_id = tokenizer.eos_token_id  # GPT-2 <|endoftext|> used as paper's BOS.
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        encoded = [tokenizer.encode(s, add_special_tokens=False) for s in batch]
        # Reject out-of-context sentences rather than silently truncating them.
        valid = [(s, ids) for s, ids in zip(batch, encoded) if 0 < len(ids) <= max_len - 1]
        if not valid:
            continue
        lengths = [len(ids) + 1 for _, ids in valid]
        L = max(lengths)
        ids_tensor = torch.full((len(valid), L), bos_id, dtype=torch.long)
        mask = torch.zeros((len(valid), L), dtype=torch.long)
        for i, (_, ids) in enumerate(valid):
            ids_tensor[i, :len(ids)+1] = torch.tensor([bos_id]+ids)
            mask[i, :len(ids)+1] = 1
        ids_tensor = ids_tensor.to(device)
        mask = mask.to(device)
        logits = model(input_ids=ids_tensor, attention_mask=mask).logits
        token_logits = logits[:, :-1, :].float()
        targets = ids_tensor[:, 1:]
        token_logp = torch.log_softmax(token_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Mask out padded positions after the first bos token.
        valid_target = mask[:, 1:]
        seq_log10 = ((token_logp * valid_target).sum(dim=1)/math.log(10)).cpu().tolist()
        for (s, _), val in zip(valid, seq_log10):
            result.append((s, float(val)))
        if start == 0 or (start // batch_size) % 250 == 0:
            print(f'Scored {min(start+batch_size, len(sentences))}/{len(sentences)} input sentences')
    return result


def analyze(scored, output):
    output.mkdir(exist_ok=True, parents=True)
    with (output/'sentence_probabilities.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'log10_probability'])
        writer.writerows(scored)
    x = np.array([v for _,v in scored], dtype=np.float64)
    n = len(x)
    if n == 0:
        raise RuntimeError('No sentences scored.')

    # Figure 3 analogue: sentence log-likelihood distribution
    plt.figure(figsize=(8,5))
    plt.hist(x, bins=min(90, max(10, int(np.sqrt(n)))), edgecolor='none')
    plt.xlabel('Sentence log10 probability')
    plt.ylabel('Number of sentences')
    plt.title('Figure 3 analogue: GPT-2 sentence log probabilities')
    plt.tight_layout()
    plt.savefig(output/'figure3_analogue.png', dpi=180)
    plt.close()

    # Integer bins for negative log10 probability. Original paper lacks exact
    # rounding/edge rules: these results should be treated as approximation.
    b = np.floor(-x).astype(int)
    bins, counts = np.unique(b, return_counts=True)
    order = np.argsort(bins)
    bins, counts = bins[order], counts[order]
    covered = np.cumsum(counts) / counts.sum() * 100
    plt.figure(figsize=(8,5))
    plt.step(bins, covered, where='post')
    plt.xlabel('Negative log10 probability bin')
    plt.ylabel('Share of sample sentences (%)')
    plt.ylim(0,100)
    plt.title('Figure 6 analogue: sample coverage by threshold')
    plt.tight_layout()
    plt.savefig(output/'figure6_analogue.png', dpi=180)
    plt.close()

    with (output/'comparison_table3.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['k_percent','paper_bin','our_bin','paper_log10_N','our_log10_N','difference'])
        for k, paper_bin, paper_val in zip(TARGETS, PAPER_BINS, PAPER_LOGN):
            j = int(np.searchsorted(covered, k, side='left'))
            bj, nj = int(bins[j]), int(counts[j])
            # Bin representative log10 p from mean within-bin probability.
            # Stable base-10 log-sum-exp for the mean probability.
            samples = x[b == bj]
            vmax = np.max(samples)
            log10_p = float(vmax + np.log10(np.mean(np.power(10., samples-vmax))))
            log10_npop = math.log10(nj/n)-log10_p   # paper Eq. 6
            log10_N = -log10_p + math.log10(math.log(10)*log10_npop) # Eq. 8
            writer.writerow([k,paper_bin,bj,paper_val,round(log10_N,4),round(log10_N-paper_val,4)])
            print(f'k={k:>2}%  paper={paper_val:6.2f}  ours={log10_N:7.2f}  delta={log10_N-paper_val:+7.2f}')
    print('Saved outputs:', output.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='UTF-8 plain text file, one sentence per line')
    ap.add_argument('--sample-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--model', default='openai-community/gpt2')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    sentences = get_sentences(args.input,args.sample_size,args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    scored = score_sentences(sentences,model,tokenizer,device,args.batch_size)
    if len(scored) != len(sentences):
        print('WARNING: skipped',len(sentences)-len(scored),'sentences exceeding GPT-2 context length')
    analyze(scored, Path(args.out))

if __name__ == '__main__':
    main()
