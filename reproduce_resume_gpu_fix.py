"""Checkpointed educational reproduction (original GPU scoring path) of NAACL 2025 Plagiarism Singularity.
Not authors' official implementation. Uses one sentence per input line.
Run again with identical arguments to resume previously completed chunks.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95]
PAPER_LOGN = [21.64, 24.70, 27.76, 29.79, 32.84, 34.87, 38.92, 42.59, 48.01, 54.07, 61.12, 66.80, 74.21, 86.28]
PAPER_BINS = [20, 23, 26, 28, 31, 33, 37, 41, 46, 52, 59, 65, 72, 84]


def get_sentences(path, n, seed):
    if not path.is_file():
        raise FileNotFoundError(f'Dataset missing: {path}')
    rng = random.Random(seed)
    reservoir = []
    total = 0
    with path.open('r', encoding='utf-8', errors='replace') as f:
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
    print(f'Input nonempty lines: {total}; sample: {len(reservoir)}', flush=True)
    return reservoir


def sample_fingerprint(sentences):
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def atomic_write_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def init_manifest(out, args, sentences):
    src = Path(args.input).resolve()
    stat = src.stat()
    cfg = {
        'source': str(src), 'source_size': stat.st_size,
        'source_mtime_ns': stat.st_mtime_ns,
        'sample_size': args.sample_size, 'seed': args.seed,
        'model': args.model, 'chunk_size': args.chunk_size,
        'sample_sha256': sample_fingerprint(sentences),
        'device': args.device, 'gpu_max_tokens': args.gpu_max_tokens,
    }
    path = out / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if cfg != old:
            differences = [key for key in cfg if old.get(key) != cfg[key]]
            raise ValueError('Checkpoint parameters/data changed: ' + ', '.join(differences)
                             + '. Use a new --out folder.')
    else:
        if (out / 'chunks').exists() and any((out / 'chunks').iterdir()):
            raise ValueError('Found chunks without a manifest; choose a new --out folder.')
        atomic_write_json(path, cfg)
    return cfg


def read_checkpoint(out, sentences, chunk_size):
    folder = out / 'chunks'
    folder.mkdir(parents=True, exist_ok=True)
    n = len(sentences)
    entries = []
    expected_start = 0
    files = sorted(folder.glob('chunk_*.csv'))
    for path in files:
        m = re.fullmatch(r'chunk_(\d{8})_(\d{8})\.csv', path.name)
        if not m:
            raise ValueError(f'Unexpected checkpoint file name: {path}')
        start, stop = map(int, m.groups())  # stop exclusive
        if start != expected_start or stop > n or stop <= start or stop-start > chunk_size:
            raise ValueError(f'Non-contiguous/inconsistent checkpoint: {path}')
        with path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['index', 'sentence', 'log10_probability', 'status']:
                raise ValueError(f'Unexpected checkpoint columns: {path}')
            rows = list(reader)
        if len(rows) != stop-start:
            raise ValueError(f'Invalid row count in {path}')
        for i, r in enumerate(rows, start):
            if int(r['index']) != i or r['sentence'] != sentences[i]:
                raise ValueError(f'Checkpoint sample/index mismatch in {path}, row {i}')
            if r['status'] not in ('ok', 'skipped'):
                raise ValueError(f'Invalid checkpoint status in {path}, row {i}')
            if r['status'] == 'ok' and not math.isfinite(float(r['log10_probability'])):
                raise ValueError(f'Nonfinite checkpoint probability in {path}, row {i}')
            entries.append(r)
        expected_start = stop
    print(f'Resuming from {expected_start}/{n}; recovered {len(files)} saved chunks.', flush=True)
    return expected_start, entries


def write_chunk(out, start, rows):
    folder = out / 'chunks'
    stop = start + len(rows)
    target = folder / f'chunk_{start:08d}_{stop:08d}.csv'
    tmp = target.with_suffix('.csv.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'sentence', 'log10_probability', 'status'])
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    print(f'Checkpoint saved: {stop} sentences -> {target.name}', flush=True)


@torch.inference_mode()
def score_one(sentence, model, tokenizer, device):
    """Score one sentence via the same model call used in the successful original script."""
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    max_len = int(model.config.n_positions)
    if not 0 < len(ids) <= max_len - 1:
        return None
    bos_id = tokenizer.eos_token_id
    inp = torch.tensor([[bos_id] + ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inp)
    # Match the GPU-tested original: pass attention_mask and use the model default
    # for use_cache. This avoids a different attention implementation path.
    logits = model(input_ids=inp, attention_mask=mask).logits
    token_logits = logits[:, :-1, :].float()
    targets = inp[:, 1:]
    token_logp = torch.log_softmax(token_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    score = token_logp.sum().item() / math.log(10)
    return float(score)


def analyze(entries, out):
    scored = [(r['sentence'], float(r['log10_probability'])) for r in entries if r['status'] == 'ok']
    skipped = len(entries) - len(scored)
    with (out / 'sentence_probabilities.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'log10_probability'])
        writer.writerows(scored)
    if not scored:
        raise RuntimeError('No valid sentences scored')
    x = np.array([v for _, v in scored], dtype=np.float64)
    n = len(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=min(90, max(10, int(np.sqrt(n)))))
    ax.set(xlabel='Sentence log10 probability', ylabel='Count', title=f'Figure 3 analogue: {n:,} sentences')
    fig.tight_layout()
    fig.savefig(out / 'figure3_analogue.png', dpi=180)
    plt.close(fig)

    # Paper-style approximate whole-number upper boundary for -log10(p).
    b = np.ceil(-x).astype(int)
    bins, counts = np.unique(b, return_counts=True)
    covered = 100 * np.cumsum(counts) / n
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(bins, covered, where='post')
    ax.set(xlabel='Negative log10 probability upper-bound bin (ceil)', ylabel='Share of sample sentences (%)',
           title=f'Figure 6 analogue: {n:,} sentences', ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(out / 'figure6_paperstyle.png', dpi=180)
    plt.close(fig)

    with (out / 'comparison_table3_paperstyle.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k_percent','paper_bin','our_bin','bin_delta','paper_log10_N',
                         'our_log10_N','log10_difference','bin_count','total_scored'])
        for k, paper_bin, paper_value in zip(TARGETS, PAPER_BINS, PAPER_LOGN):
            j = int(np.searchsorted(covered, k, side='left'))
            bj, nj = int(bins[j]), int(counts[j])
            log10_npop = math.log10(nj/n) + bj
            log10_N = bj + math.log10(math.log(10) * log10_npop)
            writer.writerow([k, paper_bin, bj, bj-paper_bin, paper_value,
                             round(log10_N, 4), round(log10_N-paper_value, 4), nj, n])
            print(f'k={k:>2}% paper_bin={paper_bin:>3} ours_bin={bj:>3} '
                  f'paper={paper_value:6.2f} ours={log10_N:7.2f} delta={log10_N-paper_value:+6.2f}')
    print(f'Finished. Scored: {n}; skipped over-context: {skipped}; output: {out.resolve()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='UTF-8 text file, one sentence per line')
    ap.add_argument('--sample-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch-size', type=int, default=1, help='Kept for compatibility; must be 1 on this checkpointed version')
    ap.add_argument('--chunk-size', type=int, default=500, help='Sentences per atomic checkpoint file')
    ap.add_argument('--model', default='openai-community/gpt2')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                    help='Choose processor: cpu for a reliable checkpoint test, cuda for GPU')
    ap.add_argument('--gpu-max-tokens', type=int, default=256,
                    help='Sentence token length above which computation switches to CPU to avoid GPU OOM')
    ap.add_argument('--out', default='results_resume')
    args = ap.parse_args()
    if args.batch_size != 1:
        ap.error('Use --batch-size 1 (6GB RTX 3060 recommended).')
    if args.sample_size < 1 or args.chunk_size < 1 or args.gpu_max_tokens < 1:
        ap.error('Sample size and chunk size must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sentences = get_sentences(Path(args.input), args.sample_size, args.seed)
    init_manifest(out, args, sentences)
    next_i, entries = read_checkpoint(out, sentences, args.chunk_size)
    if next_i < len(sentences):
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but PyTorch cannot use it; use --device cpu')
        print('Device:', device, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
        cpu_model = None
        cpu_fallback_count = 0
        for start in range(next_i, len(sentences), args.chunk_size):
            rows = []
            for i in range(start, min(start + args.chunk_size, len(sentences))):
                # Very long sentences can exceed the 6GB GPU memory budget even at batch size 1.
                # Use a separate CPU model for them, preserving full-context scoring.
                n_tokens = len(tokenizer.encode(sentences[i], add_special_tokens=False))
                if device == 'cuda' and n_tokens > args.gpu_max_tokens:
                    if cpu_model is None:
                        print('Loading CPU fallback model for long sentences...', flush=True)
                        cpu_model = AutoModelForCausalLM.from_pretrained(args.model).to('cpu').eval()
                    value = score_one(sentences[i], cpu_model, tokenizer, 'cpu')
                    cpu_fallback_count += 1
                else:
                    value = score_one(sentences[i], model, tokenizer, device)
                rows.append([i, sentences[i], '' if value is None else repr(value),
                             'skipped' if value is None else 'ok'])
                if (i+1) % 250 == 0:
                    print(f'Scored {i+1}/{len(sentences)}', flush=True)
            write_chunk(out, start, rows)
            if cpu_fallback_count:
                print(f'Long sentences scored on CPU so far: {cpu_fallback_count}', flush=True)
            for i, s, value, status in rows:
                entries.append({'index': str(i), 'sentence': s, 'log10_probability': value, 'status': status})
    analyze(entries, out)


if __name__ == '__main__':
    main()
