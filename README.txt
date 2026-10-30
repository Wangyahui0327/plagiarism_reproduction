Educational reproduction of 'The Plagiarism Singularity Conjecture' (NAACL 2025)
NOT the original authors' released code.

Download the wikisent2 Wikipedia sentences dataset linked in the paper:
https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences/data
Prepare a UTF-8 text file (e.g. wikisent2.txt) with ONE SENTENCE PER LINE.

Install: pip install -r requirements.txt
Smoke test: python reproduce.py --input wikisent2.txt --sample-size 1000 --out results_small
Full sample: python reproduce.py --input wikisent2.txt --sample-size 500000 --batch-size 8 --out results_full

Outputs:
 sentence_probabilities.csv, figure3_analogue.png, figure6_analogue.png, comparison_table3.csv

IMPORTANT limitations:
- Paper does not provide the exact sampling seed or unambiguous bin edge rules; this implementation uses seed 123 and floor(-log10 p).
- The paper's Figure 6 wording concerns sample coverage, and its probabilistic interpretation is an assumption; this implementation uses cumulative sentence counts per bin.
- This code approximates Table 3, not an independently verified exact reproduction.
- The sentence probability uses GPT-2 token probabilities prefixed by <|endoftext|>.
- Out-of-context sentences are skipped and reported, not silently truncated.
