"""
TempoRAG-KG — Temporal Question EDA
รองรับทั้ง .parquet (HuggingFace) และ .json (official)
"""

import json, re, os
from collections import Counter, defaultdict

HOTPOTQA_PARQUET = "data/hotpot_dev.parquet"
HOTPOTQA_JSON    = "data/hotpot_dev_distractor_v1.json"
MUSIQUE_PATH     = "data/musique_ans_v1.0_dev.jsonl"

PATTERNS = {
    "year_mention":      r"\b(1[5-9]\d{2}|20\d{2})\b",
    "when_what_year":    r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bwhat decade\b|\bin what year\b",
    "before_after":      r"\bbefore\b|\bafter\b|\bsince\b|\buntil\b|\bprior to\b|\bfollowing\b",
    "time_period":       r"\bduring\b|\bin the \d{4}s\b|\bera\b|\bdecade\b|\bcentury\b|\bat the time\b",
    "first_last":        r"\bfirst\b|\blast\b|\bearli(er|est)\b|\blatest\b|\bmost recent\b|\boriginal\b",
    "age_duration":      r"\bhow (long|old)\b|\bage\b|\byears? (old|later|earlier|ago|apart|before|after)\b|\bduration\b",
    "temporal_relation": r"\bsame (year|time|period|decade)\b|\bcontemporary\b|\bpredecessor\b|\bsuccessor\b",
}

def is_temporal(question):
    q = question.lower()
    matched = [name for name, pat in PATTERNS.items() if re.search(pat, q, re.IGNORECASE)]
    return bool(matched), matched

def classify_difficulty(question):
    q = question.lower()
    if re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", q) and re.search(r"\bbefore\b|\bafter\b|\bprior\b", q):
        return "hard"
    if re.search(r"\bwhen\b|\bwhat year\b|\bwhich year\b", q):
        return "medium"
    return "easy"

def run_analysis(questions, answers, types=None):
    total = len(questions)
    pattern_counts = Counter()
    difficulty_counts = Counter()
    type_counts = Counter(types) if types else Counter()
    temporal_by_type = Counter()
    temporal_examples = []
    non_temporal_examples = []

    for i, q in enumerate(questions):
        is_t, matched = is_temporal(q)
        for m in matched:
            pattern_counts[m] += 1
        if is_t:
            diff = classify_difficulty(q)
            difficulty_counts[diff] += 1
            if types:
                temporal_by_type[types[i]] += 1
            if len(temporal_examples) < 20:
                temporal_examples.append({
                    "question": q,
                    "answer": answers[i],
                    "type": types[i] if types else "",
                    "patterns": matched,
                    "difficulty": diff
                })
        else:
            if len(non_temporal_examples) < 5:
                non_temporal_examples.append({"question": q, "answer": answers[i]})

    temporal_count = sum(1 for q in questions if is_temporal(q)[0])
    pct = temporal_count / total * 100

    print(f"Total questions:      {total:,}")
    print(f"Temporal questions:   {temporal_count:,} ({pct:.1f}%)")
    print(f"\nPattern breakdown:")
    for k, v in pattern_counts.most_common():
        print(f"  {k:30s}: {v:4d}  ({v/total*100:.1f}%)")

    if type_counts:
        print(f"\nQuestion type breakdown:")
        for t, c in type_counts.most_common():
            t_pct = temporal_by_type[t] / c * 100 if c > 0 else 0
            print(f"  {t:20s}: {c:4d} total | {temporal_by_type[t]:4d} temporal ({t_pct:.0f}%)")

    print(f"\nDifficulty distribution (temporal only):")
    for d in ["easy", "medium", "hard"]:
        c = difficulty_counts[d]
        pct_d = c / temporal_count * 100 if temporal_count > 0 else 0
        print(f"  {d:10s}: {c:4d}  ({pct_d:.1f}%)")

    print(f"\nSample temporal questions (10):")
    for i, ex in enumerate(temporal_examples[:10]):
        print(f"\n  [{i+1}] Q: {ex['question']}")
        print(f"       A: {ex['answer']}")
        print(f"       Pattern: {', '.join(ex['patterns'])}  |  Diff: {ex['difficulty']}")

    print(f"\nSample NON-temporal questions (5):")
    for i, ex in enumerate(non_temporal_examples[:5]):
        print(f"\n  [{i+1}] Q: {ex['question']}")
        print(f"       A: {ex['answer']}")

    return {
        "total": total,
        "temporal_count": temporal_count,
        "temporal_pct": round(pct, 1),
        "pattern_breakdown": dict(pattern_counts),
        "difficulty_breakdown": dict(difficulty_counts),
        "type_breakdown": dict(type_counts),
        "temporal_by_type": dict(temporal_by_type),
        "temporal_examples": temporal_examples,
    }

def analyze_hotpotqa():
    print(f"\n{'='*60}")
    print("HOTPOTQA ANALYSIS")
    print(f"{'='*60}")

    if os.path.exists(HOTPOTQA_PARQUET):
        print(f"Loading from parquet: {HOTPOTQA_PARQUET}")
        try:
            import pandas as pd
            df = pd.read_parquet(HOTPOTQA_PARQUET)
            print(f"Columns found: {list(df.columns)}")

            q_col = next((c for c in df.columns if 'question' in c.lower()), None)
            a_col = next((c for c in df.columns if 'answer' in c.lower()), None)
            t_col = next((c for c in df.columns if 'type' in c.lower()), None)

            if not q_col:
                print("Cannot find question column. Columns:", list(df.columns))
                return {}

            questions = df[q_col].astype(str).tolist()
            answers   = df[a_col].astype(str).tolist() if a_col else [""] * len(questions)
            types     = df[t_col].astype(str).tolist() if t_col else None

            result = run_analysis(questions, answers, types)
            result["dataset"] = "HotpotQA"
            return result

        except ImportError:
            print("Missing pandas — run: pip install pandas pyarrow")
            return {}
        except Exception as e:
            print(f"Error reading parquet: {e}")
            return {}

    elif os.path.exists(HOTPOTQA_JSON):
        print(f"Loading from JSON: {HOTPOTQA_JSON}")
        with open(HOTPOTQA_JSON) as f:
            data = json.load(f)
        questions = [item["question"] for item in data]
        answers   = [item.get("answer", "") for item in data]
        types     = [item.get("type", "unknown") for item in data]
        result = run_analysis(questions, answers, types)
        result["dataset"] = "HotpotQA"
        return result

    else:
        print("No HotpotQA file found.")
        print(f"  Need: {HOTPOTQA_PARQUET}  OR  {HOTPOTQA_JSON}")
        return {}

def analyze_musique():
    if not os.path.exists(MUSIQUE_PATH):
        print(f"\nMuSiQue not found at: {MUSIQUE_PATH}")
        return {}

    print(f"\n{'='*60}")
    print("MUSIQUE ANALYSIS")
    print(f"{'='*60}")

    data = []
    with open(MUSIQUE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    total = len(data)
    pattern_counts = Counter()
    hop_counts = Counter()
    temporal_by_hop = defaultdict(int)
    temporal_examples = []

    for item in data:
        q = item.get("question", "")
        is_t, matched = is_temporal(q)
        n_hops = len(item.get("question_decomposition", []))
        hop_counts[n_hops] += 1
        for m in matched:
            pattern_counts[m] += 1
        if is_t:
            temporal_by_hop[n_hops] += 1
            if len(temporal_examples) < 20:
                temporal_examples.append({
                    "question": q,
                    "answer": item.get("answer", ""),
                    "n_hops": n_hops,
                    "patterns": matched
                })

    temporal_count = sum(1 for item in data if is_temporal(item.get("question",""))[0])
    pct = temporal_count / total * 100

    print(f"Total questions:      {total:,}")
    print(f"Temporal questions:   {temporal_count:,} ({pct:.1f}%)")
    print(f"\nHop distribution:")
    for h in sorted(hop_counts.keys()):
        c = hop_counts[h]
        tc = temporal_by_hop[h]
        print(f"  {h}-hop: {c:4d} total | {tc:4d} temporal ({tc/c*100:.1f}%)")
    print(f"\nPattern breakdown:")
    for k, v in pattern_counts.most_common():
        print(f"  {k:30s}: {v:4d}  ({v/total*100:.1f}%)")
    print(f"\nSample temporal questions (8):")
    for i, ex in enumerate(temporal_examples[:8]):
        print(f"\n  [{i+1}] ({ex['n_hops']}-hop) Q: {ex['question']}")
        print(f"           A: {ex['answer']}")
        print(f"           Pattern: {', '.join(ex['patterns'])}")

    return {
        "dataset": "MuSiQue",
        "total": total,
        "temporal_count": temporal_count,
        "temporal_pct": round(pct, 1),
        "pattern_breakdown": dict(pattern_counts),
        "hop_distribution": dict(hop_counts),
        "temporal_by_hop": dict(temporal_by_hop),
        "temporal_examples": temporal_examples,
    }

def print_verdict(hp, ms):
    print(f"\n{'='*60}")
    print("VERDICT — WHAT THIS MEANS FOR TEMPORAG-KG")
    print(f"{'='*60}")

    hp_pct   = hp.get("temporal_pct", 0)
    hp_count = hp.get("temporal_count", 0)
    hp_total = hp.get("total", 0)
    ms_pct   = ms.get("temporal_pct", 0)
    ms_count = ms.get("temporal_count", 0)
    ms_total = ms.get("total", 0)

    if hp_pct >= 25:
        print(f"\n✅  STRONG — {hp_pct}% of HotpotQA is temporal")
        print( "    'Temporal filtering addresses a significant failure mode")
        print( "     in KG2RAG, affecting 1 in 4 benchmark questions.'")
    elif hp_pct >= 15:
        print(f"\n🟡  MODERATE — {hp_pct}% of HotpotQA is temporal")
        print( "    'Temporal questions form a meaningful subset.")
        print( "     Focus ablation on temporal subset F1 separately.'")
    else:
        print(f"\n🔴  WEAK — only {hp_pct}% of HotpotQA is temporal")
        print( "    Consider adding TimeQuestions or MultiTQ as primary temporal benchmark.")

    print(f"\nKey numbers for proposal:")
    if hp_total:
        print(f"  HotpotQA : {hp_pct}% temporal  ({hp_count:,} / {hp_total:,})")
    if ms_total:
        print(f"  MuSiQue  : {ms_pct}% temporal  ({ms_count:,} / {ms_total:,})")

    if hp_pct >= 15 and hp_total:
        print(f'\nSuggested sentence for proposal:')
        print(f'  "Our EDA reveals that {hp_pct}% of HotpotQA dev questions contain')
        print(f'   temporal expressions ({hp_count:,} / {hp_total:,}), representing cases')
        print(f'   where temporal validity filtering directly impacts answer correctness."')

if __name__ == "__main__":
    print("\nTempoRAG-KG — Temporal EDA")
    print("Supanut Kompayak (st126055) | AIT NLP 2026")

    results = {}

    hp = analyze_hotpotqa()
    if hp:
        results["hotpotqa"] = hp

    ms = analyze_musique()
    if ms:
        results["musique"] = ms

    if results:
        print_verdict(results.get("hotpotqa", {}), results.get("musique", {}))
        with open("results/temporal_eda_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to results/temporal_eda_results.json")
    else:
        print("\nNo datasets found.")
        print("  Need: hotpot_dev.parquet  (from HuggingFace)")
        print("  OR:   hotpot_dev_distractor_v1.json  (from CMU)")