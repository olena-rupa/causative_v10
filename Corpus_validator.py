# -*- coding: utf-8 -*-
"""
Corpus_validator.py - Post-processing and Sketch Engine export utility.
"""
import os
import re
from html import escape as html_escape

import pandas as pd

FILES = [
    {
        "lang": "EN",
        "input": "corpus_EN_v10.xlsx",
        "gold": "GOLD_corpus_EN_thesis.xlsx",
        "sketch": "sketchengine_upload_EN.txt",
    },
    {
        "lang": "UK",
        "input": "corpus_UK_v10.xlsx",
        "gold": "GOLD_corpus_UK_thesis.xlsx",
        "sketch": "sketchengine_upload_UK.txt",
    },
]

RU_ALIEN = re.compile(r"[ыэъёЫЭЪЁ]")
CYRILLIC = re.compile(r"[А-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]")


def get_verb_set(lang: str):
    """Return sorted list of target verb lemmas for the specified language."""
    if lang == "EN":
        return sorted({
            "make", "have", "force", "cause", "get", "bring", "compel", "drive",
            "command", "press", "oblige", "ask", "encourage", "lead", "urge", "beg",
            "prompt", "advise", "persuade", "challenge", "dare", "implore",
            "influence", "recommend", "stimulate", "beseech", "plead",
        })
    return sorted({
        "примусити", "змусити", "казати", "веліти", "наказувати", "заставляти",
        "силувати", "приневолити", "попросити", "просити", "пропонувати",
        "запропонувати", "порекомендувати", "рекомендувати", "радити", "спонукати",
        "умовляти", "вмовляти", "заохочувати", "стимулювати", "благати",
        "переконувати", "підштовхувати", "підбурювати", "схиляти", "підбивати",
        "заклинати", "раяти", "нараяти",
    })


def clean_year(val) -> str:
    """Convert numeric or NaN year values to clean strings."""
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val)))
    except ValueError:
        return str(val).strip()


def clean_meta(val) -> str:
    """Escape XML special characters in metadata attributes."""
    if pd.isna(val):
        return ""
    return html_escape(str(val).strip())


def process_file(cfg: dict) -> None:
    lang = cfg["lang"]
    infile = cfg["input"]
    goldfile = cfg["gold"]
    sketchfile = cfg["sketch"]

    print(f"\n{'=' * 65}")
    print(f"  PROCESSING: {infile} ({lang})")
    print(f"{'=' * 65}")

    if not os.path.exists(infile):
        print(f"  [WARN] File not found: {infile} - skipping")
        return

    # Step 1: Load data and apply manual Keep filter
    print("\n  [1/5] Loading data & applying Keep filter...")
    df = pd.read_excel(infile, sheet_name="Corpus", engine="openpyxl")
    initial_count = len(df)
    print(f"        Corpus sheet: {initial_count} rows loaded")

    try:
        df_stats_old = pd.read_excel(infile, sheet_name="Statistics", engine="openpyxl")
        tws_values = df_stats_old["Total_Words_Scanned"].dropna()
        total_words_scanned = int(tws_values[tws_values > 0].max())
        print(f"        Total_Words_Scanned (preserved): {total_words_scanned:,}")
    except Exception as e:
        print(f"        [WARN] Could not read Total_Words_Scanned: {e}")
        total_words_scanned = initial_count * 5000
        print(f"        Estimated Total_Words_Scanned: {total_words_scanned:,}")

    if "Keep" in df.columns:
        keep_col = pd.to_numeric(df["Keep"], errors="coerce")
        false_positive_mask = keep_col == 0
        fp_dropped = int(false_positive_mask.sum())
        df = df[~false_positive_mask].copy()
        print(f"        False positives (Keep=0) removed: {fp_dropped}")
    else:
        fp_dropped = 0
        print("        No 'Keep' column found - all rows retained")

    print(f"        Rows after Keep filter: {len(df)}")

    # Step 2: Deduplication
    print("\n  [2/5] Deduplicating by Matched_Sentence...")
    df["Matched_Sentence"] = df["Matched_Sentence"].astype(str).str.strip()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Matched_Sentence"], keep="first")
    dupes_dropped = before_dedup - len(df)
    print(f"        Duplicates removed: {dupes_dropped}")
    print(f"        Rows after dedup:   {len(df)}")

    # Step 3: Language validation (alien character filter)
    print(f"\n  [3/5] Language validation ('{lang}' alien filter)...")
    if lang == "UK":
        mask = df["Matched_Sentence"].apply(lambda s: bool(RU_ALIEN.search(str(s))))
    elif lang == "EN":
        mask = df["Matched_Sentence"].apply(lambda s: bool(CYRILLIC.search(str(s))))
    else:
        mask = pd.Series([False] * len(df), index=df.index)

    aliens_found = int(mask.sum())
    if aliens_found > 0:
        alien_label = "Russian-alien" if lang == "UK" else "Cyrillic-alien"
        print(f"        {alien_label} sentences found: {aliens_found}")
        examples = df.loc[mask, "Matched_Sentence"].head(3)
        for i, ex in enumerate(examples, 1):
            print(f"          Example {i}: {str(ex)[:80]}...")
    df = df[~mask].copy()
    print(f"        Alien rows removed: {aliens_found}")
    print(f"        Rows after filter:  {len(df)}")

    # Step 4: Recalculate statistics and export verified Excel file
    print(f"\n  [4/5] Recalculating statistics & exporting {goldfile}...")
    verbs = get_verb_set(lang)
    stats_rows = []
    total_a, total_b = 0, 0

    for v in verbs:
        v_rows = df[df["Target_Verb_Lemma"] == v]
        af_a = len(v_rows[v_rows["Pattern"] == "A"]) if "Pattern" in df.columns else 0
        af_b = len(v_rows[v_rows["Pattern"] == "B"]) if "Pattern" in df.columns else 0
        af_total = af_a + af_b
        rf = round(af_total / total_words_scanned * 10000, 4) if total_words_scanned > 0 else 0
        total_a += af_a
        total_b += af_b
        stats_rows.append({
            "Verb_Lemma": v,
            "Pattern_A_Count": af_a,
            "Pattern_B_Count": af_b,
            "Total": af_total,
            "Total_Words_Scanned": total_words_scanned,
            "Relative_Freq_per_10k": rf,
        })

    df_stats = pd.DataFrame(stats_rows).sort_values("Total", ascending=False, ignore_index=True)

    print(f"        Pattern A: {total_a}")
    print(f"        Pattern B: {total_b}")
    print(f"        Combined:  {total_a + total_b}")
    verbs_found = len([r for r in stats_rows if r["Total"] > 0])
    print(f"        Verbs found: {verbs_found}/{len(verbs)}")

    sort_cols = [c for c in ("Pattern", "Target_Verb_Lemma") if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ignore_index=True)

    export_df = df.drop(columns=["Keep"], errors="ignore")

    with pd.ExcelWriter(goldfile, engine="openpyxl") as w:
        export_df.to_excel(w, sheet_name="Corpus", index=False)
        df_stats.to_excel(w, sheet_name="Statistics", index=False)
    print(f"        [OK] Saved {len(export_df)} rows + {len(df_stats)} stats -> {goldfile}")

    # Step 5: Export formatted corpus for Sketch Engine
    print(f"\n  [5/5] Generating Sketch Engine upload: {sketchfile}...")
    lines = []
    for _, row in df.iterrows():
        sentence = str(row.get("Matched_Sentence", "")).strip()
        sentence = re.sub(r"\[.*?\]", "", sentence)
        sentence = html_escape(" ".join(sentence.split()))

        if not sentence or sentence.lower() == "nan":
            continue

        year = clean_year(row.get("Year"))
        source = clean_meta(row.get("Media_Source"))
        verb = clean_meta(row.get("Target_Verb_Lemma"))
        url = clean_meta(row.get("Article_URL"))

        lines.append(
            f'<doc language="{lang}" year="{year}" source="{source}" '
            f'verb="{verb}" url="{url}">'
        )
        lines.append(sentence)
        lines.append("</doc>")

    with open(sketchfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    doc_count = lines.count("</doc>")
    file_size = os.path.getsize(sketchfile)
    print(f"        [OK] Exported {doc_count} documents -> {sketchfile} ({file_size / 1024:.1f} KB)")

    total_dropped = initial_count - len(df)
    print("\n  " + "-" * 48)
    print(f"  SUMMARY ({lang}):")
    print(f"    Initial rows:        {initial_count:>6d}")
    print(f"    Keep=0 dropped:      {fp_dropped:>6d}")
    print(f"    Duplicates dropped:  {dupes_dropped:>6d}")
    print(f"    Aliens dropped:      {aliens_found:>6d}")
    print(f"    TOTAL dropped:       {total_dropped:>6d}")
    print(f"    FINAL rows:          {len(df):>6d} (A: {total_a}, B: {total_b})")
    print(f"    Words scanned:   {total_words_scanned:>10,d}")
    print("  " + "-" * 48)


def main():
    print("=" * 65)
    print("CORPUS VALIDATOR - Post-Processing & Sketch Engine Export")
    print("=" * 65)

    for cfg in FILES:
        process_file(cfg)

    print(f"\n{'=' * 65}")
    print("  Validation and export completed.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
