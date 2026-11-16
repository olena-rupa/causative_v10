# -*- coding: utf-8 -*-
"""
run_uk.py — Entry point for Ukrainian causative corpus collection (v10)
========================================================================
Collects URLs from 18 Ukrainian news sources (2021-2026),
processes articles through the 5-stage NLP pipeline, and saves
the corpus to Excel + VRT.

INSTALL:
  pip install cloudscraper beautifulsoup4 lxml newspaper4k spacy pandas openpyxl
  python -m spacy download uk_core_news_sm

RUN:
  python run_uk.py
"""
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import spacy

from causative_v10.config import UK_CONFIG, SAVE_INTERVAL, SLEEP_RANGE, MIN_DATE
from causative_v10.nlp_engine import analyze, build_uk_matcher
from causative_v10.scraping import collect_urls, parse_date, scrape
from causative_v10.quota import Quota
from causative_v10.exporters import save_excel, save_vrt

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v10.UK")


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def process(urls, nlp, dm, quota, records, seen, tw):
    """Process URL list through the NLP pipeline.

    Args:
        urls: list of (url, date_str, source_name) tuples
        nlp: loaded spaCy model
        dm: built DependencyMatcher
        quota: Quota instance
        records: list to append accepted records to
        seen: set of already-seen sentences (deduplication)
        tw: total words scanned so far

    Returns:
        Updated total_words count.
    """
    scanned = 0
    next_save_target = SAVE_INTERVAL

    log.info(f"\n{'=' * 60}")
    log.info(f"  UK: {quota.summary()} | {len(urls)} URLs")
    log.info(f"{'=' * 60}")

    for url, ds, src in urls:
        if quota.done():
            break
        scanned += 1
        if scanned % 40 == 0:
            log.info(
                f"  [UK] {quota.summary()} | scanned:{scanned} | words:{tw:,}")

        text, pd_str = scrape(url)
        if not text:
            time.sleep(random.uniform(1, 2))
            continue

        fdate = pd_str or ds
        dt = parse_date(fdate)
        if dt and dt < MIN_DATE:
            continue

        matches, wc = analyze(text, nlp, dm, UK_CONFIG)
        tw += wc

        for m in matches:
            if quota.done():
                break
            pat = m.pattern
            if not quota.should_accept(pat):
                continue

            # Whitespace normalization for deduplication
            clean_sent = " ".join(m.sentence.split())
            if clean_sent in seen:
                continue
            seen.add(clean_sent)
            quota.accept(m.verb_lemma, pat)

            # Derive year from date
            year_str = fdate[:4] if fdate and len(fdate) >= 4 else ""

            records.append({
                "Language": "UK",
                "Year": year_str,
                "Date_Published": fdate,
                "Media_Source": src,
                "Article_URL": url,
                "Target_Verb_Lemma": m.verb_lemma,
                "Pattern": pat,
                "Confidence": m.confidence,
                "Left_Context": " ".join(m.left_context.split()),
                "Right_Context": " ".join(m.right_context.split()),
                "Matched_Sentence": clean_sent,
            })
            log.info(
                f"    [UK {quota.summary()}] "
                f"P{pat} c={m.confidence:.2f} '{m.verb_lemma}' "
                f"| {clean_sent[:55]}...")

        # Safe checkpoint: fire once when threshold is crossed, then bump
        if quota.total >= next_save_target:
            save_excel(records, tw, UK_CONFIG.checkpoint_file, UK_CONFIG)
            next_save_target = quota.total + SAVE_INTERVAL

        time.sleep(random.uniform(*SLEEP_RANGE))

    return tw


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = UK_CONFIG
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  CAUSATIVE EXTRACTION PIPELINE v10 — UKRAINIAN              ║
    ║  Target: {cfg.target_a} Pattern A  +  {cfg.target_b} Pattern B  =  {cfg.target_a + cfg.target_b} total     ║
    ║  Sources: {len(cfg.domains)} domains | CDX yearly chunks 2021–2026          ║
    ║  Output: {cfg.output_file:<52s} ║
    ║  NLP: 5-stage pipeline (Matcher→LVC→B-heuristics→Scorer)   ║
    ╚═══════════════════════════════════════════════════════════════╝""")

    t0 = datetime.now()
    records, seen = [], set()

    # ── Load spaCy ──────────────────────────────────────────────────
    log.info(f"Loading {cfg.spacy_model}...")
    try:
        nlp = spacy.load(cfg.spacy_model)
        log.info(f"  {cfg.spacy_model} loaded.")
    except OSError:
        log.error(
            f"Model missing! Run: python -m spacy download {cfg.spacy_model}")
        sys.exit(1)

    dm = build_uk_matcher(nlp)
    quota = Quota(cfg)

    # Clear rejection log for fresh run
    if os.path.exists(cfg.rejected_log):
        os.remove(cfg.rejected_log)

    # ── PHASE 1: URL collection ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 1: Collecting URLs (yearly CDX + sitemaps + Pravda)")
    log.info("=" * 60)
    urls = collect_urls(cfg)

    if len(urls) < 5000:
        log.warning(f"  URL pool is small ({len(urls)}). May not reach target.")

    # ── PHASE 2: NLP Processing ─────────────────────────────────────
    log.info("\nPHASE 2: Processing Ukrainian articles...")
    tw = process(urls, nlp, dm, quota, records, seen, 0)

    # ── PHASE 3: Final export ───────────────────────────────────────
    log.info(f"\nPHASE 3: Saving final results...")
    save_excel(records, tw, cfg.output_file, cfg)

    # VRT export for Sketch Engine
    vrt_path = cfg.output_file.replace(".xlsx", ".vrt")
    save_vrt(records, vrt_path, cfg)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = datetime.now() - t0
    print(f"\n{'=' * 60}")
    print(f"  RESULT: {quota.summary()}")
    print(f"  Words scanned: {tw:,}")
    print(f"  Output: {cfg.output_file} | Time: {elapsed}")
    print(f"\n  Per-verb breakdown (Pattern A / Pattern B):")
    for v in sorted(cfg.verb_all):
        ca = quota.verb_counts_a[v]
        cb = quota.verb_counts_b[v]
        if ca + cb > 0:
            print(f"    {v:15s}  A:{ca:4d}  B:{cb:3d}  Total:{ca + cb}")

    # Source distribution
    src_counts: dict = defaultdict(int)
    for r in records:
        src_counts[r["Media_Source"]] += 1
    print(f"\n  Source distribution:")
    for src in sorted(src_counts, key=src_counts.get, reverse=True):
        print(f"    {src:25s}  {src_counts[src]:5d}")

    # Rejection log stats
    if os.path.exists(cfg.rejected_log):
        with open(cfg.rejected_log, "r", encoding="utf-8") as f:
            rej_count = sum(1 for _ in f)
        print(f"\n  Rejected matches (see {cfg.rejected_log}): {rej_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
