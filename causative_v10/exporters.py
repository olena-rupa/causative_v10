# -*- coding: utf-8 -*-
"""
exporters.py — Data export for causative corpus (v10)
======================================================
Exports to:
  1. Excel (.xlsx) with Corpus + Statistics sheets
  2. VRT (Vertical Text) with XML metadata for Sketch Engine
"""
import logging
import re
from collections import defaultdict
from html import escape as html_escape
from typing import Dict, List

import pandas as pd

from causative_v10.config import CorpusConfig

log = logging.getLogger("v10.export")


# ═══════════════════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_excel(records: List[dict], total_words: int,
               path: str, config: CorpusConfig) -> None:
    """Save corpus to Excel with Corpus + Statistics sheets.

    Corpus sheet columns:
        Language, Year, Date_Published, Media_Source, Article_URL,
        Target_Verb_Lemma, Pattern, Confidence, Left_Context,
        Matched_Sentence, Right_Context

    Statistics sheet columns:
        Verb_Lemma, Pattern_A_Count, Pattern_B_Count, Total,
        Total_Words_Scanned, Relative_Freq_per_10k
    """
    if not records:
        log.warning("No records to save.")
        return

    df = pd.DataFrame(records, columns=[
        "Language", "Year", "Date_Published", "Media_Source", "Article_URL",
        "Target_Verb_Lemma", "Pattern", "Confidence",
        "Left_Context", "Matched_Sentence", "Right_Context",
    ])

    # Sort: Pattern A first, then B; within each, alphabetical by verb
    df = df.sort_values(
        by=["Pattern", "Target_Verb_Lemma"],
        ignore_index=True,
    )

    # ── Statistics sheet ────────────────────────────────────────────
    stats_rows = []
    for v in sorted(config.verb_all):
        af_a = sum(
            1 for r in records
            if r["Target_Verb_Lemma"] == v and r["Pattern"] == "A"
        )
        af_b = sum(
            1 for r in records
            if r["Target_Verb_Lemma"] == v and r["Pattern"] == "B"
        )
        af_total = af_a + af_b
        rf = round(af_total / total_words * 10_000, 4) if total_words > 0 else 0
        stats_rows.append({
            "Verb_Lemma": v,
            "Pattern_A_Count": af_a,
            "Pattern_B_Count": af_b,
            "Total": af_total,
            "Total_Words_Scanned": total_words,
            "Relative_Freq_per_10k": rf,
        })

    df_stats = pd.DataFrame(stats_rows).sort_values("Total", ascending=False)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Corpus", index=False)
        df_stats.to_excel(writer, sheet_name="Statistics", index=False)

    log.info(f"  Saved {len(df)} records -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  VRT EXPORT (Sketch Engine)
# ═══════════════════════════════════════════════════════════════════════════

def _clean_for_xml(text: str) -> str:
    """Normalize whitespace and escape HTML/XML entities."""
    return html_escape(" ".join(text.split()))


def save_vrt(records: List[dict], path: str, config: CorpusConfig) -> None:
    """Export corpus as VRT (Vertical Text) for Sketch Engine.

    Format:
        <doc id="N" verb="..." pattern="..." confidence="..." source="..." url="...">
        One    word    per    line
        </doc>

    All metadata values are XML-escaped to prevent parsing errors.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(f'<corpus lang="{config.language}">\n')

        for i, rec in enumerate(records, 1):
            verb = html_escape(str(rec.get("Target_Verb_Lemma", "")))
            pat = html_escape(str(rec.get("Pattern", "")))
            conf = str(rec.get("Confidence", ""))
            src = html_escape(str(rec.get("Media_Source", "")))
            url = html_escape(str(rec.get("Article_URL", "")))
            year = html_escape(str(rec.get("Year", "")))

            f.write(
                f'<doc id="{i}" verb="{verb}" pattern="{pat}" '
                f'confidence="{conf}" source="{src}" url="{url}" '
                f'year="{year}">\n'
            )

            sent = str(rec.get("Matched_Sentence", "")).strip()
            f.write(f"{html_escape(sent)}\n")
            f.write("</doc>\n")

        f.write("</corpus>\n")

    log.info(f"  VRT saved: {len(records)} documents -> {path}")
