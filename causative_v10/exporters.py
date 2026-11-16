# -*- coding: utf-8 -*-
"""Excel (.xlsx) and Sketch Engine XML/VRT exporters."""

import logging
from html import escape as html_escape
from typing import List

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from causative_v10.config import CorpusConfig

log = logging.getLogger("v10.export")


def _sanitize_cell(val):
    if isinstance(val, str):
        return ILLEGAL_CHARACTERS_RE.sub("", val)
    return val


def save_excel(records: List[dict], total_words: int,
               path: str, config: CorpusConfig) -> None:
    """Export extracted contexts and per-lemma statistics to Excel."""
    if not records:
        log.warning("No records to save.")
        return

    df = pd.DataFrame(records, columns=[
        "Language", "Year", "Date_Published", "Media_Source", "Article_URL",
        "Target_Verb_Lemma", "Pattern", "Confidence",
        "Left_Context", "Matched_Sentence", "Right_Context",
    ])

    for col in ("Left_Context", "Matched_Sentence", "Right_Context", "Article_URL"):
        df[col] = df[col].apply(_sanitize_cell)

    df = df.sort_values(
        by=["Pattern", "Target_Verb_Lemma"],
        ignore_index=True,
    )

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


def save_vrt(records: List[dict], path: str, config: CorpusConfig) -> None:
    """Export corpus in XML-annotated document format for Sketch Engine."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(f'<corpus lang="{config.language}">\n')

        for i, rec in enumerate(records, 1):
            verb = html_escape(str(rec.get("Target_Verb_Lemma", "")), quote=True)
            pat = html_escape(str(rec.get("Pattern", "")), quote=True)
            conf = str(rec.get("Confidence", ""))
            src = html_escape(str(rec.get("Media_Source", "")), quote=True)
            url = html_escape(str(rec.get("Article_URL", "")), quote=True)
            year = html_escape(str(rec.get("Year", "")), quote=True)

            f.write(
                f'<doc id="{i}" verb="{verb}" pattern="{pat}" '
                f'confidence="{conf}" source="{src}" url="{url}" '
                f'year="{year}">\n'
            )

            sent = str(rec.get("Matched_Sentence", "")).strip()
            f.write(f"{html_escape(sent, quote=False)}\n")
            f.write("</doc>\n")

        f.write("</corpus>\n")

    log.info(f"  VRT saved: {len(records)} documents -> {path}")
