# -*- coding: utf-8 -*-
"""
test_nlp.py — Test suite for causative extraction pipeline v10
================================================================
20 edge-case sentences proving the NLP engine works before
touching web scrapers. Tests cover:
  - True causatives (EN/UK, Pattern A/B, to-inf/bare-inf)
  - LVC false positives (make+choice, make-it-ADJ, have+look)
  - Imperatives, quotations, coordinated subjects
  - Short sentences, relative clauses, complex structures

RUN:
  python test_nlp.py
"""
import sys
import os

# Ensure the parent directory is on the path so causative_v10 is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spacy
from causative_v10.config import EN_CONFIG, UK_CONFIG
from causative_v10.nlp_engine import (
    analyze, build_en_matcher, build_uk_matcher, CausativeMatch,
)


# ═══════════════════════════════════════════════════════════════════════════
#  TEST CASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Each test case is: (id, language, sentence, expected_outcome, description)
#   expected_outcome: "MATCH" if we expect at least one CausativeMatch,
#                     "REJECT" if we expect zero matches.

EN_TEST_CASES = [
    # ── True causatives (should MATCH) ──────────────────────────────
    (1, "The government forced companies to lower prices.",
     "MATCH", "True causative, Pattern A, to-inf, verb=force"),

    (2, "The crisis made investors flee the market.",
     "MATCH", "True causative, Pattern A, bare-inf, verb=make"),

    (3, "The results encouraged researchers to continue the study.",
     "MATCH", "True causative, Pattern A, to-inf, verb=encourage"),

    (4, "The villagers beseeched the king to spare their lives.",
     "MATCH", "True causative, Pattern A, to-inf, verb=beseech"),

    (5, "Aggressive media coverage led the president to attack the press.",
     "MATCH", "True causative, Pattern A, to-inf, verb=lead"),

    # ── LVC false positives (should REJECT) ─────────────────────────
    (6, "They made rational choices based on their economic circumstances.",
     "REJECT", "LVC: make + choices (blacklisted object)"),

    (7, "Technology made it possible to work remotely during the pandemic.",
     "REJECT", "LVC: make it + ADJ (Rule L2)"),

    (8, "Let us have a look at the latest economic report from the ministry.",
     "REJECT", "LVC: have + look (blacklisted object)"),

    (9, "Students can get access to the university library online.",
     "REJECT", "LVC: get + access (blacklisted object)"),

    # ── Passive voice (should REJECT) ───────────────────────────────
    (10, "He was forced to resign by the board of directors.",
     "REJECT", "Passive causative: auxpass + nsubjpass"),

    # ── Pattern B heuristics ────────────────────────────────────────
    (11, "Force them to comply with the regulations immediately!",
     "REJECT", "Imperative: sentence-initial verb, no nsubj (B1)"),

    # ── Coordination recovery (B→A promotion via B3) ────────────────
    (12, "The minister arrived at the meeting and forced the board to resign.",
     "MATCH", "Conj subject recovery: forced shares nsubj with arrived"),

    # ── Non-causative verb (should REJECT) ──────────────────────────
    (13, "She decided to leave early after the conference ended.",
     "REJECT", "Non-causative verb: 'decide' not in verb list"),

    # ── get + participle (should REJECT) ────────────────────────────
    (14, "She got the car repaired at the local mechanic shop.",
     "REJECT", "get + past participle: not infinitive predication"),

    # ── Multi-clause complex (should MATCH) ─────────────────────────
    (15, "The report, which was leaked to the press, forced regulators to act swiftly.",
     "MATCH", "Complex sentence with relative clause, verb=force"),

    # ── Idiomatic numeral (should REJECT) ───────────────────────────
    (16, "England were well on course to make it 11 wins from 12 matches.",
     "REJECT", "Idiom: make it NUM (Rule L3 or non-matching parse)"),

    # ── Borderline polysemous verb (should MATCH if genuine) ────────
    (17, "The new evidence compelled the jury to acquit the defendant.",
     "MATCH", "True causative with unambiguous verb, verb=compel"),

    # ── Pattern B — genuine implicit subject ────────────────────────
    (18, "The situation was dire. This prompted officials to seek emergency funding.",
     "MATCH", "True Pattern B or A: 'This' may parse as nsubj, verb=prompt"),
]

UK_TEST_CASES = [
    # ── True Ukrainian causatives ───────────────────────────────────
    (19, "Уряд змусив компанії знизити ціни на основні продукти.",
     "MATCH", "True UK causative, Pattern A, verb=змусити"),

    (20, "Депутати переконували уряд відмінити рішення про підвищення тарифів.",
     "MATCH", "True UK causative, Pattern A, verb=переконувати"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_tests():
    """Load spaCy models, run all test cases, print results."""

    # ── Load English model ──────────────────────────────────────────
    print("Loading en_core_web_sm...")
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: en_core_web_sm not found. Run:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    dm_en = build_en_matcher(nlp_en)

    # ── Load Ukrainian model ────────────────────────────────────────
    print("Loading uk_core_news_sm...")
    try:
        nlp_uk = spacy.load("uk_core_news_sm")
    except OSError:
        print("WARNING: uk_core_news_sm not found. Skipping UK tests.")
        nlp_uk = None
        dm_uk = None
    if nlp_uk:
        dm_uk = build_uk_matcher(nlp_uk)

    # ── Suppress rejection log during tests ─────────────────────────
    # (use temp paths to avoid polluting the real logs)
    import tempfile
    en_cfg_test = EN_CONFIG
    uk_cfg_test = UK_CONFIG
    # We'll use the default rejected_log paths — they're harmless

    # ── Run tests ───────────────────────────────────────────────────
    print()
    print(f"{'#':>3} | {'EXPECTED':>8} | {'ACTUAL':>8} | {'VERB':>15} | "
          f"{'PAT':>3} | {'CONF':>5} | {'RESULT':>6} | DESCRIPTION")
    print("-" * 110)

    passed = 0
    failed = 0
    total = len(EN_TEST_CASES) + len(UK_TEST_CASES)

    # English tests
    for test_id, sentence, expected, desc in EN_TEST_CASES:
        matches, wc = analyze(sentence, nlp_en, dm_en, en_cfg_test)
        actual = "MATCH" if matches else "REJECT"
        ok = actual == expected

        if matches:
            m = matches[0]
            verb = m.verb_lemma
            pat = m.pattern
            conf = f"{m.confidence:.2f}"
        else:
            verb = "-"
            pat = "-"
            conf = "-"

        status = "  PASS" if ok else "* FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"{test_id:>3} | {expected:>8} | {actual:>8} | {verb:>15} | "
            f"{pat:>3} | {conf:>5} | {status:>6} | {desc}")

    # Ukrainian tests
    if nlp_uk:
        for test_id, sentence, expected, desc in UK_TEST_CASES:
            matches, wc = analyze(sentence, nlp_uk, dm_uk, uk_cfg_test)
            actual = "MATCH" if matches else "REJECT"
            ok = actual == expected

            if matches:
                m = matches[0]
                verb = m.verb_lemma
                pat = m.pattern
                conf = f"{m.confidence:.2f}"
            else:
                verb = "-"
                pat = "-"
                conf = "-"

            status = "  PASS" if ok else "* FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(
                f"{test_id:>3} | {expected:>8} | {actual:>8} | {verb:>15} | "
                f"{pat:>3} | {conf:>5} | {status:>6} | {desc}")
    else:
        skipped = len(UK_TEST_CASES)
        total -= skipped
        print(f"\n  (Skipped {skipped} UK tests — model not installed)")

    # ── Summary ─────────────────────────────────────────────────────
    print("-" * 110)
    print(f"\nRESULT: {passed}/{total} PASSED", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" -- ALL TESTS PASSED")

    # ── Cleanup test rejection log ──────────────────────────────────
    for log_path in [en_cfg_test.rejected_log, uk_cfg_test.rejected_log]:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                print(f"\n  Rejections logged during tests ({log_path}):")
                for line in lines[:10]:
                    print(f"    {line.rstrip()}")
                if len(lines) > 10:
                    print(f"    ... and {len(lines) - 10} more")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
