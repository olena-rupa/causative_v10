# -*- coding: utf-8 -*-
"""
nlp_engine.py — NLP extraction engine for causative constructions (v10)
========================================================================
5-stage pipeline per article:
  Stage 1: spaCy DependencyMatcher → raw candidates
  Stage 2: Passive voice filter    → reject passive CTP
  Stage 3: LVC filter              → reject Light Verb Constructions
  Stage 4: Pattern B heuristics    → reject imperatives, quotes; recover conj subj
  Stage 5: Confidence scorer       → assign 0.0–1.0 score

All logic is deterministic, transparent, and reproducible.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import os
import re

from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Span, Token

from causative_v10.config import (
    CorpusConfig, W_PAT, W_VERB, W_STRUCT,
)

log = logging.getLogger("v10.nlp")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CausativeMatch:
    """A single extracted causative construction."""
    verb_lemma: str          # CTP verb lemma (e.g., "force", "змусити")
    sentence: str            # matched sentence text (whitespace-normalized)
    left_context: str        # preceding sentence text (or "")
    right_context: str       # following sentence text (or "")
    pattern: str             # "A" or "B"
    confidence: float        # 0.0 – 1.0
    conj_subject: str        # recovered subject via coordination ("" if none)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1: DEPENDENCY MATCHER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

# POS constraint for nominal arguments (subject, object)
_NP = {"IN": ["NOUN", "PRON", "PROPN"]}


def build_en_matcher(nlp) -> DependencyMatcher:
    """Build 6 English patterns: TO_A, TO_B, BARE_A, BARE_B, BARE_CCOMP_A, BARE_CCOMP_B.

    Pattern A (strict): requires nsubj -> full S1+P1+S2+P2 structure.
    Pattern B (relaxed): no nsubj node -> subject is contextually implicit.
    TO patterns: for verbs taking to-infinitive (force, cause, get...).
    BARE patterns: for make/have taking bare infinitive.

    NOTE: en_core_web_sm sometimes parses the infinitive complement as
    'advcl' (for to-inf) or 'ccomp' (for bare-inf). We include both
    dep labels to maximize recall without sacrificing precision, since
    the verb lemma list already constrains to causative verbs only.
    """
    from causative_v10.config import EN_TO, EN_BARE

    dm = DependencyMatcher(nlp.vocab)

    # Dep labels that en_core_web_sm may assign to the infinitive complement
    _TO_INF_DEPS = {"IN": ["xcomp", "advcl"]}
    _BARE_INF_DEPS = {"IN": ["xcomp", "ccomp", "oprd"]}

    # ── TO-infinitive, Pattern A (strict: with nsubj) ───────────────
    dm.add("EN_TO_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_TO}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _TO_INF_DEPS, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "to",
         "RIGHT_ATTRS": {"DEP": {"IN": ["mark", "aux"]}, "LOWER": "to"}},
    ]])

    # ── TO-infinitive, Pattern B (relaxed: no nsubj) ────────────────
    dm.add("EN_TO_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_TO}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _TO_INF_DEPS, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "to",
         "RIGHT_ATTRS": {"DEP": {"IN": ["mark", "aux"]}, "LOWER": "to"}},
    ]])

    # ── BARE infinitive, Pattern A (make/have, strict) ──────────────
    dm.add("EN_BARE_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": "VERB"}},
    ]])

    # ── BARE infinitive, Pattern B (make/have, relaxed) ─────────────
    dm.add("EN_BARE_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": "VERB"}},
    ]])

    # ── BARE infinitive alt parse: en_core_web_sm sometimes parses
    #    "made investors flee" as: made -> ccomp(flee -> nsubj(investors))
    #    In this case, S2 is a child of the INF, not of the CTP.
    #    We capture this with alternative patterns. ───────────────────
    dm.add("EN_BARE_ALT_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
    ]])

    dm.add("EN_BARE_ALT_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
    ]])

    return dm


def build_uk_matcher(nlp) -> DependencyMatcher:
    """Build 2 Ukrainian patterns: UK_A (strict), UK_B (relaxed).

    Ukrainian has no bare/to-infinitive distinction.
    Object deps include 'obl' to handle oblique case marking.
    """
    from causative_v10.config import UK_ALL
    dm = DependencyMatcher(nlp.vocab)

    # Pattern A: CTP → nsubj(S₁) + obj(S₂) + xcomp(Inf)
    dm.add("UK_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": list(UK_ALL)}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}},
    ]])

    # Pattern B: CTP → obj(S₂) + xcomp(Inf) — no nsubj required
    dm.add("UK_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": list(UK_ALL)}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB"}},
    ]])
    return dm


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2: PASSIVE VOICE FILTER (unchanged from v9)
# ═══════════════════════════════════════════════════════════════════════════

def _is_passive(tok: Token) -> bool:
    """Check if a verb token is in passive voice (triple-check).

    1. Morphological Voice=Pass feature
    2. auxpass / aux:pass dependent
    3. Token's own dep is nsubjpass / nsubj:pass / csubjpass
    """
    if tok.morph.to_dict().get("Voice") == "Pass":
        return True
    for c in tok.children:
        if c.dep_ in ("auxpass", "aux:pass"):
            return True
    return tok.dep_ in ("nsubjpass", "nsubj:pass", "csubjpass")


def _has_to(tok: Token) -> bool:
    """Check if an infinitive verb has a 'to' particle among its children."""
    for c in tok.children:
        if c.dep_ in ("mark", "aux") and c.text.lower() == "to":
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3: LIGHT VERB CONSTRUCTION FILTER
# ═══════════════════════════════════════════════════════════════════════════

def check_lvc(verb_lemma: str, obj_tok: Token, xcomp_tok: Token,
              config: CorpusConfig) -> Tuple[bool, str]:
    """Check if a match is a Light Verb Construction (false positive).

    Returns:
        (is_lvc: bool, reason: str)
        If is_lvc is True, the match should be rejected.
    """
    v = verb_lemma.lower()

    # Rule L1: Object-Head Blacklist
    if v in config.lvc_blacklist:
        if obj_tok.lemma_.lower() in config.lvc_blacklist[v]:
            return True, f"LVC_BLACKLIST:{v}+{obj_tok.lemma_.lower()}"

    # Rule L2: "make it + ADJ" (EN only)
    if v == "make" and obj_tok.lemma_.lower() == "it" and obj_tok.pos_ == "PRON":
        if xcomp_tok.pos_ == "ADJ":
            return True, "LVC_MAKE_IT_ADJ"

    # Rule L3: Numeral object (sports scores, quantities)
    if obj_tok.pos_ == "NUM":
        return True, "LVC_NUM_OBJ"
    # Also check if the object head is a numeral
    if obj_tok.head.pos_ == "NUM" and obj_tok.dep_ in ("appos", "nummod"):
        return True, "LVC_NUM_HEAD"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 4: PATTERN B HEURISTICS
# ═══════════════════════════════════════════════════════════════════════════

def _is_likely_imperative(ctp_tok: Token, sent: Span) -> bool:
    """Heuristic B1: Reject imperatives (sentence-initial verb, no subject).

    An imperative has no nsubj anywhere in the sentence and the CTP
    verb is at or near the sentence start (preceded only by punctuation,
    interjections, adverbs, or whitespace).
    """
    # If ANY token in the sentence carries nsubj, this is not imperative
    has_any_nsubj = any(t.dep_ == "nsubj" for t in sent)
    if has_any_nsubj:
        return False

    # Check if CTP is at/near sentence start
    preceding_content = [
        t for t in sent
        if t.i < ctp_tok.i
        and t.pos_ not in ("PUNCT", "INTJ", "ADV", "SPACE", "CCONJ")
    ]
    return len(preceding_content) == 0


def _is_inside_quotes(ctp_tok: Token, sent: Span) -> bool:
    """Heuristic B2: Reject matches inside direct speech quotation marks.

    Checks if the CTP verb appears inside an unclosed quote.
    Handles English (\", \\u201c, \\u201d) and Ukrainian («, ») marks.
    """
    text_before_ctp = sent.text[:ctp_tok.idx - sent.start_char]

    open_count = (
        text_before_ctp.count('"') +
        text_before_ctp.count('\u00ab') +    # «
        text_before_ctp.count('\u201c')       # "
    )
    close_count = (
        text_before_ctp.count('"') +
        text_before_ctp.count('\u00bb') +    # »
        text_before_ctp.count('\u201d')       # "
    )
    # If we've seen more opens than closes, CTP is inside a quote
    return open_count > close_count


def _has_conj_subject(ctp_tok: Token) -> Tuple[bool, str]:
    """Heuristic B3: Recover subject from coordinated verb.

    If the CTP is a conjunct (dep_='conj') of another verb that HAS
    nsubj, the CTP shares that subject. Returns (found, subject_text).
    """
    if ctp_tok.dep_ == "conj":
        head = ctp_tok.head
        for child in head.children:
            if child.dep_ == "nsubj":
                return True, child.text
    return False, ""


def _sent_too_short_for_b(sent: Span, min_tokens: int = 6) -> bool:
    """Heuristic B4: Reject Pattern B from very short sentences.

    Ultra-short sentences are almost always headlines, captions,
    or fragments that produce unreliable dependency parses.
    """
    content_tokens = sum(
        1 for t in sent
        if not t.is_punct and not t.is_space
    )
    return content_tokens < min_tokens


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 5: CONFIDENCE SCORER
# ═══════════════════════════════════════════════════════════════════════════

def compute_confidence(pattern: str, verb_lemma: str, sent: Span,
                       config: CorpusConfig,
                       has_conj_recovery: bool = False) -> float:
    """Compute a transparent confidence score for a causative match.

    Formula: c = W_PAT * c_pat + W_VERB * c_verb + W_STRUCT * c_struct

    c_pat:    A=1.0, B+conj=0.9, B=0.6
    c_verb:   specific=1.0, polysemous=0.7
    c_struct: optimal length=1.0, short/long=0.7, extreme=0.5
    """
    # Pattern signal
    if pattern == "A":
        c_pat = 1.0
    elif has_conj_recovery:
        c_pat = 0.9
    else:
        c_pat = 0.6

    # Verb specificity signal
    c_verb = 0.7 if verb_lemma.lower() in config.polysemous_verbs else 1.0

    # Structural quality signal (based on content token count)
    n_tokens = sum(1 for t in sent if not t.is_punct and not t.is_space)
    if 20 <= n_tokens <= 150:
        c_struct = 1.0
    elif 6 <= n_tokens < 20 or 150 < n_tokens <= 300:
        c_struct = 0.7
    else:
        c_struct = 0.5

    return round(W_PAT * c_pat + W_VERB * c_verb + W_STRUCT * c_struct, 3)


# ═══════════════════════════════════════════════════════════════════════════
#  REJECTION LOGGER
# ═══════════════════════════════════════════════════════════════════════════

def log_rejection(log_path: str, sentence: str, verb: str,
                  pattern: str, reason: str) -> None:
    """Append a rejected match to the rejection log for researcher audit."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            clean = " ".join(sentence.split())[:200]
            f.write(f"[{reason}] verb={verb} pat={pattern} | {clean}\n")
    except OSError:
        pass  # non-critical; don't crash the pipeline


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION — 5-stage pipeline
# ═══════════════════════════════════════════════════════════════════════════

def analyze(text: str, nlp, dm: DependencyMatcher,
            config: CorpusConfig) -> Tuple[List[CausativeMatch], int]:
    """Process article text through the 5-stage NLP pipeline.

    Returns:
        (matches: list of CausativeMatch, word_count: int)
    """
    # Safety: truncate very long texts to prevent memory issues
    if len(text) > 100_000:
        text = text[:100_000]

    try:
        doc = nlp(text)
    except Exception:
        return [], 0

    # Count words (alphabetic tokens only)
    wc = sum(1 for t in doc if t.is_alpha)
    sents = list(doc.sents)

    # ── Stage 1: Run DependencyMatcher ──────────────────────────────
    matches_raw = dm(doc)

    # Sort: Pattern A matches first, then Pattern B
    a_matches = [(mid, tids) for mid, tids in matches_raw
                 if doc.vocab.strings[mid].endswith("_A")]
    b_matches = [(mid, tids) for mid, tids in matches_raw
                 if doc.vocab.strings[mid].endswith("_B")]

    out: List[CausativeMatch] = []
    used_sents: set = set()

    for mid, tids in a_matches + b_matches:
        name = doc.vocab.strings[mid]
        is_a = name.endswith("_A")
        ctp = doc[tids[0]]

        # Determine token indices based on pattern structure.
        # Standard patterns:
        #   A: [ctp, s1, s2, inf, ...]   B: [ctp, s2, inf, ...]
        # ALT patterns (ccomp parse):
        #   ALT_A: [ctp, s1, inf, s2]    ALT_B: [ctp, inf, s2]
        is_alt = "ALT" in name
        if is_alt:
            if is_a:
                # ALT_A: tids = [ctp, s1, inf, s2]
                obj_tok = doc[tids[3]]
                inf_tok = doc[tids[2]]
            else:
                # ALT_B: tids = [ctp, inf, s2]
                obj_tok = doc[tids[2]]
                inf_tok = doc[tids[1]]
        else:
            obj_idx = 2 if is_a else 1
            inf_idx = 3 if is_a else 2
            obj_tok = doc[tids[obj_idx]]
            inf_tok = doc[tids[inf_idx]]

        # ── Stage 2: Passive voice filter ───────────────────────────
        if _is_passive(ctp):
            continue

        # Bare infinitive must NOT have 'to' (EN only)
        if "BARE" in name:
            if _has_to(inf_tok):
                continue

        # ── Stage 3: LVC filter ─────────────────────────────────────
        is_lvc, lvc_reason = check_lvc(ctp.lemma_.lower(), obj_tok,
                                        inf_tok, config)
        if is_lvc:
            # Find sentence for logging
            for s in sents:
                if s.start <= ctp.i < s.end:
                    log_rejection(config.rejected_log, s.text,
                                  ctp.lemma_.lower(), "A" if is_a else "B",
                                  lvc_reason)
                    break
            continue

        # ── Find the containing sentence ────────────────────────────
        si = None
        for idx, s in enumerate(sents):
            if s.start <= ctp.i < s.end:
                si = idx
                break
        if si is None or si in used_sents:
            continue

        sent = sents[si]
        st = sent.text.strip()

        # Sentence length constraints (min 15 chars, max 700 chars)
        if len(st) < 15 or len(st) > 700:
            continue

        # ── Stage 4: Pattern B heuristics ───────────────────────────
        conj_subj = ""
        pattern = "A" if is_a else "B"
        has_conj_recovery = False

        if not is_a:
            # B1: Imperative rejection
            if _is_likely_imperative(ctp, sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "IMPERATIVE")
                continue

            # B2: Quotation rejection
            if _is_inside_quotes(ctp, sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "INSIDE_QUOTES")
                continue

            # B3: Conjunct subject recovery (B→A promotion)
            found_conj, conj_text = _has_conj_subject(ctp)
            if found_conj:
                pattern = "A"
                conj_subj = conj_text
                has_conj_recovery = True

            # B4: Minimum sentence length for Pattern B
            if pattern == "B" and _sent_too_short_for_b(sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "TOO_SHORT")
                continue

        used_sents.add(si)

        # ── Stage 5: Confidence score ───────────────────────────────
        conf = compute_confidence(pattern, ctp.lemma_.lower(), sent,
                                  config, has_conj_recovery)

        # ── Build context ───────────────────────────────────────────
        left = sents[si - 1].text.strip() if si > 0 else ""
        right = sents[si + 1].text.strip() if si < len(sents) - 1 else ""

        out.append(CausativeMatch(
            verb_lemma=ctp.lemma_.lower(),
            sentence=st,
            left_context=left,
            right_context=right,
            pattern=pattern,
            confidence=conf,
            conj_subject=conj_subj,
        ))

    return out, wc
