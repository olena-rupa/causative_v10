# -*- coding: utf-8 -*-
"""
nlp_engine.py - Syntactic extraction and filtering engine (causative_v10).
"""
from dataclasses import dataclass
from typing import List, Tuple
import logging

import spacy
from spacy.matcher import DependencyMatcher
from spacy.tokens import Span, Token

from causative_v10.config import (
    CorpusConfig, PATTERN_B_MIN_WORDS, W_PAT, W_VERB, W_STRUCT,
)

log = logging.getLogger("v10.nlp")


@dataclass
class CausativeMatch:
    """Extracted causative construction record."""
    verb_lemma: str
    sentence: str
    left_context: str
    right_context: str
    pattern: str
    confidence: float
    conj_subject: str


def load_nlp(model_name: str):
    """Load spaCy pipeline with ner and senter disabled."""
    return spacy.load(model_name, disable=["ner", "senter"])


# --- Stage 1: DependencyMatcher patterns ---

_NP = {"IN": ["NOUN", "PRON", "PROPN"]}


def build_en_matcher(nlp) -> DependencyMatcher:
    """Build 6 English dependency patterns (EN_TO_A/B, EN_BARE_A/B, EN_BARE_ALT_A/B)."""
    from causative_v10.config import EN_TO, EN_BARE

    dm = DependencyMatcher(nlp.vocab)
    _TO_INF_DEPS = {"IN": ["xcomp", "advcl"]}
    _BARE_INF_DEPS = {"IN": ["xcomp", "ccomp", "oprd"]}

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

    dm.add("EN_TO_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_TO}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _TO_INF_DEPS, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "to",
         "RIGHT_ATTRS": {"DEP": {"IN": ["mark", "aux"]}, "LOWER": "to"}},
    ]])

    dm.add("EN_BARE_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": "VERB"}},
    ]])

    dm.add("EN_BARE_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": EN_BARE}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": "VERB"}},
    ]])

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
    """Build 2 Ukrainian dependency patterns (UK_A, UK_B) with VerbForm=Inf constraint."""
    from causative_v10.config import UK_ALL

    dm = DependencyMatcher(nlp.vocab)

    dm.add("UK_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": list(UK_ALL)}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "ccomp", "advcl"]}, "POS": "VERB"}},
    ]])

    dm.add("UK_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": list(UK_ALL)}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["xcomp", "ccomp", "advcl"]}, "POS": "VERB"}},
    ]])

    return dm


# --- Stage 2: Passive voice and morphological checks ---

def _is_passive(tok: Token) -> bool:
    """Check whether the matrix predicate is in passive voice."""
    if tok.morph.to_dict().get("Voice") == "Pass":
        return True
    for c in tok.children:
        if c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass", "csubjpass"):
            return True
    if tok.text.lower().endswith(("но", "то")) and tok.pos_ == "VERB":
        if tok.morph.to_dict().get("VerbForm") != "Conv":
            return True
    return tok.dep_ in ("nsubjpass", "nsubj:pass", "csubjpass")


def _has_to(tok: Token) -> bool:
    """Check if an infinitive verb has a 'to' particle child."""
    for c in tok.children:
        if c.dep_ in ("mark", "aux") and c.text.lower() == "to":
            return True
    return False


def _is_uk_infinitive(tok: Token) -> bool:
    """Verify VerbForm=Inf feature for Ukrainian infinitive complements."""
    vf = tok.morph.to_dict().get("VerbForm")
    return vf in (None, "Inf")


# --- Stage 3: Light Verb Construction filter ---

def _is_lvc(verb_lemma: str, obj_tok: Token, xcomp_tok: Token,
            config: CorpusConfig) -> Tuple[bool, str]:
    """Three-level LVC filter (L1: stop-list, L2: make+it+ADJ, L3: numeral object)."""
    v = verb_lemma.lower()

    if v in config.lvc_blacklist:
        if obj_tok.lemma_.lower() in config.lvc_blacklist[v]:
            return True, f"LVC_BLACKLIST:{v}+{obj_tok.lemma_.lower()}"

    if v == "make" and obj_tok.lemma_.lower() == "it" and obj_tok.pos_ == "PRON":
        if xcomp_tok.pos_ == "ADJ":
            return True, "LVC_MAKE_IT_ADJ"

    if obj_tok.pos_ == "NUM":
        return True, "LVC_NUM_OBJ"
    if obj_tok.head.pos_ == "NUM" and obj_tok.dep_ in ("appos", "nummod"):
        return True, "LVC_NUM_HEAD"

    return False, ""


check_lvc = _is_lvc


# --- Stage 4: Pattern B validation heuristics (B1-B4) ---

def _is_likely_imperative(ctp_tok: Token, sent: Span) -> bool:
    """Heuristic B1: reject imperatives (Mood=Imp or sentence-initial verb without nsubj)."""
    if ctp_tok.morph.to_dict().get("Mood") == "Imp":
        return True
    has_any_nsubj = any(t.dep_ == "nsubj" for t in sent)
    if has_any_nsubj:
        return False

    preceding_content = [
        t for t in sent
        if t.i < ctp_tok.i
        and t.pos_ not in ("PUNCT", "INTJ", "ADV", "SPACE", "CCONJ")
    ]
    return len(preceding_content) == 0 and (ctp_tok.tag_ == "VB" or ctp_tok.pos_ == "VERB")


def _is_inside_quotes(ctp_tok: Token, sent: Span) -> bool:
    """Heuristic B2: reject matches inside unclosed quotation marks."""
    text_before_ctp = sent.text[:ctp_tok.idx - sent.start_char]

    open_count = (
        text_before_ctp.count('"') +
        text_before_ctp.count('\u00ab') +
        text_before_ctp.count('\u201c')
    )
    close_count = (
        text_before_ctp.count('"') +
        text_before_ctp.count('\u00bb') +
        text_before_ctp.count('\u201d')
    )
    return open_count > close_count


def _has_conj_subject(ctp_tok: Token) -> Tuple[bool, str]:
    """Heuristic B3: recover shared subject from coordinated head verb (dep='conj')."""
    if ctp_tok.dep_ == "conj":
        head = ctp_tok.head
        while head.dep_ == "conj" and head.head != head:
            head = head.head
        for child in head.children:
            if child.dep_ == "nsubj":
                return True, child.text
    return False, ""


def _sent_too_short_for_b(sent: Span,
                          min_tokens: int = PATTERN_B_MIN_WORDS) -> bool:
    """Heuristic B4: enforce minimum threshold of 8 content words for Pattern B."""
    content_tokens = sum(
        1 for t in sent
        if not t.is_punct and not t.is_space
    )
    return content_tokens < min_tokens


def _validate_pattern_b(ctp_tok: Token, sent: Span) -> Tuple[bool, str, str, bool]:
    """Execute B1-B4 heuristics; return (is_valid, reason, conj_subj, promoted)."""
    if _is_likely_imperative(ctp_tok, sent):
        return False, "IMPERATIVE", "", False

    if _is_inside_quotes(ctp_tok, sent):
        return False, "INSIDE_QUOTES", "", False

    found_conj, conj_text = _has_conj_subject(ctp_tok)
    if found_conj:
        return True, "", conj_text, True

    if _sent_too_short_for_b(sent, PATTERN_B_MIN_WORDS):
        return False, "TOO_SHORT", "", False

    return True, "", "", False


# --- Stage 5: Confidence scoring and logging ---

def _compute_confidence(pattern: str, verb_lemma: str, sent: Span,
                        config: CorpusConfig,
                        promoted: bool = False) -> float:
    """Compute weighted confidence score: c = 0.50*c_pat + 0.30*c_verb + 0.20*c_struct."""
    if pattern == "A" and not promoted:
        c_pat = 1.0
    elif promoted:
        c_pat = 0.9
    else:
        c_pat = 0.6

    c_verb = 0.7 if verb_lemma.lower() in config.polysemous_verbs else 1.0

    n_tokens = sum(1 for t in sent if not t.is_punct and not t.is_space)
    if 20 <= n_tokens <= 150:
        c_struct = 1.0
    elif 6 <= n_tokens < 20 or 150 < n_tokens <= 300:
        c_struct = 0.7
    else:
        c_struct = 0.5

    return round(W_PAT * c_pat + W_VERB * c_verb + W_STRUCT * c_struct, 3)


compute_confidence = _compute_confidence


def log_rejection(log_path: str, sentence: str, verb: str,
                  pattern: str, reason: str) -> None:
    """Write rejected false-positive match to audit log."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            clean = " ".join(sentence.split())[:200]
            f.write(f"[{reason}] verb={verb} pat={pattern} | {clean}\n")
    except OSError:
        pass


# --- Main analysis pipeline ---

def analyze(text: str, nlp, dm: DependencyMatcher,
            config: CorpusConfig) -> Tuple[List[CausativeMatch], int]:
    """Run the 5-stage extraction pipeline on article text."""
    if len(text) > 100_000:
        text = text[:100_000]

    try:
        doc = nlp(text)
    except Exception:
        return [], 0

    wc = sum(1 for t in doc if t.is_alpha)
    sents = list(doc.sents)

    matches_raw = dm(doc)
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

        is_alt = "ALT" in name
        if is_alt:
            if is_a:
                obj_tok = doc[tids[3]]
                inf_tok = doc[tids[2]]
            else:
                obj_tok = doc[tids[2]]
                inf_tok = doc[tids[1]]
        else:
            obj_idx = 2 if is_a else 1
            inf_idx = 3 if is_a else 2
            obj_tok = doc[tids[obj_idx]]
            inf_tok = doc[tids[inf_idx]]

        if _is_passive(ctp):
            continue

        if "BARE" in name and _has_to(inf_tok):
            continue

        if name.startswith("UK_") and not _is_uk_infinitive(inf_tok):
            continue

        is_lvc_match, lvc_reason = _is_lvc(
            ctp.lemma_.lower(), obj_tok, inf_tok, config
        )
        if is_lvc_match:
            for s in sents:
                if s.start <= ctp.i < s.end:
                    log_rejection(config.rejected_log, s.text,
                                  ctp.lemma_.lower(), "A" if is_a else "B",
                                  lvc_reason)
                    break
            continue

        si = None
        for idx, s in enumerate(sents):
            if s.start <= ctp.i < s.end:
                si = idx
                break
        if si is None or si in used_sents:
            continue

        sent = sents[si]
        st = sent.text.strip()

        if len(st) < 15 or len(st) > 700:
            continue

        conj_subj = ""
        pattern = "A" if is_a else "B"
        promoted = False

        if not is_a:
            valid_b, b_reason, conj_subj, promoted = _validate_pattern_b(ctp, sent)
            if not valid_b:
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", b_reason)
                continue
            if promoted:
                pattern = "A"

        used_sents.add(si)

        conf = _compute_confidence(
            pattern, ctp.lemma_.lower(), sent, config, promoted
        )

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
