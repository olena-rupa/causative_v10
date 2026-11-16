# -*- coding: utf-8 -*-
"""Syntactic dependency matching and post-filtering for causative constructions."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from spacy.matcher import DependencyMatcher
from spacy.tokens import Span, Token

from causative_v10.config import (
    CorpusConfig, EN_CONFIG, UK_CONFIG, W_PAT, W_VERB, W_STRUCT,
)

log = logging.getLogger("v10.nlp")


@dataclass
class CausativeMatch:
    verb_lemma: str
    sentence: str
    left_context: str
    right_context: str
    pattern: str
    confidence: float
    conj_subject: str


_NP = {"IN": ["NOUN", "PRON", "PROPN"]}
_OBJ_POS = {"IN": ["NOUN", "PRON", "PROPN", "NUM"]}


def build_en_matcher(nlp, config: Optional[CorpusConfig] = None) -> DependencyMatcher:
    """Compile English dependency graph patterns (EN_TO_A/B, EN_BARE_A/B, EN_BARE_ALT_A/B)."""
    cfg = config or EN_CONFIG
    verb_to = list(cfg.verb_to)
    verb_bare = list(cfg.verb_bare)

    dm = DependencyMatcher(nlp.vocab)
    _TO_INF_DEPS = {"IN": ["xcomp", "advcl"]}
    _BARE_INF_DEPS = {"IN": ["xcomp", "ccomp", "oprd"]}
    _BARE_COMP_POS = {"IN": ["VERB", "ADJ", "NOUN", "NUM"]}

    dm.add("EN_TO_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_to}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _TO_INF_DEPS, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "to",
         "RIGHT_ATTRS": {"DEP": {"IN": ["mark", "aux"]}, "LOWER": "to"}},
    ]])

    dm.add("EN_TO_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_to}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _TO_INF_DEPS, "POS": "VERB"}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "to",
         "RIGHT_ATTRS": {"DEP": {"IN": ["mark", "aux"]}, "LOWER": "to"}},
    ]])

    dm.add("EN_BARE_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_bare}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": _BARE_COMP_POS}},
    ]])

    dm.add("EN_BARE_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_bare}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": _BARE_INF_DEPS, "POS": _BARE_COMP_POS}},
    ]])

    dm.add("EN_BARE_ALT_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_bare}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}, "POS": _BARE_COMP_POS}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _OBJ_POS}},
    ]])

    dm.add("EN_BARE_ALT_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_bare}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": {"IN": ["ccomp", "xcomp"]}, "POS": _BARE_COMP_POS}},
        {"LEFT_ID": "inf", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _OBJ_POS}},
    ]])

    return dm


def build_uk_matcher(nlp, config: Optional[CorpusConfig] = None) -> DependencyMatcher:
    """Compile Ukrainian dependency patterns (UK_A, UK_B) with VerbForm=Inf check."""
    cfg = config or UK_CONFIG
    verb_all = list(cfg.verb_all)

    dm = DependencyMatcher(nlp.vocab)

    dm.add("UK_A", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_all}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s1",
         "RIGHT_ATTRS": {"DEP": "nsubj", "POS": _NP}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB",
                         "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}},
    ]])

    dm.add("UK_B", [[
        {"RIGHT_ID": "ctp", "RIGHT_ATTRS": {"POS": "VERB", "LEMMA": {"IN": verb_all}}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "s2",
         "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "obl", "dobj"]}, "POS": _OBJ_POS}},
        {"LEFT_ID": "ctp", "REL_OP": ">", "RIGHT_ID": "inf",
         "RIGHT_ATTRS": {"DEP": "xcomp", "POS": "VERB",
                         "MORPH": {"IS_SUPERSET": ["VerbForm=Inf"]}}},
    ]])

    return dm


def _is_passive(tok: Token) -> bool:
    """Reject passive matrix predicates via morphology or passive dependents."""
    if tok.morph.to_dict().get("Voice") == "Pass":
        return True
    for c in tok.children:
        if c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass", "csubjpass"):
            return True
    return False


def _has_to(tok: Token) -> bool:
    for c in tok.children:
        if c.dep_ in ("mark", "aux") and c.text.lower() == "to":
            return True
    return False


def _is_lvc(verb_lemma: str, obj_tok: Token, xcomp_tok: Token,
            config: CorpusConfig) -> Tuple[bool, str]:
    """Filter Light Verb Constructions and idiomatic patterns (L1, L2, L3)."""
    v = verb_lemma.lower()

    # L1: abstract object noun stop-list
    if v in config.lvc_blacklist:
        if obj_tok.lemma_.lower() in config.lvc_blacklist[v]:
            return True, f"LVC_BLACKLIST:{v}+{obj_tok.lemma_.lower()}"

    # L2: make + it + ADJ construction
    if v == "make" and obj_tok.lemma_.lower() == "it" and obj_tok.pos_ == "PRON":
        if xcomp_tok.pos_ == "ADJ":
            return True, "LVC_MAKE_IT_ADJ"

    # L3: numeral object or score idiom (e.g., "make it 11 wins from 12")
    if obj_tok.pos_ == "NUM" or xcomp_tok.pos_ == "NUM":
        return True, "LVC_NUM_OBJ"
    if (obj_tok.head.pos_ == "NUM" and obj_tok.dep_ in ("appos", "nummod")) or any(
        c.pos_ == "NUM" and c.dep_ == "nummod" for c in xcomp_tok.children
    ):
        return True, "LVC_NUM_HEAD"

    return False, ""


def _is_likely_imperative(ctp_tok: Token, sent: Span) -> bool:
    has_any_nsubj = any(t.dep_ == "nsubj" for t in sent)
    if has_any_nsubj:
        return False

    preceding_content = [
        t for t in sent
        if t.i < ctp_tok.i
        and t.pos_ not in ("PUNCT", "INTJ", "ADV", "SPACE", "CCONJ")
    ]
    return len(preceding_content) == 0


def _is_inside_quotes(ctp_tok: Token, sent: Span) -> bool:
    text_before_ctp = sent.text[:ctp_tok.idx - sent.start_char]

    straight_quotes = text_before_ctp.count('"')
    open_count = (
        text_before_ctp.count('\u00ab') +
        text_before_ctp.count('\u201c')
    )
    close_count = (
        text_before_ctp.count('\u00bb') +
        text_before_ctp.count('\u201d')
    )
    return open_count > close_count or (straight_quotes % 2 == 1)


def _has_conj_subject(ctp_tok: Token) -> Tuple[bool, str]:
    if ctp_tok.dep_ == "conj":
        head = ctp_tok.head
        for child in head.children:
            if child.dep_ == "nsubj":
                return True, child.text
    return False, ""


def _sent_too_short_for_b(sent: Span, min_tokens: int = 8) -> bool:
    non_punct_tokens = sum(
        1 for t in sent
        if not t.is_punct and not t.is_space
    )
    return non_punct_tokens < min_tokens


def compute_confidence(pattern: str, verb_lemma: str, sent: Span,
                       config: CorpusConfig,
                       promoted: bool = False) -> float:
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


def log_rejection(log_path: str, sentence: str, verb: str,
                  pattern: str, reason: str) -> None:
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            clean = " ".join(sentence.split())[:200]
            f.write(f"[{reason}] verb={verb} pat={pattern} | {clean}\n")
    except OSError:
        pass


def analyze(text: str, nlp, dm: DependencyMatcher,
            config: CorpusConfig) -> Tuple[List[CausativeMatch], int]:
    """Extract and validate causative constructions from raw article text."""
    if len(text) > 100_000:
        text = text[:100_000]

    try:
        doc = nlp(text)
    except Exception as e:
        log.debug(f"spaCy processing error: {e}")
        return [], 0

    wc = sum(1 for t in doc if t.is_alpha)
    sents = list(doc.sents)
    sent_idx_map = {s.start: idx for idx, s in enumerate(sents)}

    matches_raw = dm(doc)
    a_matches = [(mid, tids) for mid, tids in matches_raw
                 if doc.vocab.strings[mid].endswith("_A")]
    b_matches = [(mid, tids) for mid, tids in matches_raw
                 if doc.vocab.strings[mid].endswith("_B")]

    out: List[CausativeMatch] = []
    used_sents: set = set()
    checked_ctps: set = set()

    for mid, tids in a_matches + b_matches:
        name = doc.vocab.strings[mid]
        is_a = name.endswith("_A")
        ctp = doc[tids[0]]

        if ctp.i in checked_ctps:
            continue

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

        sent = ctp.sent
        si = sent_idx_map.get(sent.start)
        if si is None or si in used_sents:
            continue

        if _is_passive(ctp):
            checked_ctps.add(ctp.i)
            continue

        if "BARE" in name and _has_to(inf_tok):
            continue

        is_lvc_match, lvc_reason = _is_lvc(
            ctp.lemma_.lower(), obj_tok, inf_tok, config
        )
        if is_lvc_match:
            checked_ctps.add(ctp.i)
            log_rejection(config.rejected_log, sent.text,
                          ctp.lemma_.lower(), "A" if is_a else "B",
                          lvc_reason)
            continue

        if inf_tok.pos_ != "VERB" or obj_tok.pos_ == "NUM":
            continue

        checked_ctps.add(ctp.i)
        st = sent.text.strip()

        if len(st) < 15 or len(st) > 700:
            continue

        conj_subj = ""
        pattern = "A" if is_a else "B"
        promoted = False

        if not is_a:
            if _is_likely_imperative(ctp, sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "IMPERATIVE")
                continue

            if _is_inside_quotes(ctp, sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "INSIDE_QUOTES")
                continue

            found_conj, conj_text = _has_conj_subject(ctp)
            if found_conj:
                pattern = "A"
                conj_subj = conj_text
                promoted = True

            if pattern == "B" and _sent_too_short_for_b(sent):
                log_rejection(config.rejected_log, st,
                              ctp.lemma_.lower(), "B", "TOO_SHORT")
                continue

        used_sents.add(si)

        conf = compute_confidence(
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
