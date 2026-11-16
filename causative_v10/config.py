# -*- coding: utf-8 -*-
"""Configuration parameters, verb inventories, and source lists."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set
import re

TIMEOUT: int = 25
CDX_TIMEOUT: int = 90
CDX_RETRIES: int = 3
CDX_RETRY_SLEEP: int = 15
CDX_PER_YEAR: int = 1000
SITEMAP_MAX_PER_DOMAIN: int = 2000
SLEEP_RANGE = (3, 7)
SAVE_INTERVAL: int = 20
MIN_ARTICLE_WORDS: int = 300

MIN_DATE: datetime = datetime(2021, 1, 1)
CDX_YEARS: range = range(2021, 2027)

SKIP_URL = re.compile(
    r"/(rus|ru|eng|en|opinion|opinions|columns?|voices|advertorial"
    r"|promoted|blogs?|author|contributors|editorial|editorials|letters"
    r"|cartoon|cartoons|video|videos|gallery|podcast"
    r"|interactive|live|quiz"
    r"|sport|sports|football|basketball|tennis"
    r"|recipes|weather|horoscope)/", re.I
)

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36 Edg/124.0",
]

EN_BARE: List[str] = ["make", "have"]

EN_TO: List[str] = [
    "force", "cause", "get", "bring", "compel", "drive", "command",
    "press", "oblige", "ask", "encourage", "lead", "urge", "beg",
    "prompt", "advise", "persuade", "challenge", "dare", "implore",
    "influence", "recommend", "stimulate", "beseech", "plead",
]

EN_ALL: Set[str] = set(EN_BARE + EN_TO)

UK_ALL: Set[str] = {
    "примусити", "змусити", "казати", "веліти", "наказувати",
    "заставляти", "силувати", "приневолити",
    "попросити", "просити", "пропонувати", "запропонувати",
    "порекомендувати", "рекомендувати", "радити",
    "спонукати", "умовляти", "вмовляти",
    "заохочувати", "стимулювати", "благати",
    "переконувати", "підштовхувати", "підбурювати",
    "схиляти", "підбивати", "заклинати",
    "раяти", "нараяти",
}

EN_LVC_BLACKLIST: Dict[str, Set[str]] = {
    "make": {
        "decision", "choice", "move", "effort", "mistake", "point",
        "difference", "sense", "progress", "money", "change", "use",
        "room", "way", "deal", "attempt", "plan", "statement",
        "comment", "case", "argument", "suggestion", "claim",
        "announcement", "appearance", "contribution", "impression",
        "reference", "note", "commitment", "promise", "request",
        "demand", "offer", "bid", "call", "break", "turn", "shift",
    },
    "have": {
        "look", "effect", "impact", "say", "go", "time", "chance",
        "right", "ability", "opportunity", "conversation", "meeting",
        "discussion", "experience", "word",
    },
    "get": {
        "idea", "sense", "feeling", "impression", "picture",
        "grip", "hold", "access", "permission",
    },
}

UK_LVC_BLACKLIST: Dict[str, Set[str]] = {
    "просити": {
        "пробачення", "вибачення", "слово", "дозвіл", "допомога",
        "порада", "підтримка", "захист", "притулок", "милостиня",
        "пощада", "відстрочка", "аудієнція", "коментар",
    },
    "попросити": {
        "пробачення", "вибачення", "слово", "дозвіл", "допомога",
        "порада", "підтримка", "коментар",
    },
    "казати": {
        "слово", "промова", "правда", "неправда", "комплімент",
        "жарт", "тост",
    },
    "пропонувати": {
        "допомога", "послуга", "рішення", "варіант", "альтернатива",
        "угода",
    },
}

EN_POLYSEMOUS_VERBS: Set[str] = {"make", "have", "get"}
UK_POLYSEMOUS_VERBS: Set[str] = {"казати", "просити"}

W_PAT: float = 0.50
W_VERB: float = 0.30
W_STRUCT: float = 0.20

EN_DOMAINS: List[str] = [
    "reuters.com", "apnews.com", "theguardian.com",
    "washingtonpost.com", "bbc.com", "ft.com",
    "nytimes.com", "economist.com", "theatlantic.com", "politico.com",
]

EN_SRC_NAMES: Dict[str, str] = {
    "reuters.com": "Reuters", "apnews.com": "AP News",
    "theguardian.com": "The Guardian", "washingtonpost.com": "Washington Post",
    "bbc.com": "BBC News", "ft.com": "Financial Times",
    "nytimes.com": "NY Times", "economist.com": "The Economist",
    "theatlantic.com": "The Atlantic", "politico.com": "Politico",
}

EN_SITEMAP_ROOTS: Dict[str, List[str]] = {
    "reuters.com": [
        "https://www.reuters.com/sitemap_index.xml",
        "https://www.reuters.com/arc/outboundfeeds/sitemap-index/?outputType=xml",
    ],
    "apnews.com": [
        "https://apnews.com/sitemap.xml",
        "https://apnews.com/sitemap-index.xml",
    ],
    "theguardian.com": [
        "https://www.theguardian.com/sitemap.xml",
        "https://www.theguardian.com/sitemaps/news.xml",
    ],
    "washingtonpost.com": [
        "https://www.washingtonpost.com/sitemaps/sitemap-index.xml",
        "https://www.washingtonpost.com/sitemap_index.xml",
    ],
    "bbc.com": [
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
        "https://www.bbc.com/sitemap.xml",
    ],
    "ft.com": [
        "https://www.ft.com/sitemaps/sitemap-index.xml",
        "https://www.ft.com/sitemap.xml",
    ],
    "nytimes.com": [
        "https://www.nytimes.com/sitemap.xml",
        "https://www.nytimes.com/sitemaps/new/sitemap.xml.gz",
    ],
    "economist.com": [
        "https://www.economist.com/sitemap.xml",
    ],
    "theatlantic.com": [
        "https://www.theatlantic.com/sitemap.xml",
        "https://www.theatlantic.com/sitemaps/sitemap-index.xml",
    ],
    "politico.com": [
        "https://www.politico.com/sitemap.xml",
        "https://www.politico.com/sitemaps/sitemap-index.xml",
    ],
}

UK_DOMAINS: List[str] = [
    "texty.org.ua", "tyzhden.ua", "novynarnia.com", "frontliner.ua",
    "slovoidilo.ua", "hromadske.ua", "graty.me", "babel.ua",
    "nv.ua", "suspilne.media", "hromadske.radio", "rubryka.com",
    "espreso.tv", "lb.ua", "ukrinform.ua", "radiosvoboda.org",
    "pravda.com.ua", "zn.ua",
]

UK_SRC_NAMES: Dict[str, str] = {
    "texty.org.ua": "Texty", "tyzhden.ua": "Tyzhden",
    "novynarnia.com": "Novynarnia", "frontliner.ua": "Frontliner",
    "slovoidilo.ua": "Slovo i Dilo", "hromadske.ua": "Hromadske",
    "graty.me": "Graty", "babel.ua": "Babel",
    "nv.ua": "NV", "suspilne.media": "Suspilne",
    "hromadske.radio": "Hromadske Radio", "rubryka.com": "Rubryka",
    "espreso.tv": "Espreso", "lb.ua": "LB.ua",
    "ukrinform.ua": "Ukrinform", "radiosvoboda.org": "Radio Svoboda",
    "pravda.com.ua": "Ukrainska Pravda", "zn.ua": "ZN.UA",
}

UK_SITEMAP_ROOTS: Dict[str, List[str]] = {
    "texty.org.ua": ["https://texty.org.ua/sitemap.xml"],
    "tyzhden.ua": [
        "https://tyzhden.ua/sitemap.xml",
        "https://tyzhden.ua/sitemap_index.xml",
    ],
    "novynarnia.com": [
        "https://novynarnia.com/sitemap.xml",
        "https://novynarnia.com/sitemap_index.xml",
    ],
    "frontliner.ua": ["https://frontliner.ua/sitemap.xml"],
    "slovoidilo.ua": [
        "https://www.slovoidilo.ua/sitemap.xml",
        "https://slovoidilo.ua/sitemap.xml",
    ],
    "hromadske.ua": [
        "https://hromadske.ua/sitemap.xml",
        "https://hromadske.ua/sitemap-index.xml",
    ],
    "graty.me": ["https://graty.me/sitemap.xml"],
    "babel.ua": [
        "https://babel.ua/sitemap.xml",
        "https://babel.ua/sitemap_index.xml",
    ],
    "nv.ua": [
        "https://nv.ua/sitemap.xml",
        "https://nv.ua/sitemap-index.xml",
    ],
    "suspilne.media": [
        "https://suspilne.media/suspilne/sitemap/sitemap.xml",
        "https://suspilne.media/sitemap.xml",
    ],
    "hromadske.radio": ["https://hromadske.radio/sitemap.xml"],
    "rubryka.com": [
        "https://rubryka.com/sitemap.xml",
        "https://rubryka.com/sitemap_index.xml",
    ],
    "espreso.tv": ["https://espreso.tv/sitemap.xml"],
    "lb.ua": ["https://lb.ua/sitemap.xml"],
    "ukrinform.ua": [
        "https://www.ukrinform.ua/sitemap.xml",
        "https://ukrinform.ua/sitemap.xml",
    ],
    "radiosvoboda.org": [
        "https://www.radiosvoboda.org/sitemap.xml",
        "https://radiosvoboda.org/sitemap.xml",
    ],
    "pravda.com.ua": ["https://www.pravda.com.ua/sitemap.xml"],
    "zn.ua": ["https://zn.ua/sitemap.xml"],
}


@dataclass(frozen=True)
class CorpusConfig:
    language: str
    target_a: int
    target_b: int
    spacy_model: str
    verb_all: frozenset
    verb_bare: tuple
    verb_to: tuple
    domains: tuple
    src_names: dict
    sitemap_roots: dict
    lvc_blacklist: dict
    polysemous_verbs: frozenset
    output_file: str
    checkpoint_file: str
    rejected_log: str


EN_CONFIG = CorpusConfig(
    language="EN",
    target_a=2000,
    target_b=500,
    spacy_model="en_core_web_sm",
    verb_all=frozenset(EN_ALL),
    verb_bare=tuple(EN_BARE),
    verb_to=tuple(EN_TO),
    domains=tuple(EN_DOMAINS),
    src_names=EN_SRC_NAMES,
    sitemap_roots=EN_SITEMAP_ROOTS,
    lvc_blacklist=EN_LVC_BLACKLIST,
    polysemous_verbs=frozenset(EN_POLYSEMOUS_VERBS),
    output_file="corpus_EN_v10.xlsx",
    checkpoint_file="corpus_EN_v10_checkpoint.xlsx",
    rejected_log="rejected_EN_v10.log",
)

UK_CONFIG = CorpusConfig(
    language="UK",
    target_a=1000,
    target_b=500,
    spacy_model="uk_core_news_sm",
    verb_all=frozenset(UK_ALL),
    verb_bare=(),
    verb_to=(),
    domains=tuple(UK_DOMAINS),
    src_names=UK_SRC_NAMES,
    sitemap_roots=UK_SITEMAP_ROOTS,
    lvc_blacklist=UK_LVC_BLACKLIST,
    polysemous_verbs=frozenset(UK_POLYSEMOUS_VERBS),
    output_file="corpus_UK_v10.xlsx",
    checkpoint_file="corpus_UK_v10_checkpoint.xlsx",
    rejected_log="rejected_UK_v10.log",
)
