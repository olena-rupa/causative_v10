# -*- coding: utf-8 -*-
"""
scraping.py — URL collection and article extraction (v10)
==========================================================
Proven infrastructure from v9, refactored into standalone module.
Three URL collection strategies:
  1. Yearly CDX chunking (Wayback Machine, 2021-2026)
  2. Deep sitemap crawling (recursive XML parsing)
  3. Pravda daily archives (UK only, pravda.com.ua)
All strategies feed into round-robin mixing for source balance.
"""
import logging
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

import cloudscraper
from bs4 import BeautifulSoup

from causative_v10.config import (
    CDX_PER_YEAR, CDX_RETRIES, CDX_RETRY_SLEEP, CDX_TIMEOUT,
    CDX_YEARS, CorpusConfig, MIN_ARTICLE_WORDS, MIN_DATE,
    SITEMAP_MAX_PER_DOMAIN, SKIP_URL, TIMEOUT, USER_AGENTS,
)

try:
    from newspaper import Article as NpArticle
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

log = logging.getLogger("v10.scraping")

# Global cloudscraper session (reused across all requests)
_SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows"})


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP REQUEST WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def cget(url: str, retries: int = 3,
         timeout: int = TIMEOUT) -> Optional[object]:
    """HTTP GET with cloudscraper, retries, and random User-Agent.

    Returns the Response object on success, or None on failure.
    """
    for att in range(1, retries + 1):
        try:
            _SCRAPER.headers.update({"User-Agent": random.choice(USER_AGENTS)})
            r = _SCRAPER.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            log.debug(f"GET fail ({att}/{retries}) {url[:65]}: {e}")
            if att < retries:
                time.sleep(random.uniform(2, 5))
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  DATE PARSING
# ═══════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
    "%Y%m%d%H%M%S", "%d.%m.%Y", "%d/%m/%Y",
)


def parse_date(s: str) -> Optional[datetime]:
    """Attempt to parse a date string using multiple formats."""
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s[:30].strip(), fmt).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  ARTICLE SCRAPING
# ═══════════════════════════════════════════════════════════════════════════

def scrape(url: str) -> Tuple[Optional[str], str]:
    """Extract article text and publication date from a URL.

    Uses newspaper4k as primary extractor, falls back to BeautifulSoup.
    Rejects articles with fewer than MIN_ARTICLE_WORDS words.

    Returns:
        (text, date_string) or (None, "") on failure.
    """
    # Strategy 1: newspaper4k
    if HAS_NEWSPAPER:
        try:
            a = NpArticle(url)
            a.download()
            a.parse()
            if a.text and len(a.text.split()) >= MIN_ARTICLE_WORDS:
                ds = str(a.publish_date) if a.publish_date else ""
                return a.text, ds
        except Exception:
            pass

    # Strategy 2: BeautifulSoup fallback
    try:
        r = cget(url)
        if not r:
            return None, ""

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "figure", "figcaption", "noscript",
                         "iframe", "form"]):
            tag.decompose()

        # Find article body
        body = (
            soup.find("article") or
            soup.find("div", class_=re.compile(
                r"article|story|content|post-body", re.I)) or
            soup.find("main") or
            soup
        )

        text = "\n".join(
            p.get_text(strip=True)
            for p in body.find_all("p")
            if len(p.get_text(strip=True)) > 25
        )

        # Extract date
        ds = ""
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            ds = time_tag["datetime"]

        if text and len(text.split()) >= MIN_ARTICLE_WORDS:
            return text, ds
    except Exception:
        pass

    return None, ""


# ═══════════════════════════════════════════════════════════════════════════
#  URL COLLECTION: YEARLY CDX CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def wayback_cdx_yearly(domain: str, src_names: Dict[str, str],
                       max_per_year: int = CDX_PER_YEAR
                       ) -> List[Tuple[str, str, str]]:
    """Query Wayback CDX API year-by-year (2021–2026).

    Returns list of (url, date_string, source_name) tuples.
    Uses 90-second timeout with 3 retries per year chunk.
    """
    results: List[Tuple[str, str, str]] = []
    seen: set = set()
    src = src_names.get(domain, domain)

    for year in CDX_YEARS:
        cdx_url = (
            f"https://web.archive.org/cdx/search/cdx"
            f"?url={quote(domain, safe='')}/*"
            f"&output=json&fl=timestamp,original,statuscode,mimetype"
            f"&from={year}0101&to={year}1231"
            f"&filter=statuscode:200&filter=mimetype:text/html"
            f"&collapse=urlkey&limit={max_per_year * 2}"
        )

        data = None
        for attempt in range(1, CDX_RETRIES + 1):
            log.info(f"  CDX [{attempt}/{CDX_RETRIES}] {domain} year={year}...")
            try:
                _SCRAPER.headers.update(
                    {"User-Agent": random.choice(USER_AGENTS)})
                r = _SCRAPER.get(cdx_url, timeout=CDX_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    log.info(
                        f"    CDX OK {domain}/{year}: {len(data)-1} raw rows")
                    break
                else:
                    log.warning(
                        f"    CDX HTTP {r.status_code} for {domain}/{year}")
            except Exception as e:
                log.warning(
                    f"    CDX FAIL {domain}/{year} (att {attempt}): {e}")
            if attempt < CDX_RETRIES:
                time.sleep(CDX_RETRY_SLEEP)

        if data is None:
            log.warning(f"    CDX exhausted for {domain}/{year}")
            continue

        year_count = 0
        for row in data[1:]:
            if len(row) < 2:
                continue
            ts, orig = row[0], row[1]
            if SKIP_URL.search(orig):
                continue
            if orig in seen:
                continue
            seen.add(orig)
            date_str = (
                f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
            )
            results.append((orig, date_str, src))
            year_count += 1
            if year_count >= max_per_year:
                break

        log.info(f"    {domain}/{year}: kept {year_count} URLs")
        time.sleep(random.uniform(2, 4))

    log.info(f"  CDX TOTAL for {domain}: {len(results)} URLs across all years")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  URL COLLECTION: SITEMAP CRAWLING
# ═══════════════════════════════════════════════════════════════════════════

def crawl_sitemap(url: str,
                  max_urls: int = SITEMAP_MAX_PER_DOMAIN,
                  depth: int = 0) -> List[Tuple[str, str]]:
    """Recursively crawl XML sitemaps.

    Returns list of (url, date_string) tuples.
    Limits recursion depth to 4 and applies SKIP_URL and MIN_DATE filters.
    """
    if depth > 4:
        return []

    results: List[Tuple[str, str]] = []
    r = cget(url, timeout=40)
    if not r:
        return results

    soup = BeautifulSoup(r.content, "xml")

    # Process nested sitemaps
    for sm in soup.find_all("sitemap"):
        loc = sm.find("loc")
        if not loc or not loc.string:
            continue
        child_url = loc.string.strip()
        yr_match = re.search(r"(20\d{2})", child_url)
        if yr_match and int(yr_match.group(1)) < 2021:
            continue
        sub = crawl_sitemap(child_url, max_urls - len(results), depth + 1)
        results.extend(sub)
        if len(results) >= max_urls:
            break
        time.sleep(random.uniform(0.8, 2))

    # Process URL entries
    for ut in soup.find_all("url"):
        loc = ut.find("loc")
        if not loc or not loc.string:
            continue
        u = loc.string.strip()
        if SKIP_URL.search(u):
            continue
        date_str = ""
        lm = ut.find("lastmod")
        if lm and lm.string:
            date_str = lm.string.strip()
        dt = parse_date(date_str)
        if dt and dt < MIN_DATE:
            continue
        yr2 = re.search(r"/(20\d{2})/", u)
        if yr2 and int(yr2.group(1)) < 2021:
            continue
        results.append((u, date_str))
        if len(results) >= max_urls:
            break

    if depth == 0:
        log.info(f"  Sitemap: {len(results)} URLs from {url[:55]}")
    return results[:max_urls]


# ═══════════════════════════════════════════════════════════════════════════
#  URL COLLECTION: PRAVDA ARCHIVES (UK only)
# ═══════════════════════════════════════════════════════════════════════════

def pravda_archives(days: int = 300,
                    max_urls: int = 3000
                    ) -> List[Tuple[str, str, str]]:
    """Scrape Ukrainska Pravda daily archive pages.

    Returns list of (url, date_string, "Ukrainska Pravda") tuples.
    Only used for the UK pipeline (pravda.com.ua).
    """
    results: List[Tuple[str, str, str]] = []
    seen: set = set()
    base = "https://www.pravda.com.ua"

    for i in range(days):
        d = datetime.now() - timedelta(days=i)
        if d < MIN_DATE:
            break
        page_url = f"{base}/archives/date_{d.strftime('%d%m%Y')}/"
        r = cget(page_url)
        if not r:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            h = a_tag["href"]
            if "/news/" in h or "/articles/" in h:
                fu = urljoin(base, h)
                if fu not in seen and not SKIP_URL.search(fu):
                    seen.add(fu)
                    results.append(
                        (fu, d.strftime("%Y-%m-%d"), "Ukrainska Pravda"))

        time.sleep(random.uniform(1.5, 3))
        if len(results) >= max_urls:
            break

    log.info(f"  Pravda archives: {len(results)} URLs")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  ROUND-ROBIN MIXING
# ═══════════════════════════════════════════════════════════════════════════

def round_robin_mix(domain_buckets: Dict[str, list]) -> list:
    """Interleave URLs from all domains in round-robin order.

    Pops one URL from each domain in turn, guaranteeing balanced
    source distribution in the processing order.
    """
    queues = {
        dom: deque(urls)
        for dom, urls in domain_buckets.items()
        if urls
    }
    mixed: list = []
    while queues:
        empty = []
        for dom in list(queues.keys()):
            if queues[dom]:
                mixed.append(queues[dom].popleft())
            else:
                empty.append(dom)
        for dom in empty:
            del queues[dom]
    return mixed


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN URL COLLECTION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def collect_urls(config: CorpusConfig) -> List[Tuple[str, str, str]]:
    """Collect all URLs for a corpus using CDX + Sitemaps + archives.

    Pipeline:
      1. For each domain: query CDX yearly → get URLs
      2. If CDX yields < 100: double sitemap budget as fallback
      3. Crawl sitemaps for each domain
      4. (UK only) Scrape Pravda daily archives
      5. Shuffle each domain's bucket
      6. Round-robin mix all domains

    Returns list of (url, date_string, source_name) tuples.
    """
    domain_buckets: Dict[str, list] = {}

    for dom in config.domains:
        src = config.src_names.get(dom, dom)
        seen: set = set()
        bucket: list = []

        # Phase A: Yearly CDX
        for u, d, s in wayback_cdx_yearly(dom, config.src_names):
            if u not in seen:
                seen.add(u)
                bucket.append((u, d, s))
        cdx_got = len(bucket)
        log.info(f"  {dom}: CDX yielded {cdx_got}")

        # Phase B: Sitemap fallback
        sitemap_budget = SITEMAP_MAX_PER_DOMAIN
        if cdx_got < 100:
            sitemap_budget = SITEMAP_MAX_PER_DOMAIN * 2
            log.info(
                f"  {dom}: CDX weak, doubling sitemap budget to "
                f"{sitemap_budget}")

        sm_got = 0
        for sm_url in config.sitemap_roots.get(dom, []):
            if sm_got >= sitemap_budget:
                break
            for u, d in crawl_sitemap(sm_url,
                                      max_urls=sitemap_budget - sm_got):
                if u not in seen:
                    seen.add(u)
                    bucket.append((u, d, src))
                    sm_got += 1
            time.sleep(random.uniform(2, 4))
        log.info(
            f"  {dom}: Sitemaps yielded {sm_got} | "
            f"Domain total: {len(bucket)}")

        # Phase C: Pravda archives (UK only, pravda.com.ua)
        if dom == "pravda.com.ua":
            for u, d, s in pravda_archives():
                if u not in seen:
                    seen.add(u)
                    bucket.append((u, d, s))
            log.info(f"  {dom}: Total after archives: {len(bucket)}")

        random.shuffle(bucket)
        domain_buckets[dom] = bucket

    # Round-robin mix
    mixed = round_robin_mix(domain_buckets)
    log.info(f"\nTOTAL {config.language} URLs (round-robin mixed): {len(mixed)}")
    for dom in config.domains:
        c = len(domain_buckets.get(dom, []))
        log.info(f"  {config.src_names.get(dom, dom)}: {c} URLs")
    return mixed
