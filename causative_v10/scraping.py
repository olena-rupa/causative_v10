# -*- coding: utf-8 -*-
"""URL collection and article text extraction utilities."""

import gzip
import logging
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

import cloudscraper
import requests
from bs4 import BeautifulSoup

from causative_v10.config import (
    CDX_PER_YEAR, CDX_RETRIES, CDX_RETRY_SLEEP, CDX_TIMEOUT,
    CDX_YEARS, CorpusConfig, MIN_ARTICLE_WORDS, MIN_DATE,
    SITEMAP_MAX_PER_DOMAIN, SKIP_URL, TIMEOUT,
)

try:
    from newspaper import Article as NpArticle
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

log = logging.getLogger("v10.scraping")

_SCRAPER = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows"}
)


def cget(url: str, retries: int = 3,
         timeout: int = TIMEOUT) -> Optional[requests.Response]:
    """Execute HTTP GET request with retry backoff on transient errors."""
    for att in range(1, retries + 1):
        try:
            r = _SCRAPER.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code in (403, 404, 410):
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            log.debug(f"GET fail ({att}/{retries}) {url[:65]}: {e}")
            if att < retries:
                time.sleep(random.uniform(2, 5))
    return None


_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
    "%Y%m%d%H%M%S", "%d.%m.%Y", "%d/%m/%Y",
)


def parse_date(s: str) -> Optional[datetime]:
    """Parse publication timestamp string into a naive datetime object."""
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


def scrape(url: str) -> Tuple[Optional[str], str]:
    """Extract clean article body and publication date (minimum 300 words)."""
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

    try:
        r = cget(url)
        if not r:
            return None, ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "figure", "figcaption", "noscript",
                         "iframe", "form"]):
            tag.decompose()

        body = (
            soup.find("article") or
            soup.find("div", class_=re.compile(
                r"article|story|content|post-body", re.I)) or
            soup.find("main") or
            soup
        )

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in body.find_all("p")
            if len(p.get_text(" ", strip=True)) > 25
        )

        ds = ""
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            ds = time_tag["datetime"]

        if text and len(text.split()) >= MIN_ARTICLE_WORDS:
            return text, ds
    except Exception:
        pass

    return None, ""


def wayback_cdx_yearly(domain: str, src_names: Dict[str, str],
                       max_per_year: int = CDX_PER_YEAR
                       ) -> List[Tuple[str, str, str]]:
    """Query Wayback Machine CDX API across yearly intervals (2021-2026)."""
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
                r = _SCRAPER.get(cdx_url, timeout=CDX_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    log.info(
                        f"    CDX OK {domain}/{year}: {len(data)-1} raw rows"
                    )
                    break
                else:
                    log.warning(
                        f"    CDX HTTP {r.status_code} for {domain}/{year}"
                    )
            except Exception as e:
                log.warning(
                    f"    CDX FAIL {domain}/{year} (att {attempt}): {e}"
                )
            if attempt < CDX_RETRIES:
                time.sleep(CDX_RETRY_SLEEP)

        if data is None:
            log.warning(f"    CDX exhausted for {domain}/{year}")
            continue

        rows = data[1:]
        random.shuffle(rows)

        year_count = 0
        for row in rows:
            if len(row) < 2:
                continue
            ts, orig = row[0], row[1]
            if SKIP_URL.search(orig) or orig in seen:
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


def crawl_sitemap(url: str,
                  max_urls: int = SITEMAP_MAX_PER_DOMAIN,
                  depth: int = 0) -> List[Tuple[str, str]]:
    """Recursively parse XML and .xml.gz sitemaps up to depth 4."""
    if depth > 4:
        return []

    results: List[Tuple[str, str]] = []
    r = cget(url, timeout=40)
    if not r:
        return results

    raw_bytes = r.content
    if url.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except OSError:
            return results

    soup = BeautifulSoup(raw_bytes, "xml")

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


def pravda_archives(days: int = 300,
                    max_urls: int = 3000
                    ) -> List[Tuple[str, str, str]]:
    """Collect Ukrainian article links from Ukrainska Pravda daily archives."""
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
            if ("/news/" in h or "/articles/" in h) and "/rus/" not in h and "/eng/" not in h:
                fu = urljoin(base, h)
                if fu not in seen and not SKIP_URL.search(fu):
                    seen.add(fu)
                    results.append(
                        (fu, d.strftime("%Y-%m-%d"), "Ukrainska Pravda")
                    )

        time.sleep(random.uniform(1.5, 3))
        if len(results) >= max_urls:
            break

    log.info(f"  Pravda archives: {len(results)} URLs")
    return results


def round_robin_mix(domain_buckets: Dict[str, list]) -> list:
    """Interleave collected URLs across media domains using round-robin queues."""
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


def collect_urls(config: CorpusConfig) -> List[Tuple[str, str, str]]:
    """Aggregate and balance candidate URLs across all configured domains."""
    domain_buckets: Dict[str, list] = {}

    for dom in config.domains:
        src = config.src_names.get(dom, dom)
        seen: set = set()
        bucket: list = []

        for u, d, s in wayback_cdx_yearly(dom, config.src_names):
            if u not in seen:
                seen.add(u)
                bucket.append((u, d, s))
        cdx_got = len(bucket)
        log.info(f"  {dom}: CDX yielded {cdx_got}")

        sitemap_budget = SITEMAP_MAX_PER_DOMAIN
        if cdx_got < 100:
            sitemap_budget = SITEMAP_MAX_PER_DOMAIN * 2
            log.info(
                f"  {dom}: CDX weak, doubling sitemap budget to {sitemap_budget}"
            )

        sm_got = 0
        for sm_url in config.sitemap_roots.get(dom, []):
            if sm_got >= sitemap_budget:
                break
            for u, d in crawl_sitemap(sm_url, max_urls=sitemap_budget - sm_got):
                if u not in seen:
                    seen.add(u)
                    bucket.append((u, d, src))
                    sm_got += 1
            time.sleep(random.uniform(2, 4))
        log.info(
            f"  {dom}: Sitemaps yielded {sm_got} | Domain total: {len(bucket)}"
        )

        if dom == "pravda.com.ua":
            for u, d, s in pravda_archives():
                if u not in seen:
                    seen.add(u)
                    bucket.append((u, d, s))
            log.info(f"  {dom}: Total after archives: {len(bucket)}")

        random.shuffle(bucket)
        domain_buckets[dom] = bucket

    mixed = round_robin_mix(domain_buckets)
    log.info(f"\nTOTAL {config.language} URLs (round-robin mixed): {len(mixed)}")
    for dom in config.domains:
        c = len(domain_buckets.get(dom, []))
        log.info(f"  {config.src_names.get(dom, dom)}: {c} URLs")
    return mixed
