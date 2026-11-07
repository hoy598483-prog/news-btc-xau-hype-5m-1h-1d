from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Iterable

from dateutil.parser import parse as parse_datetime

from .config import CrawlConfig
from .models import NewsEvent


USER_AGENT = "news-btc-token-research/0.1 (+local research; metadata only)"
ARTICLE_TYPES = {"article", "newsarticle", "reportagenewsarticle", "blogposting"}
DATE_META_KEYS = (
    "article:published_time",
    "datepublished",
    "datecreated",
    "publish-date",
    "pubdate",
    "date",
)
TITLE_META_KEYS = ("og:title", "twitter:title", "headline")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_timestamp(value: str) -> datetime | None:
    if not value or not value.strip():
        return None
    try:
        parsed = parse_datetime(value.strip())
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def normalize_url(base_url: str, value: str) -> str | None:
    absolute = urllib.parse.urljoin(base_url, value.strip())
    parsed = urllib.parse.urlsplit(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if any(parsed.path.lower().endswith(ext) for ext in (
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf",
        ".mp3", ".mp4", ".mov", ".zip", ".xml.gz",
    )):
        return None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def event_id(url: str, published_at: datetime, title: str) -> str:
    payload = f"{url}\n{published_at.isoformat()}\n{title}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class HTTPClient:
    def __init__(self, timeout_seconds: float, delay_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self._last_request = 0.0

    def get_text(self, url: str) -> tuple[str, str]:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml,application/rss+xml,application/atom+xml;q=0.9,*/*;q=0.5",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
        if content_type in {"application/xml", "application/rss+xml", "application/atom+xml", "text/xml"}:
            match = re.match(br"\s*<\?xml[^>]*encoding=[\"']([^\"']+)[\"']", body, re.I)
            if match:
                charset = match.group(1).decode("ascii", errors="ignore") or charset
        self._last_request = time.monotonic()
        return body.decode(charset, errors="replace"), content_type


class MetadataHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.meta: dict[str, str] = {}
        self.time_values: list[str] = []
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.jsonld_blocks: list[str] = []
        self._capture_title = False
        self._capture_h1 = False
        self._jsonld_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        lowered = tag.lower()
        if lowered == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content", "").strip()
            if key and content:
                self.meta[key] = content
        elif lowered == "time" and attributes.get("datetime"):
            self.time_values.append(attributes["datetime"])
        elif lowered == "a" and attributes.get("href"):
            normalized = normalize_url(self.base_url, attributes["href"])
            if normalized:
                self.links.append(normalized)
        elif lowered == "title":
            self._capture_title = True
        elif lowered == "h1":
            self._capture_h1 = True
        elif lowered == "script" and "ld+json" in attributes.get("type", "").lower():
            self._jsonld_parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._capture_title = False
        elif lowered == "h1":
            self._capture_h1 = False
        elif lowered == "script":
            if self._jsonld_parts is not None:
                raw = "".join(self._jsonld_parts).strip()
                if raw:
                    self.jsonld_blocks.append(raw)
                self._jsonld_parts = None

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self.title_parts.append(data)
        if self._capture_h1:
            self.h1_parts.append(data)
        if self._jsonld_parts is not None:
            self._jsonld_parts.append(data)


def iter_jsonld_objects(value: object) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in value:
                yield from iter_jsonld_objects(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from iter_jsonld_objects(item)


def jsonld_candidates(parser: MetadataHTMLParser) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    for raw in parser.jsonld_blocks:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in iter_jsonld_objects(payload):
            obj_type = obj.get("@type", "")
            types = obj_type if isinstance(obj_type, list) else [obj_type]
            normalized_types = {str(item).lower() for item in types}
            if normalized_types & ARTICLE_TYPES or "datePublished" in obj:
                title = str(obj.get("headline") or obj.get("name") or "").strip()
                date_value = str(obj.get("datePublished") or obj.get("dateCreated") or "").strip()
                url_value = obj.get("url") or obj.get("mainEntityOfPage") or ""
                url = str(url_value).strip()
                if isinstance(obj.get("mainEntityOfPage"), dict):
                    url = str(obj["mainEntityOfPage"].get("@id") or url)
                if title and date_value:
                    candidates.append((title, date_value, url))
    return candidates


def extract_html_event(url: str, text: str) -> tuple[NewsEvent | None, list[str]]:
    parser = MetadataHTMLParser(url)
    parser.feed(text)

    for title, date_value, jsonld_url in jsonld_candidates(parser):
        published = parse_timestamp(date_value)
        if published:
            final_url = normalize_url(url, jsonld_url) if jsonld_url else url
            final_url = final_url or url
            return NewsEvent(
                event_id(final_url, published, title), final_url, title, published, "json-ld"
            ), parser.links

    date_value = ""
    date_source = ""
    for key in DATE_META_KEYS:
        if parser.meta.get(key):
            date_value, date_source = parser.meta[key], f"meta:{key}"
            break
    if not date_value and parser.time_values:
        date_value, date_source = parser.time_values[0], "time:datetime"
    published = parse_timestamp(date_value)
    if not published:
        return None, parser.links

    title = ""
    for key in TITLE_META_KEYS:
        if parser.meta.get(key):
            title = parser.meta[key].strip()
            break
    if not title:
        title = " ".join("".join(parser.h1_parts).split())
    if not title:
        title = " ".join("".join(parser.title_parts).split())
    if not title:
        return None, parser.links
    return NewsEvent(event_id(url, published, title), url, title, published, date_source), parser.links


def element_text(parent: ET.Element, names: set[str]) -> str:
    for element in parent.iter():
        if local_name(element.tag) in names and element.text and element.text.strip():
            return element.text.strip()
    return ""


def parse_xml_document(url: str, text: str) -> tuple[list[NewsEvent], list[str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [], []
    root_name = local_name(root.tag)
    if root_name == "urlset" or root_name == "sitemapindex":
        return [], [
            element.text.strip()
            for element in root.iter()
            if local_name(element.tag) == "loc" and element.text and element.text.strip()
        ]
    if root_name not in {"rss", "feed", "rdf"}:
        return [], []

    entries = [
        element for element in root.iter()
        if local_name(element.tag) in {"item", "entry"}
    ]
    events: list[NewsEvent] = []
    for entry in entries:
        title = element_text(entry, {"title"})
        date_value = element_text(entry, {"pubdate", "published", "updated", "date"})
        link = element_text(entry, {"link"})
        if not link:
            for element in entry.iter():
                if local_name(element.tag) == "link" and element.attrib.get("href"):
                    link = element.attrib["href"]
                    break
        published = parse_timestamp(date_value)
        final_url = normalize_url(url, link) if link else None
        if title and published and final_url:
            events.append(NewsEvent(
                event_id(final_url, published, title), final_url, title, published, "feed"
            ))
    return events, []


def parse_trumpstruth_feed(url: str, text: str) -> tuple[list[NewsEvent], list[str]]:
    """Parse only original, media-free Truth Social posts from archive RSS."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [], []
    if local_name(root.tag) not in {"rss", "feed", "rdf"}:
        return [], []
    events: list[NewsEvent] = []
    for entry in (item for item in root.iter() if local_name(item.tag) in {"item", "entry"}):
        title = element_text(entry, {"title"})
        post_type = element_text(entry, {"posttype"}).lower()
        has_media = any(local_name(item.tag) == "content" and "medium" in item.attrib for item in entry.iter())
        placeholder = bool(re.match(r"^(video|image|quote) post from ", title, re.I))
        if post_type != "original" or has_media or placeholder:
            continue
        published = parse_timestamp(element_text(entry, {"pubdate", "published", "publishedat"}))
        original_url = element_text(entry, {"originalurl"})
        final_url = normalize_url(url, original_url) if original_url else None
        author = element_text(entry, {"creator"})
        if title and published and final_url and author:
            events.append(NewsEvent(event_id(final_url, published, title), final_url, title,
                                    published, "truthsocial-rss:pubDate", author))
    return events, []


class NewsCrawler:
    def __init__(self, source_url: str, config: CrawlConfig) -> None:
        self.source_url = source_url
        self.config = config
        self.client = HTTPClient(config.timeout_seconds, config.request_delay_seconds)
        self.source_host = urllib.parse.urlsplit(source_url).netloc.lower()
        self.robots = self._load_robots() if config.respect_robots else None

    def _load_robots(self) -> urllib.robotparser.RobotFileParser | None:
        parsed = urllib.parse.urlsplit(self.source_url)
        robots_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
        try:
            text, _ = self.client.get_text(robots_url)
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(text.splitlines())
        return parser

    def _allowed(self, url: str) -> bool:
        if self.config.same_host_only and urllib.parse.urlsplit(url).netloc.lower() != self.source_host:
            return False
        return self.robots is None or self.robots.can_fetch(USER_AGENT, url)

    def crawl(self, start: datetime, end: datetime) -> list[NewsEvent]:
        if urllib.parse.urlsplit(self.source_url).netloc.lower().endswith("trumpstruth.org"):
            return self._crawl_trumpstruth(start, end)
        queue: deque[str] = deque([self.source_url])
        visited: set[str] = set()
        found: dict[str, NewsEvent] = {}
        while queue and len(visited) < self.config.max_pages:
            url = queue.popleft()
            if url in visited or not self._allowed(url):
                continue
            visited.add(url)
            try:
                text, content_type = self.client.get_text(url)
            except (urllib.error.URLError, TimeoutError, ValueError):
                continue

            xml_events: list[NewsEvent] = []
            discovered: list[str] = []
            if "xml" in content_type or text.lstrip().startswith("<?xml"):
                xml_events, discovered = parse_xml_document(url, text)
            if xml_events:
                for event in xml_events:
                    if start <= event.published_at < end:
                        found[event.event_id] = event
                continue
            if discovered:
                for candidate in discovered:
                    normalized = normalize_url(url, candidate)
                    if normalized and normalized not in visited:
                        queue.append(normalized)
                continue

            event, links = extract_html_event(url, text)
            if event and start <= event.published_at < end:
                found[event.event_id] = event
            for candidate in links:
                if candidate not in visited and self._allowed(candidate):
                    queue.append(candidate)

        return sorted(found.values(), key=lambda event: event.published_at)

    def _crawl_trumpstruth(self, start: datetime, end: datetime) -> list[NewsEvent]:
        """Use daily feed slices so an active day cannot hit the feed item cap."""
        found: dict[str, NewsEvent] = {}
        day, end_day = start.date(), end.date()
        pages = 0
        while day < end_day and pages < self.config.max_pages:
            next_day = day.fromordinal(day.toordinal() + 1)
            url = "https://www.trumpstruth.org/feed?" + urllib.parse.urlencode(
                {"start_date": day.isoformat(), "end_date": next_day.isoformat(), "limit": "500"}
            )
            pages += 1
            try:
                text, _ = self.client.get_text(url)
            except (urllib.error.URLError, TimeoutError, ValueError):
                day = next_day
                continue
            events, _ = parse_trumpstruth_feed(url, text)
            for event in events:
                if start <= event.published_at < end:
                    found[event.event_id] = event
            day = next_day
        return sorted(found.values(), key=lambda event: event.published_at)
