from __future__ import annotations

import unittest

from news_btc_research.web import extract_html_event, parse_trumpstruth_feed, parse_xml_document


class WebMetadataTests(unittest.TestCase):
    def test_multiple_jsonld_blocks_are_parsed_separately(self) -> None:
        html = """
        <html><head>
          <script type="application/ld+json">{"@type":"WebSite","name":"Example"}</script>
          <script type="application/ld+json">
            {"@type":"NewsArticle","headline":"Market update",
             "datePublished":"2026-09-01T12:34:56Z","url":"/news/1"}
          </script>
        </head><body><a href="/news/2">next</a></body></html>
        """
        event, links = extract_html_event("https://example.com/index", html)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.title, "Market update")
        self.assertEqual(event.url, "https://example.com/news/1")
        self.assertEqual(event.timestamp_source, "json-ld")
        self.assertIn("https://example.com/news/2", links)

    def test_feed_extracts_title_time_and_link_only(self) -> None:
        feed = """<?xml version="1.0"?>
        <rss><channel><item><title>Policy headline</title>
        <pubDate>Tue, 01 Sep 2026 12:34:56 GMT</pubDate>
        <link>https://example.com/news/policy</link><description>Ignored body</description>
        </item></channel></rss>"""
        events, links = parse_xml_document("https://example.com/feed", feed)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].title, "Policy headline")
        self.assertEqual(links, [])

    def test_truthsocial_feed_filters_media_and_keeps_author(self) -> None:
        feed = """<?xml version="1.0"?><rss xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:truth="https://truthsocial.com/ns" xmlns:archive="https://www.trumpstruth.org/rss-extensions"
        xmlns:media="http://search.yahoo.com/mrss/"><channel>
        <item><title>Text only</title><pubDate>Thu, 15 Oct 2026 23:17:18 +0000</pubDate><dc:creator>Donald J. Trump</dc:creator><truth:originalUrl>https://truthsocial.com/@realDonaldTrump/1</truth:originalUrl><archive:postType>original</archive:postType></item>
        <item><title>Video post from October 15, 2026</title><pubDate>Thu, 15 Oct 2026 23:37:15 +0000</pubDate><dc:creator>Donald J. Trump</dc:creator><truth:originalUrl>https://truthsocial.com/@realDonaldTrump/2</truth:originalUrl><archive:postType>original</archive:postType><media:content medium="video" /></item>
        </channel></rss>"""
        events, _ = parse_trumpstruth_feed("https://www.trumpstruth.org/feed", feed)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].author, "Donald J. Trump")
        self.assertEqual(events[0].timestamp_source, "truthsocial-rss:pubDate")


if __name__ == "__main__":
    unittest.main()
