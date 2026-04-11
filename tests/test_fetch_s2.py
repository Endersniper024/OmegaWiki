"""Tests for tools/fetch_s2.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "tools")
from fetch_s2 import (
    BASE_URL,
    FIELDS,
    MAX_RETRIES,
    citations,
    paper,
    references,
    search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAPER = {
    "paperId": "abc123",
    "title": "LoRA: Low-Rank Adaptation",
    "abstract": "We propose LoRA...",
    "authors": [{"authorId": "1", "name": "Edward Hu"}],
    "year": 2021,
    "citationCount": 5000,
    "venue": "ICLR",
    "externalIds": {"ArXiv": "2106.09685"},
    "url": "https://api.semanticscholar.org/abc123",
}

SAMPLE_PAPER_2 = {
    "paperId": "def456",
    "title": "QLoRA: Quantized Low-Rank Adaptation",
    "abstract": "We present QLoRA...",
    "authors": [{"authorId": "2", "name": "Tim Dettmers"}],
    "year": 2023,
    "citationCount": 2000,
    "venue": "NeurIPS",
    "externalIds": {"ArXiv": "2305.14314"},
    "url": "https://api.semanticscholar.org/def456",
}


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for search function."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_basic_search(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [SAMPLE_PAPER]}
        mock_get.return_value = mock_resp

        results = search("low rank adaptation")
        assert len(results) == 1
        assert results[0]["title"] == "LoRA: Low-Rank Adaptation"

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_search_passes_query_and_limit(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        search("attention mechanism", limit=5)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["query"] == "attention mechanism"
        assert kwargs["params"]["limit"] == 5

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_search_empty_results(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        results = search("nonexistent topic xyz")
        assert results == []

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_search_missing_data_key(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        results = search("test")
        assert results == []

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_search_multiple_results(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [SAMPLE_PAPER, SAMPLE_PAPER_2]}
        mock_get.return_value = mock_resp

        results = search("lora", limit=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# paper()
# ---------------------------------------------------------------------------


class TestPaper:
    """Tests for paper function."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_paper_by_arxiv_id(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_PAPER
        mock_get.return_value = mock_resp

        result = paper("2106.09685")
        assert result["title"] == "LoRA: Low-Rank Adaptation"

        # Verify correct endpoint
        call_args = mock_get.call_args
        assert "ARXIV:2106.09685" in call_args[0][0]

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_paper_includes_fields_param(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_PAPER
        mock_get.return_value = mock_resp

        paper("2106.09685")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["fields"] == FIELDS


# ---------------------------------------------------------------------------
# citations()
# ---------------------------------------------------------------------------


class TestCitations:
    """Tests for citations function."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_citations_returns_citing_papers(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"citingPaper": SAMPLE_PAPER_2},
            ]
        }
        mock_get.return_value = mock_resp

        results = citations("2106.09685")
        assert len(results) == 1
        assert results[0]["title"] == "QLoRA: Quantized Low-Rank Adaptation"

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_citations_empty(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        results = citations("2106.09685")
        assert results == []

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_citations_passes_limit(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        citations("2106.09685", limit=50)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 50

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_citations_missing_citing_paper(self, mock_get, mock_sleep):
        """Handles entries where citingPaper is missing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{}]}
        mock_get.return_value = mock_resp

        results = citations("2106.09685")
        assert len(results) == 1
        assert results[0] == {}


# ---------------------------------------------------------------------------
# references()
# ---------------------------------------------------------------------------


class TestReferences:
    """Tests for references function."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_references_returns_cited_papers(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"citedPaper": SAMPLE_PAPER},
            ]
        }
        mock_get.return_value = mock_resp

        results = references("2305.14314")
        assert len(results) == 1
        assert results[0]["title"] == "LoRA: Low-Rank Adaptation"

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_references_empty(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        results = references("2305.14314")
        assert results == []


# ---------------------------------------------------------------------------
# FIELDS constant
# ---------------------------------------------------------------------------


class TestFieldsConstant:
    """Tests for the FIELDS constant passed to every S2 request."""

    def test_fields_includes_publication_date(self):
        """publicationDate must be requested so /init recency-weight and date-from filter can work."""
        assert "publicationDate" in FIELDS

    def test_fields_includes_year(self):
        """year is the fallback when publicationDate is missing."""
        assert "year" in FIELDS

    def test_fields_includes_citation_count(self):
        """citationCount drives the sqrt-dampened ranking in /init Phase B."""
        assert "citationCount" in FIELDS


# ---------------------------------------------------------------------------
# Rate limiting / retries
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting and retry behavior."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_rate_limit_delay_called(self, mock_get, mock_sleep):
        """_get() sleeps before each request for rate limiting."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        search("test")
        # time.sleep should be called at least once (rate limit delay)
        assert mock_sleep.call_count >= 1

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_429_retries(self, mock_get, mock_sleep):
        """429 response triggers retry with backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"data": [SAMPLE_PAPER]}

        mock_get.side_effect = [resp_429, resp_200]

        results = search("test")
        assert len(results) == 1
        assert mock_get.call_count == 2

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_429_exhausts_retries(self, mock_get, mock_sleep):
        """Repeated 429s exhaust retries and raise RuntimeError."""
        resp_429 = MagicMock()
        resp_429.status_code = 429

        mock_get.return_value = resp_429

        with pytest.raises(RuntimeError, match="rate limited"):
            search("test")
        assert mock_get.call_count == MAX_RETRIES

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_http_error_raises(self, mock_get, mock_sleep):
        """Non-429 HTTP errors raise immediately via raise_for_status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Server Error")
        mock_get.return_value = mock_resp

        with pytest.raises(Exception, match="Server Error"):
            search("test")


# ---------------------------------------------------------------------------
# API key header
# ---------------------------------------------------------------------------


class TestAPIKey:
    """Tests for API key handling."""

    @patch("fetch_s2.time.sleep")
    @patch("fetch_s2.requests.get")
    def test_headers_passed_to_requests(self, mock_get, mock_sleep):
        """Verify that _HEADERS are passed to requests.get."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        search("test")
        _, kwargs = mock_get.call_args
        # headers kwarg should be present (may be empty dict if no key)
        assert "headers" in kwargs


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI invocation via subprocess."""

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Semantic Scholar" in result.stdout

    def test_cli_search_help(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py", "search", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "query" in result.stdout.lower()

    def test_cli_paper_help(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py", "paper", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "arxiv_id" in result.stdout

    def test_cli_citations_help(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py", "citations", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0

    def test_cli_references_help(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py", "references", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0

    def test_cli_no_command_fails(self):
        result = subprocess.run(
            [sys.executable, "tools/fetch_s2.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode != 0
