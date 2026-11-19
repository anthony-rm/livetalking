#!/usr/bin/env python3
"""Tests for the emotion SURFACE: the vocabulary route and the UI contract.

The model side is covered by `test_ditto_emotion.py`; this file covers what the
browser talks to. `GET /api/emotions` is the single source of the label list the
page renders, so a stale hardcoded option list in the HTML is a defect this
route exists to prevent.

No server, no aiohttp test harness: the handler is an async function taking a
request-like object, so a 20-line stub exercises every branch.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from avatars.ditto.emotion import EMOTIONS
from server import routes


class FakeSession:
    def __init__(self, emotion=None, supports=True):
        self._emotion = emotion
        self._gain = 1.0
        if supports:
            self.set_emotion = self._set   # attribute presence IS the capability
            self.current_emotion = self._current

    def _set(self, value, gain=None):
        # The real DittoReal.set_emotion parses through
        # `avatars.ditto.emotion.parse_emotion`, which raises on an unknown
        # label; the route's job is only to surface that as a JSON error.
        from avatars.ditto.emotion import parse_emotion
        if gain is not None:
            if gain < 0:
                raise ValueError(f"emotion gain must be >= 0, got {gain}")
            self._gain = gain
        self._emotion = parse_emotion(value)
        return self._emotion

    def _current(self):
        from avatars.ditto.emotion import emotion_name
        name = emotion_name(self._emotion)
        if self._emotion is None or self._gain == 1.0:
            return name
        return f"{name}@{self._gain:g}"


class FakeRequest:
    def __init__(self, query=None, body=None):
        self.query = query or {}
        self._body = body or {}
        self.app = {}

    async def json(self):
        return self._body


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def body_of(response) -> dict:
    return json.loads(response.text)


@pytest.fixture
def no_sessions(monkeypatch):
    monkeypatch.setattr(routes, "get_session", lambda request, sid: None)


@pytest.fixture
def one_session(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(routes, "get_session", lambda request, sid: session)
    return session


# ─── vocabulary endpoint ───────────────────────────────────────────────────
def test_emotions_route_serves_all_eight_in_upstream_order(no_sessions):
    """The page must be able to render 8 buttons — not the 3 it hardcoded."""
    data = body_of(run(routes.emotions(FakeRequest())))["data"]
    assert data["emotions"] == list(EMOTIONS)
    assert len(data["emotions"]) == 8
    assert data["emotions"][3] == "happy"
    assert data["emotions"][4] == "neutral"


def test_emotions_route_reports_unsupported_without_a_session(no_sessions):
    """No session yet -> controls disabled, but the vocabulary is still served
    so the page can build them before the first connect."""
    data = body_of(run(routes.emotions(FakeRequest())))["data"]
    assert data["supported"] is False
    assert data["current"] is None


def test_emotions_route_reports_current_from_the_session(one_session):
    one_session._emotion = 6                       # parsed form, as the backend stores it
    data = body_of(run(routes.emotions(
        FakeRequest(query={"sessionid": "abc"}))))["data"]
    assert data["supported"] is True
    assert data["current"] == "surprise"


def test_emotions_route_reports_default_when_nothing_is_set(one_session):
    data = body_of(run(routes.emotions(
        FakeRequest(query={"sessionid": "abc"}))))["data"]
    assert data["current"] == "default"


def test_emotions_route_marks_a_backend_without_emotion_unsupported(monkeypatch):
    plain = FakeSession(supports=False)       # e.g. a wav2lip session
    monkeypatch.setattr(routes, "get_session", lambda request, sid: plain)
    data = body_of(run(routes.emotions(
        FakeRequest(query={"sessionid": "abc"}))))["data"]
    assert data["supported"] is False


# ─── sticky setter (what the buttons call) ────────────────────────────────
def test_set_emotion_accepts_a_name_and_echoes_the_canonical_form(one_session):
    data = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "happy"}))))["data"]
    assert data["emotion"] == "happy"
    assert one_session._emotion == 3               # index, as the model takes it


def test_set_emotion_clears_with_null(one_session):
    one_session._emotion = 5
    data = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": None}))))["data"]
    assert data["emotion"] == "default"
    assert one_session._emotion is None


def test_set_emotion_surfaces_an_unknown_label_as_an_error(one_session):
    out = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "excited"}))))
    assert out["code"] != 0
    assert "unknown emotion" in out["msg"]


def test_set_emotion_on_a_backend_without_support_is_a_clean_error(monkeypatch):
    plain = FakeSession(supports=False)
    monkeypatch.setattr(routes, "get_session", lambda request, sid: plain)
    out = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "happy"}))))
    assert out["code"] != 0
    assert "ditto" in out["msg"]


# ─── gain (strength) ──────────────────────────────────────────────────────
def test_gain_travels_from_the_body_to_the_session(one_session):
    """The route must forward `gain`; a silent drop would look like a weak
    emotion with no error anywhere."""
    data = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "happy",
                          "gain": 2.0}))))["data"]
    assert one_session._gain == 2.0
    assert data["emotion"] == "happy@2"


def test_omitting_gain_keeps_the_current_strength(one_session):
    """Re-aiming a mood must not silently reset its strength to 1."""
    run(routes.set_emotion(FakeRequest(
        body={"sessionid": "abc", "emotion": "happy", "gain": 3.0})))
    data = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "angry"}))))["data"]
    assert one_session._gain == 3.0
    assert data["emotion"] == "angry@3"


def test_negative_gain_is_rejected(one_session):
    out = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": "happy",
                          "gain": -1.0}))))
    assert out["code"] != 0
    assert ">= 0" in out["msg"]


def test_emotions_route_publishes_the_gain_presets(no_sessions):
    """The UI builds its strength control from this, not from hardcoded
    numbers in the HTML."""
    data = body_of(run(routes.emotions(FakeRequest())))["data"]
    assert data["gain"]["default"] == 1.0
    assert 1.0 in data["gain"]["presets"]
    assert max(data["gain"]["presets"]) > 1.0


def test_a_strength_with_no_label_reads_as_plain_default(one_session):
    """Clearing the emotion must not leave "default@2" on screen: strength
    with nothing to apply to is noise, and the box gate caught it doing so."""
    run(routes.set_emotion(FakeRequest(
        body={"sessionid": "abc", "emotion": "happy", "gain": 2.0})))
    data = body_of(run(routes.set_emotion(
        FakeRequest(body={"sessionid": "abc", "emotion": None}))))["data"]
    assert data["emotion"] == "default"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
