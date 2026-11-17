"""Sentiment scoring for generated text (VADER, a lexicon/rule-based analyser; runs offline)."""
from __future__ import annotations

from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def sentiment(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    return float(_analyzer().polarity_scores(text or "")["compound"])


def label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"
