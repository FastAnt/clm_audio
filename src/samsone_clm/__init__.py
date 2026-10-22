"""Frozen Samsone state/text paths, one shared adapter, official CLM heads."""


def rank(audio, question, choices):
    from .api import rank as implementation
    return implementation(audio, question, choices)
