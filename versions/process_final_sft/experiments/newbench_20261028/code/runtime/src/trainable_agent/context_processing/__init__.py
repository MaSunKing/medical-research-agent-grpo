"""Shared rules; fetching/parsing and policy decisions remain outside this package."""
from .cleaning import clean_chunk, clean_observation
from .deterministic_context import project_evidence
