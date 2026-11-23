"""Baseline: the Question-4 cascade controller, unchanged (knows gravity, ignores wind and hazards)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lander.controllers import CascadeController


def make_policy(spec):
    return CascadeController(gravity=spec["gravity"], spec=spec)
