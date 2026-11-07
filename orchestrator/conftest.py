"""Puts orchestrator/ on sys.path so tests can `import llm`, `from audio import tts`, etc."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
