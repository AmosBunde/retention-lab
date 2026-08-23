import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prose_lint import lint_text  # noqa: E402

CONTRACTION_TEXT = "This sentence " + "does" + "n't follow the convention."
EM_DASH_TEXT = "A pause " + chr(0x2014) + " then more."


def test_detects_contraction():
    problems = lint_text(CONTRACTION_TEXT, "bad.md")
    assert any("contraction" in p for p in problems)


def test_detects_em_dash():
    problems = lint_text(EM_DASH_TEXT, "bad.md")
    assert any("em dash" in p for p in problems)


def test_detects_placeholder():
    problems = lint_text("finish this later: " + "TO" + "DO", "bad.md")
    assert any("placeholder" in p for p in problems)


def test_allows_possessives_and_clean_prose():
    clean = "The teacher's capability survives; the student does not regress."
    assert lint_text(clean, "ok.md") == []
