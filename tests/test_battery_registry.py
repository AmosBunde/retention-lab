import pytest

from retention_lab.battery.registry import (
    build_arc,
    build_boolq,
    build_hellaswag,
    build_lambada,
    build_piqa,
    build_sciq,
    build_wikitext_docs,
    build_winogrande,
    select_indices,
)


def test_sciq_gold_is_correct_answer_with_support():
    row = {
        "support": "Water boils at 100 C.",
        "question": "At what temperature does water boil?",
        "distractor1": "50 C",
        "distractor2": "150 C",
        "distractor3": "0 C",
        "correct_answer": "100 C",
    }
    item = build_sciq(row, "{support_block}Question: {question}\nAnswer:")
    assert item.gold == 3
    assert item.choices[3] == " 100 C"
    assert item.context.startswith("Water boils at 100 C.\nQuestion:")


def test_sciq_empty_support_leaves_no_blank_line():
    row = {
        "support": "",
        "question": "Q?",
        "distractor1": "a",
        "distractor2": "b",
        "distractor3": "c",
        "correct_answer": "d",
    }
    item = build_sciq(row, "{support_block}Question: {question}\nAnswer:")
    assert item.context.startswith("Question:")


def test_arc_maps_letter_and_number_answer_keys():
    base = {"question": "Q", "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]}}
    assert build_arc({**base, "answerKey": "B"}, "Q: {question}\nA:").gold == 1
    numeric = {"question": "Q", "choices": {"label": ["1", "2"], "text": ["x", "y"]}}
    assert build_arc({**numeric, "answerKey": "2"}, "Q: {question}\nA:").gold == 1


def test_hellaswag_cleanup_removes_bracket_artifacts():
    row = {
        "activity_label": "Cooking",
        "ctx_a": "The chef preheats the oven.",
        "ctx_b": "then she",
        "endings": ["mixes [step] the dough.", "leaves.", "sings.", "waits."],
        "label": "0",
    }
    item = build_hellaswag(row, "{activity_label}: {ctx}")
    assert "[" not in item.choices[0]
    assert item.context.startswith("Cooking: The chef preheats the oven. Then she")
    assert item.gold == 0


def test_piqa_two_solutions():
    row = {"goal": "open a jar", "sol1": "twist the lid", "sol2": "eat the jar", "label": 0}
    item = build_piqa(row, "Question: {goal}\nAnswer:")
    assert item.choices == (" twist the lid", " eat the jar")
    assert item.gold == 0


def test_winogrande_splits_on_blank_with_shared_continuation():
    row = {
        "sentence": "The trophy did not fit in the case because _ was too big.",
        "option1": "the trophy",
        "option2": "the case",
        "answer": "1",
    }
    item = build_winogrande(row, None)
    assert item.contexts[0].endswith("because the trophy")
    assert item.continuation == " was too big."
    assert item.gold == 0


def test_winogrande_rejects_multiple_blanks():
    row = {"sentence": "a _ b _ c", "option1": "x", "option2": "y", "answer": "1"}
    with pytest.raises(ValueError):
        build_winogrande(row, None)


def test_lambada_final_word_split():
    item = build_lambada({"text": "She opened the door and saw the sea"}, None)
    assert item.context.endswith("saw the")
    assert item.continuation == " sea"


def test_boolq_gold_tracks_answer():
    row = {"passage": "P.", "question": "is it true", "answer": True}
    item = build_boolq(row, "{passage}\nQuestion: {question}?\nAnswer:")
    assert item.choices == (" no", " yes")
    assert item.gold == 1
    assert item.context.endswith("Question: is it true?\nAnswer:")


def test_wikitext_docs_are_fixed_size_chunks():
    rows = ["abc " * 100, "def " * 100]
    docs = build_wikitext_docs(rows, doc_chars=128)
    assert all(len(d) <= 128 for d in docs)
    assert all(len(d) >= 64 for d in docs)
    assert "".join(docs)[: len(rows[0])] == rows[0]


def test_select_indices_is_stable_prefix():
    full = select_indices(1000, 100, seed=20260823, task="sciq")
    ci = select_indices(1000, 24, seed=20260823, task="sciq")
    assert ci == full[:24]
    assert select_indices(1000, 100, seed=20260823, task="sciq") == full
    assert select_indices(1000, 100, seed=20260823, task="boolq") != full
