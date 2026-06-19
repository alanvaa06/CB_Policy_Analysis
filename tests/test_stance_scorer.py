from cbp.models.stance_scorer import split_sentences


def test_splits_on_terminal_punctuation():
    text = "The Committee raised rates. Inflation remains elevated! Will it persist?"
    assert split_sentences(text) == [
        "The Committee raised rates.",
        "Inflation remains elevated!",
        "Will it persist?",
    ]


def test_collapses_whitespace_and_newlines():
    text = "First sentence.\nSecond sentence.   Third sentence."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_empty_and_blank_return_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
