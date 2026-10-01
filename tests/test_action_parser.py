from backend.speech.action_parser import RuleBasedActionParser


def test_named_raise_to_total_amount() -> None:
    parser = RuleBasedActionParser()
    result = parser.parse("Jason raises to 10")

    assert result.player == "Jason"
    assert result.action == "raise"
    assert result.amount == 10
    assert result.amount_type == "total"
    assert result.confidence >= 0.9


def test_raise_more_is_additional() -> None:
    parser = RuleBasedActionParser()
    result = parser.parse("raise 15 more")

    assert result.player is None
    assert result.action == "raise"
    assert result.amount == 15
    assert result.amount_type == "additional"


def test_all_in_without_name_is_supported() -> None:
    parser = RuleBasedActionParser()
    result = parser.parse("all in")

    assert result.action == "all_in"
    assert result.player is None
    assert result.confidence >= 0.75
