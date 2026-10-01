from backend.player_database import PlayerStatsDatabase


def test_completed_hand_aggregates_and_indexed_lookup(tmp_path) -> None:
    database = PlayerStatsDatabase(tmp_path / "players.sqlite3")
    database.update_player_stats(
        "Alex",
        {
            "hand_seen": True,
            "vpip": True,
            "pfr": False,
            "calls": 2,
            "raises": 1,
            "aggressive_actions": 1,
            "total_pot": 24.5,
        },
    )

    player = database.get_player("alex")
    assert player is not None
    assert player["hands_seen"] == 1
    assert player["vpip_count"] == 1
    assert player["pfr_count"] == 0
    assert player["call_count"] == 2
    assert player["raise_count"] == 1
    assert player["total_pots"] == 24.5


def test_hand_history_and_table_size_aggregates(tmp_path) -> None:
    database = PlayerStatsDatabase(tmp_path / "players.sqlite3")
    database.record_completed_hand(
        hand_id=7,
        player_id="Alex",
        table_size=5,
        is_hero=False,
        starting_stack=100,
        ending_stack=88,
        total_pot=32,
        final_street="flop",
        won=False,
        vpip=True,
        pfr=False,
        position="CO",
        preflop_context="call_raise",
        effective_stack_bb=50,
        ended_at="2026-07-27T00:00:00+00:00",
        actions=[
            {
                "id": "action-7",
                "player": "Alex",
                "timestamp": "2026-07-27T00:00:00+00:00",
                "street": "preflop",
                "action": "call",
                "amount": 4,
                "amount_type": "total",
                "contributed": 4,
                "source": "vision",
                "confidence": 0.9,
            },
            {
                "id": "action-other",
                "player": "Hero",
                "timestamp": "2026-07-27T00:00:01+00:00",
                "street": "preflop",
                "action": "raise",
                "amount": 12,
                "amount_type": "total",
                "contributed": 12,
                "source": "manual",
                "confidence": None,
            },
        ],
    )

    bucket = database.get_player_table_stats("Alex", 5)
    assert bucket is not None
    assert bucket["table_size_bucket"] == "5-6"
    assert bucket["hands_seen"] == 1
    assert bucket["call_count"] == 1

    history = database.get_player_history("Alex")
    assert history["hands"][0]["total_contributed"] == 4
    assert history["hands"][0]["position"] == "CO"
    assert history["actions"][0]["action_id"] == "action-7"
