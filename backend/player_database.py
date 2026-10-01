"""Local hand-history and opponent-memory persistence for consent-based games."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Iterable


def table_size_bucket(player_count: int) -> str:
    """Stable population buckets used for contextual opponent tendencies."""
    if player_count <= 2:
        return "1-2"
    if player_count <= 4:
        return "3-4"
    if player_count <= 6:
        return "5-6"
    return "7-9"


class PlayerStatsDatabase:
    """SQLite repository for raw actions, per-hand records, and fast summaries."""

    SUMMARY_COLUMNS = (
        "hands_seen", "vpip_count", "pfr_count", "total_pots", "call_count",
        "raise_count", "aggressive_actions", "fold_count", "check_count",
        "bet_count", "all_in_count", "showdown_count", "win_count",
        "total_contributed", "action_count",
    )

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_stats (
                    player_id TEXT PRIMARY KEY COLLATE NOCASE,
                    hands_seen INTEGER NOT NULL DEFAULT 0,
                    vpip_count INTEGER NOT NULL DEFAULT 0,
                    pfr_count INTEGER NOT NULL DEFAULT 0,
                    total_pots REAL NOT NULL DEFAULT 0.0,
                    call_count INTEGER NOT NULL DEFAULT 0,
                    raise_count INTEGER NOT NULL DEFAULT 0,
                    aggressive_actions INTEGER NOT NULL DEFAULT 0,
                    fold_count INTEGER NOT NULL DEFAULT 0,
                    check_count INTEGER NOT NULL DEFAULT 0,
                    bet_count INTEGER NOT NULL DEFAULT 0,
                    all_in_count INTEGER NOT NULL DEFAULT 0,
                    showdown_count INTEGER NOT NULL DEFAULT 0,
                    win_count INTEGER NOT NULL DEFAULT 0,
                    total_contributed REAL NOT NULL DEFAULT 0.0,
                    action_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._migrate_summary_columns(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_table_stats (
                    player_id TEXT NOT NULL COLLATE NOCASE,
                    table_size_bucket TEXT NOT NULL,
                    hands_seen INTEGER NOT NULL DEFAULT 0,
                    vpip_count INTEGER NOT NULL DEFAULT 0,
                    pfr_count INTEGER NOT NULL DEFAULT 0,
                    total_pots REAL NOT NULL DEFAULT 0.0,
                    call_count INTEGER NOT NULL DEFAULT 0,
                    raise_count INTEGER NOT NULL DEFAULT 0,
                    aggressive_actions INTEGER NOT NULL DEFAULT 0,
                    fold_count INTEGER NOT NULL DEFAULT 0,
                    check_count INTEGER NOT NULL DEFAULT 0,
                    bet_count INTEGER NOT NULL DEFAULT 0,
                    all_in_count INTEGER NOT NULL DEFAULT 0,
                    showdown_count INTEGER NOT NULL DEFAULT 0,
                    win_count INTEGER NOT NULL DEFAULT 0,
                    total_contributed REAL NOT NULL DEFAULT 0.0,
                    action_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_id, table_size_bucket)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_hand_records (
                    hand_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL COLLATE NOCASE,
                    table_size INTEGER NOT NULL,
                    table_size_bucket TEXT NOT NULL,
                    is_hero INTEGER NOT NULL DEFAULT 0,
                    starting_stack REAL NOT NULL DEFAULT 0.0,
                    ending_stack REAL NOT NULL DEFAULT 0.0,
                    total_pot REAL NOT NULL DEFAULT 0.0,
                    total_contributed REAL NOT NULL DEFAULT 0.0,
                    vpip INTEGER NOT NULL DEFAULT 0,
                    pfr INTEGER NOT NULL DEFAULT 0,
                    calls INTEGER NOT NULL DEFAULT 0,
                    raises INTEGER NOT NULL DEFAULT 0,
                    bets INTEGER NOT NULL DEFAULT 0,
                    folds INTEGER NOT NULL DEFAULT 0,
                    checks INTEGER NOT NULL DEFAULT 0,
                    all_ins INTEGER NOT NULL DEFAULT 0,
                    position TEXT NOT NULL DEFAULT 'unknown',
                    preflop_context TEXT NOT NULL DEFAULT 'unknown',
                    effective_stack_bb REAL,
                    aggressive_actions INTEGER NOT NULL DEFAULT 0,
                    final_street TEXT NOT NULL,
                    folded_street TEXT,
                    went_to_showdown INTEGER NOT NULL DEFAULT 0,
                    won INTEGER NOT NULL DEFAULT 0,
                    ended_at TEXT NOT NULL,
                    PRIMARY KEY (hand_id, player_id)
                )
                """
            )
            self._migrate_hand_columns(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_action_records (
                    action_id TEXT PRIMARY KEY,
                    hand_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL COLLATE NOCASE,
                    table_size_bucket TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    street TEXT NOT NULL,
                    action TEXT NOT NULL,
                    amount REAL,
                    amount_type TEXT NOT NULL,
                    contributed REAL NOT NULL DEFAULT 0.0,
                    source TEXT NOT NULL,
                    confidence REAL
                )
                """
            )
            self._migrate_action_columns(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_aliases (
                    alias TEXT PRIMARY KEY COLLATE NOCASE,
                    player_id TEXT NOT NULL COLLATE NOCASE,
                    confirmed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS player_observed_stats (
                    player_id TEXT NOT NULL COLLATE NOCASE,
                    source TEXT NOT NULL,
                    hands_seen INTEGER,
                    vpip REAL,
                    pfr REAL,
                    aggression_factor REAL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_id, source)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player_id ON player_stats(player_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_table_stats_player_bucket ON player_table_stats(player_id, table_size_bucket)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_hand_records_player_hand ON player_hand_records(player_id, hand_id DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_action_records_player_hand ON player_action_records(player_id, hand_id DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_hand_records_player_phase ON player_hand_records(player_id, game_format, tournament_phase)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_hand_records_player_variant ON player_hand_records(player_id, game_variant, tournament_phase)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_player_aliases_player ON player_aliases(player_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_observed_stats_player ON player_observed_stats(player_id, observed_at DESC)")

    @staticmethod
    def _migrate_summary_columns(connection: sqlite3.Connection) -> None:
        """Upgrade existing installations created before expanded opponent memory."""
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(player_stats)")}
        migrations = {
            "fold_count": "INTEGER NOT NULL DEFAULT 0",
            "check_count": "INTEGER NOT NULL DEFAULT 0",
            "bet_count": "INTEGER NOT NULL DEFAULT 0",
            "all_in_count": "INTEGER NOT NULL DEFAULT 0",
            "showdown_count": "INTEGER NOT NULL DEFAULT 0",
            "win_count": "INTEGER NOT NULL DEFAULT 0",
            "total_contributed": "REAL NOT NULL DEFAULT 0.0",
            "action_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in migrations.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE player_stats ADD COLUMN {column} {definition}")

    @staticmethod
    def _migrate_hand_columns(connection: sqlite3.Connection) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(player_hand_records)")}
        migrations = {
            "position": "TEXT NOT NULL DEFAULT 'unknown'",
            "preflop_context": "TEXT NOT NULL DEFAULT 'unknown'",
            "effective_stack_bb": "REAL",
            "game_format": "TEXT NOT NULL DEFAULT 'cash'",
            "tournament_phase": "TEXT NOT NULL DEFAULT 'cash'",
            "players_remaining": "INTEGER",
            "game_variant": "TEXT NOT NULL DEFAULT 'nlh'",
        }
        for column, definition in migrations.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE player_hand_records ADD COLUMN {column} {definition}")

    @staticmethod
    def _migrate_action_columns(connection: sqlite3.Connection) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(player_action_records)")}
        for column in ("game_format", "tournament_phase", "game_variant"):
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE player_action_records ADD COLUMN {column} TEXT NOT NULL DEFAULT '{'nlh' if column == 'game_variant' else 'cash'}'"
                )

    def ensure_player(self, player_id: str, connection: sqlite3.Connection | None = None) -> None:
        clean_id = player_id.strip()
        if not clean_id:
            return
        if connection is not None:
            connection.execute("INSERT OR IGNORE INTO player_stats(player_id) VALUES (?)", (clean_id,))
            return
        with self._lock, self._connect() as local_connection:
            local_connection.execute("INSERT OR IGNORE INTO player_stats(player_id) VALUES (?)", (clean_id,))

    def get_player(self, player_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT player_id, {', '.join(self.SUMMARY_COLUMNS)} FROM player_stats WHERE player_id = ?",
                (player_id.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def list_players(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT player_id, {', '.join(self.SUMMARY_COLUMNS)} FROM player_stats ORDER BY player_id COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_rate(value: float | None) -> float | None:
        if value is None:
            return None
        rate = float(value)
        if rate > 1.0:
            rate /= 100.0
        return max(0.0, min(rate, 1.0))

    def upsert_observed_stats(
        self,
        player_id: str,
        *,
        hands_seen: int | None = None,
        vpip: float | None = None,
        pfr: float | None = None,
        aggression_factor: float | None = None,
        source: str = "clubgg_profile",
        confidence: float | None = None,
    ) -> None:
        """Store a visible platform stat snapshot without fabricating local actions."""
        clean_player = player_id.strip()
        clean_source = source.strip() or "external_profile"
        if not clean_player:
            return
        with self._lock, self._connect() as connection:
            self.ensure_player(clean_player, connection)
            connection.execute(
                """
                INSERT INTO player_observed_stats (
                    player_id, source, hands_seen, vpip, pfr, aggression_factor, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, source) DO UPDATE SET
                    hands_seen = COALESCE(excluded.hands_seen, player_observed_stats.hands_seen),
                    vpip = COALESCE(excluded.vpip, player_observed_stats.vpip),
                    pfr = COALESCE(excluded.pfr, player_observed_stats.pfr),
                    aggression_factor = COALESCE(excluded.aggression_factor, player_observed_stats.aggression_factor),
                    confidence = excluded.confidence,
                    observed_at = CURRENT_TIMESTAMP
                """,
                (
                    clean_player, clean_source, hands_seen,
                    self._normalize_rate(vpip), self._normalize_rate(pfr),
                    aggression_factor, 1.0 if confidence is None else max(0.0, min(float(confidence), 1.0)),
                ),
            )

    def get_observed_stats(self, player_id: str, game_variant: str | None = None) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            if game_variant is None:
                row = connection.execute(
                    "SELECT * FROM player_observed_stats WHERE player_id = ? ORDER BY observed_at DESC LIMIT 1",
                    (player_id.strip(),),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM player_observed_stats WHERE player_id = ? AND source LIKE ? "
                    "ORDER BY observed_at DESC LIMIT 1",
                    (player_id.strip(), f"%:{game_variant}"),
                ).fetchone()
        return dict(row) if row else None

    def get_player_variant_stats(self, player_id: str, game_variant: str, phase: str | None = None) -> dict[str, Any] | None:
        """Only same-variant hands contribute to a strategy profile."""
        where_phase = " AND tournament_phase = ?" if phase else ""
        parameters = (player_id.strip(), game_variant, phase) if phase else (player_id.strip(), game_variant)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT player_id, game_variant,
                    COUNT(*) AS hands_seen,
                    SUM(vpip) AS vpip_count, SUM(pfr) AS pfr_count,
                    SUM(total_pot) AS total_pots, SUM(calls) AS call_count,
                    SUM(raises) AS raise_count, SUM(aggressive_actions) AS aggressive_actions,
                    SUM(folds) AS fold_count, SUM(checks) AS check_count,
                    SUM(bets) AS bet_count, SUM(all_ins) AS all_in_count,
                    SUM(went_to_showdown) AS showdown_count, SUM(won) AS win_count,
                    SUM(total_contributed) AS total_contributed,
                    SUM(calls + raises + bets + folds + checks + all_ins) AS action_count
                FROM player_hand_records
                WHERE player_id = ? AND game_variant = ?{where_phase}
                GROUP BY player_id, game_variant
                """,
                parameters,
            ).fetchone()
        return dict(row) if row else None

    def get_player_table_stats(self, player_id: str, player_count: int) -> dict[str, Any] | None:
        bucket = table_size_bucket(player_count)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT player_id, table_size_bucket, {', '.join(self.SUMMARY_COLUMNS)} "
                "FROM player_table_stats WHERE player_id = ? AND table_size_bucket = ?",
                (player_id.strip(), bucket),
            ).fetchone()
        return dict(row) if row else None

    def get_player_table_breakdown(self, player_id: str, game_variant: str | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            if game_variant is None:
                rows = connection.execute(
                    f"SELECT player_id, table_size_bucket, {', '.join(self.SUMMARY_COLUMNS)} "
                    "FROM player_table_stats WHERE player_id = ? ORDER BY table_size_bucket",
                    (player_id.strip(),),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT player_id, table_size_bucket, game_variant,
                        COUNT(*) AS hands_seen,
                        SUM(vpip) AS vpip_count, SUM(pfr) AS pfr_count,
                        SUM(total_pot) AS total_pots, SUM(calls) AS call_count,
                        SUM(raises) AS raise_count, SUM(aggressive_actions) AS aggressive_actions,
                        SUM(folds) AS fold_count, SUM(checks) AS check_count,
                        SUM(bets) AS bet_count, SUM(all_ins) AS all_in_count,
                        SUM(went_to_showdown) AS showdown_count, SUM(won) AS win_count,
                        SUM(total_contributed) AS total_contributed,
                        SUM(calls + raises + bets + folds + checks + all_ins) AS action_count
                    FROM player_hand_records
                    WHERE player_id = ? AND game_variant = ?
                    GROUP BY player_id, table_size_bucket, game_variant
                    ORDER BY table_size_bucket
                    """,
                    (player_id.strip(), game_variant),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_player_history(self, player_id: str, limit: int = 100) -> dict[str, list[dict[str, Any]]]:
        """Return recent normalized hand and action records for model calibration."""
        safe_limit = max(1, min(int(limit), 500))
        with self._lock, self._connect() as connection:
            hands = connection.execute(
                "SELECT * FROM player_hand_records WHERE player_id = ? "
                "ORDER BY hand_id DESC LIMIT ?",
                (player_id.strip(), safe_limit),
            ).fetchall()
            actions = connection.execute(
                "SELECT * FROM player_action_records WHERE player_id = ? "
                "ORDER BY hand_id DESC, timestamp DESC LIMIT ?",
                (player_id.strip(), safe_limit),
            ).fetchall()
        return {"hands": [dict(row) for row in hands], "actions": [dict(row) for row in actions]}

    def get_player_phase_stats(self, player_id: str, phase: str) -> dict[str, Any] | None:
        """Read measured behavior in one format/stage; never infer it from a label."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT player_id, tournament_phase,
                    COUNT(*) AS hands_seen,
                    SUM(vpip) AS vpip_count, SUM(pfr) AS pfr_count,
                    SUM(total_pot) AS total_pots, SUM(calls) AS call_count,
                    SUM(raises) AS raise_count, SUM(aggressive_actions) AS aggressive_actions,
                    SUM(folds) AS fold_count, SUM(checks) AS check_count,
                    SUM(bets) AS bet_count, SUM(all_ins) AS all_in_count,
                    SUM(went_to_showdown) AS showdown_count, SUM(won) AS win_count,
                    SUM(total_contributed) AS total_contributed,
                    SUM(calls + raises + bets + folds + checks + all_ins) AS action_count
                FROM player_hand_records
                WHERE player_id = ? AND tournament_phase = ?
                GROUP BY player_id, tournament_phase
                """,
                (player_id.strip(), phase),
            ).fetchone()
        return dict(row) if row else None

    def resolve_alias(self, alias: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT player_id FROM player_aliases WHERE alias = ?", (alias.strip(),)
            ).fetchone()
        return str(row["player_id"]) if row else None

    def remember_alias(self, alias: str, player_id: str) -> None:
        """Persist a human-confirmed detector spelling without merging profiles."""
        clean_alias, clean_player = alias.strip(), player_id.strip()
        if not clean_alias or not clean_player:
            return
        with self._lock, self._connect() as connection:
            self.ensure_player(clean_player, connection)
            connection.execute(
                """
                INSERT INTO player_aliases(alias, player_id) VALUES (?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    player_id = excluded.player_id,
                    confirmed_at = CURRENT_TIMESTAMP
                """,
                (clean_alias, clean_player),
            )

    def update_player_stats(self, player_id: str, action_taken: str | dict[str, Any]) -> None:
        """Compatibility updater for integrations that submit simple aggregates."""
        update = self._normalize_update(action_taken)
        with self._lock, self._connect() as connection:
            self.ensure_player(player_id, connection)
            self._increment_summary(connection, "player_stats", player_id.strip(), update)

    def record_completed_hand(
        self,
        *,
        hand_id: int,
        player_id: str,
        table_size: int,
        is_hero: bool,
        starting_stack: float,
        ending_stack: float,
        total_pot: float,
        final_street: str,
        won: bool,
        vpip: bool,
        pfr: bool,
        position: str,
        preflop_context: str,
        effective_stack_bb: float | None,
        actions: Iterable[dict[str, Any]],
        ended_at: str,
        game_format: str = "cash",
        tournament_phase: str = "cash",
        players_remaining: int | None = None,
        game_variant: str = "nlh",
    ) -> None:
        """Atomically persist raw actions plus global and table-size summaries."""
        player_actions = [dict(action) for action in actions if action.get("player") == player_id]
        counts = self._action_counts(player_actions)
        folded_street = next((action["street"] for action in reversed(player_actions) if action["action"] == "fold"), None)
        summary = {
            "hands_seen": 1,
            "vpip_count": int(vpip),
            "pfr_count": int(pfr),
            "total_pots": max(0.0, float(total_pot)),
            "call_count": counts["calls"],
            "raise_count": counts["raises"],
            "aggressive_actions": counts["aggressive_actions"],
            "fold_count": counts["folds"],
            "check_count": counts["checks"],
            "bet_count": counts["bets"],
            "all_in_count": counts["all_ins"],
            "showdown_count": int(final_street == "showdown" and folded_street is None),
            "win_count": int(won),
            "total_contributed": sum(float(action.get("contributed") or 0.0) for action in player_actions),
            "action_count": len(player_actions),
        }
        bucket = table_size_bucket(table_size)
        with self._lock, self._connect() as connection:
            self.ensure_player(player_id, connection)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO player_hand_records (
                    hand_id, player_id, table_size, table_size_bucket, is_hero,
                    starting_stack, ending_stack, total_pot, total_contributed,
                    vpip, pfr, calls, raises, bets, folds, checks, all_ins,
                    position, preflop_context, effective_stack_bb,
                    aggressive_actions, final_street, folded_street,
                    went_to_showdown, won, ended_at, game_format,
                    tournament_phase, players_remaining, game_variant
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hand_id, player_id, table_size, bucket, int(is_hero),
                    float(starting_stack), float(ending_stack), summary["total_pots"], summary["total_contributed"],
                    summary["vpip_count"], summary["pfr_count"], summary["call_count"], summary["raise_count"],
                    summary["bet_count"], summary["fold_count"], summary["check_count"], summary["all_in_count"],
                    position, preflop_context, effective_stack_bb,
                    summary["aggressive_actions"], final_street, folded_street,
                    summary["showdown_count"], summary["win_count"], ended_at,
                    game_format, tournament_phase, players_remaining, game_variant,
                ),
            )
            if cursor.rowcount == 0:
                return
            self._increment_summary(connection, "player_stats", player_id, summary)
            self._increment_table_summary(connection, player_id, bucket, summary)
            for action in player_actions:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO player_action_records (
                        action_id, hand_id, player_id, table_size_bucket, timestamp,
                        street, action, amount, amount_type, contributed, source, confidence
                        , game_format, tournament_phase, game_variant
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        action["id"], hand_id, player_id, bucket, action["timestamp"],
                        action["street"], action["action"], action.get("amount"),
                        action["amount_type"], float(action.get("contributed") or 0.0),
                        action["source"], action.get("confidence"), game_format, tournament_phase, game_variant,
                    ),
                )

    def _increment_summary(self, connection: sqlite3.Connection, table: str, player_id: str, update: dict[str, float]) -> None:
        assignments = ", ".join(f"{column} = {column} + ?" for column in self.SUMMARY_COLUMNS)
        values = [update.get(column, 0) for column in self.SUMMARY_COLUMNS]
        connection.execute(
            f"UPDATE {table} SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE player_id = ?",
            (*values, player_id),
        )

    def _increment_table_summary(self, connection: sqlite3.Connection, player_id: str, bucket: str, update: dict[str, float]) -> None:
        columns = ", ".join(self.SUMMARY_COLUMNS)
        placeholders = ", ".join("?" for _ in self.SUMMARY_COLUMNS)
        update_columns = ", ".join(f"{column} = player_table_stats.{column} + excluded.{column}" for column in self.SUMMARY_COLUMNS)
        values = [update.get(column, 0) for column in self.SUMMARY_COLUMNS]
        connection.execute(
            f"""
            INSERT INTO player_table_stats (player_id, table_size_bucket, {columns})
            VALUES (?, ?, {placeholders})
            ON CONFLICT(player_id, table_size_bucket) DO UPDATE SET
                {update_columns}, updated_at = CURRENT_TIMESTAMP
            """,
            (player_id, bucket, *values),
        )

    @staticmethod
    def _action_counts(actions: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = {"calls": 0, "raises": 0, "bets": 0, "folds": 0, "checks": 0, "all_ins": 0, "aggressive_actions": 0}
        for action in actions:
            name = action.get("action")
            if name == "call": counts["calls"] += 1
            elif name == "raise": counts["raises"] += 1; counts["aggressive_actions"] += 1
            elif name == "bet": counts["bets"] += 1; counts["aggressive_actions"] += 1
            elif name == "all_in": counts["all_ins"] += 1; counts["aggressive_actions"] += 1
            elif name == "fold": counts["folds"] += 1
            elif name == "check": counts["checks"] += 1
        return counts

    @staticmethod
    def _normalize_update(action_taken: str | dict[str, Any]) -> dict[str, float]:
        update = {column: 0 for column in PlayerStatsDatabase.SUMMARY_COLUMNS}
        update["total_pots"] = update["total_contributed"] = 0.0
        if isinstance(action_taken, str):
            token = action_taken.strip().lower()
            if token in {"hand_seen", "hand_complete"}: update["hands_seen"] = 1
            elif token == "vpip": update["vpip_count"] = 1
            elif token == "pfr": update["pfr_count"] = 1
            elif token == "call": update["call_count"] = update["action_count"] = 1
            elif token == "fold": update["fold_count"] = update["action_count"] = 1
            elif token == "check": update["check_count"] = update["action_count"] = 1
            elif token in {"raise", "bet", "all_in"}:
                field = {"raise": "raise_count", "bet": "bet_count", "all_in": "all_in_count"}[token]
                update[field] = update["aggressive_actions"] = update["action_count"] = 1
            return update
        update["hands_seen"] = int(bool(action_taken.get("hand_seen", False)))
        update["vpip_count"] = int(bool(action_taken.get("vpip", False)))
        update["pfr_count"] = int(bool(action_taken.get("pfr", False)))
        update["total_pots"] = max(0.0, float(action_taken.get("total_pot", 0.0)))
        update["call_count"] = max(0, int(action_taken.get("calls", 0)))
        update["raise_count"] = max(0, int(action_taken.get("raises", 0)))
        update["aggressive_actions"] = max(0, int(action_taken.get("aggressive_actions", update["raise_count"])))
        return update
