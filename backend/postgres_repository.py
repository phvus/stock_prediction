from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql


def load_env_file() -> None:
    """Load key=value pairs from project .env if variables are unset."""
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.getenv(key):
            os.environ[key] = value


load_env_file()


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: str
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DbConfig":
        missing = [
            key
            for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
            if os.getenv(key) is None
        ]
        if missing:
            raise ValueError(
                "Missing PostgreSQL env vars: " + ", ".join(missing) + ". "
                "Set them in your environment or .env file."
            )
        return cls(
            host=os.getenv("DB_HOST", ""),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", ""),
        )


class PostgresRepository:
    """Data access layer for stock data in PostgreSQL.

    Supports two schemas:
    1. Legacy main-branch schema: one table per symbol + company_info.
    2. Normalized schema: stock_prices(symbol, time, close, ...) + company_info.
    """

    def __init__(self, config: Optional[DbConfig] = None) -> None:
        self.config = config or DbConfig.from_env()

    def _connect(self):
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
        )

    def _table_exists(self, conn, table_name: str) -> bool:
        query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        );
        """
        with conn.cursor() as cur:
            cur.execute(query, (table_name,))
            return bool(cur.fetchone()[0])

    def test_connection(self) -> Dict[str, str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
            return {
                "status": "ok",
                "database": self.config.dbname,
                "version": version,
            }

    def get_schema_mode(self) -> str:
        with self._connect() as conn:
            if self._table_exists(conn, "stock_prices"):
                return "normalized"
            return "legacy"

    def get_companies(self) -> pd.DataFrame:
        with self._connect() as conn:
            if self._table_exists(conn, "company_info"):
                query = """
                SELECT
                    symbol,
                    COALESCE(icb_name2, 'Unknown') AS sector,
                    listing_date,
                    ceo_name
                FROM company_info
                ORDER BY symbol;
                """
                return pd.read_sql_query(query, conn)

            symbols = self._list_legacy_symbol_tables(conn)
            if not symbols:
                return pd.DataFrame(columns=["symbol", "sector", "listing_date", "ceo_name"])
            return pd.DataFrame(
                {
                    "symbol": [s.upper() for s in symbols],
                    "sector": ["Unknown"] * len(symbols),
                    "listing_date": [None] * len(symbols),
                    "ceo_name": [None] * len(symbols),
                }
            )

    def list_symbols(self) -> List[str]:
        companies = self.get_companies()
        return sorted(companies["symbol"].dropna().astype(str).str.upper().unique().tolist())

    def list_sectors(self) -> List[str]:
        companies = self.get_companies()
        sectors = companies["sector"].dropna().astype(str).str.strip()
        sectors = sectors[sectors != ""]
        return sorted(sectors.unique().tolist())

    def symbols_by_sector(self) -> Dict[str, List[str]]:
        companies = self.get_companies().copy()
        if companies.empty:
            return {}
        companies["sector"] = companies["sector"].fillna("Unknown")
        companies["symbol"] = companies["symbol"].astype(str).str.upper()
        grouped = companies.groupby("sector")["symbol"].apply(list)
        return {sector: sorted(symbols) for sector, symbols in grouped.items()}

    def get_symbol_history(self, symbol: str, min_rows: int = 60) -> pd.DataFrame:
        normalized_symbol = symbol.upper()
        with self._connect() as conn:
            if self._table_exists(conn, "stock_prices"):
                query = """
                SELECT time AS "Date", close AS "Close", volume AS "Volume"
                FROM stock_prices
                WHERE symbol = %s
                ORDER BY time;
                """
                df = pd.read_sql_query(query, conn, params=(normalized_symbol,))
            else:
                table_name = self._resolve_legacy_symbol_table(conn, normalized_symbol)
                if table_name is None:
                    return pd.DataFrame(columns=["Date", "Close", "Volume"])

                query = sql.SQL(
                    "SELECT time AS \"Date\", close AS \"Close\", volume AS \"Volume\" FROM {} ORDER BY time"
                ).format(sql.Identifier(table_name))
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                print(f"Số lượng cột thực tế: {len(rows[0]) if rows else 0}")
                print(f"Dữ liệu mẫu: {rows[0] if rows else 'Rỗng'}")
                df = pd.DataFrame(rows, columns=["Date", "Close", "Volume"])

        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        if len(df) < min_rows:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
        return df

    def get_sector_history(self, sector: str, min_rows: int = 60) -> pd.DataFrame:
        by_sector = self.symbols_by_sector()
        symbols = by_sector.get(sector, [])
        if not symbols:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])

        frames = []
        for symbol in symbols:
            symbol_df = self.get_symbol_history(symbol, min_rows=1)
            if symbol_df.empty:
                continue
            symbol_df = symbol_df.copy()
            symbol_df["symbol"] = symbol
            frames.append(symbol_df)

        if not frames:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])

        merged = pd.concat(frames, ignore_index=True)
        sector_df = (
            merged.groupby("Date", as_index=False).agg({"Close": "mean", "Volume": "sum"})
            .sort_values("Date")
        )
        if len(sector_df) < min_rows:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
        return sector_df

    def _resolve_legacy_symbol_table(self, conn, symbol: str) -> Optional[str]:
        symbol_lower = symbol.lower()
        if self._table_exists(conn, symbol_lower):
            return symbol_lower
        if self._table_exists(conn, symbol):
            return symbol
        return None

    def _list_legacy_symbol_tables(self, conn) -> List[str]:
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        with conn.cursor() as cur:
            cur.execute(query)
            all_tables = [row[0] for row in cur.fetchall()]

        excluded = {
            "company_info",
            "stock_prices",
            "schema_migrations",
            "alembic_version",
        }
        return [
            table_name
            for table_name in all_tables
            if table_name not in excluded and table_name.replace("_", "").isalnum()
        ]
