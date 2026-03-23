from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import time
from typing import List

import pandas as pd
import psycopg2
from psycopg2 import sql
from vnstock import Company, Quote, change_api_key


def load_env_file() -> None:
    """Load key=value pairs from .env into environment if variables are unset."""
    env_path = Path(__file__).resolve().parent / ".env"
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


def configure_vnstock_api_key() -> None:
    """Apply API key for vnstock to avoid guest-rate limits."""
    api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
    if not api_key:
        return

    if not change_api_key(api_key):
        print("[WARN] VNSTOCK_API_KEY is set but could not be applied.")
        return

    masked = f"{api_key[:8]}***{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"[INFO] vnstock API key configured: {masked}")


def resolve_symbols_path(path: Path) -> Path:
    """Resolve symbols file from cwd, script dir, and common project subfolders."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        path,
        Path.cwd() / path,
        script_dir / path,
        script_dir / "stock_prediction" / path.name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    tried = "\n - " + "\n - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"Symbol list file not found for input '{path}'. Tried:{tried}\n"
        "Use --symbols-file stock_prediction/VN100.txt or place VN100.txt in project root."
    )


def load_symbols(path: Path) -> List[str]:
    resolved_path = resolve_symbols_path(path)
    return [
        line.strip().upper()
        for line in resolved_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    tokens = [
        "rate limit",
        "giới hạn api",
        "too many requests",
        "429",
    ]
    return any(token in msg for token in tokens)


def get_retry_delay_seconds(exc: BaseException, default: int = 20) -> int:
    msg = str(exc)
    # vnstock often returns text like "Chờ 11 giây để tiếp tục"
    match = re.search(r"(\d+)\s*(giây|seconds?)", msg, flags=re.IGNORECASE)
    if match:
        return max(int(match.group(1)) + 2, 5)
    return default


def info_company(symbol: str) -> pd.DataFrame:
    company_info_vci = Company(symbol=symbol, source="VCI")
    company_info_kbs = Company(symbol=symbol, source="KBS")

    company_vci = company_info_vci.overview()[["symbol", "icb_name2"]]
    company_kbs = company_info_kbs.overview()[["listing_date", "ceo_name"]]
    return company_vci.reset_index(drop=True).join(company_kbs.reset_index(drop=True))


def history_quote(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    quote = Quote(symbol=symbol, source="VCI")
    history = quote.history(start=start_date, end=end_date, interval="1d")
    if history.empty:
        return history

    history["symbol"] = symbol
    percent_change = ((history["close"] - history["close"].shift(1)) / history["close"].shift(1)) * 100
    history["percent_change"] = percent_change.round(2)
    history.loc[history.index[0], "percent_change"] = 0
    return history


def ensure_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS public.company_info (
            symbol VARCHAR(32) PRIMARY KEY,
            icb_name2 VARCHAR(255),
            listing_date DATE,
            ceo_name VARCHAR(255)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS public.stock_prices (
            symbol VARCHAR(32) NOT NULL,
            time DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            percent_change FLOAT,
            PRIMARY KEY (symbol, time)
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time ON public.stock_prices (symbol, time);")


def upsert_company(cur, company_df: pd.DataFrame) -> None:
    if company_df.empty:
        return

    payload = company_df[["symbol", "icb_name2", "listing_date", "ceo_name"]].values.tolist()
    cur.executemany(
        """
        INSERT INTO public.company_info (symbol, icb_name2, listing_date, ceo_name)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE
        SET
            icb_name2 = EXCLUDED.icb_name2,
            listing_date = EXCLUDED.listing_date,
            ceo_name = EXCLUDED.ceo_name;
        """,
        payload,
    )


def upsert_normalized_prices(cur, history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return

    df = history_df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    payload = df[["symbol", "time", "open", "high", "low", "close", "volume", "percent_change"]].values.tolist()
    cur.executemany(
        """
        INSERT INTO public.stock_prices (symbol, time, open, high, low, close, volume, percent_change)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, time) DO UPDATE
        SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            percent_change = EXCLUDED.percent_change;
        """,
        payload,
    )


def upsert_legacy_symbol_table(cur, symbol: str, history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return

    table_name = symbol.lower()
    create_stmt = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {} (
            symbol VARCHAR(32),
            time DATE PRIMARY KEY,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            percent_change FLOAT
        );
        """
    ).format(sql.Identifier(table_name))
    cur.execute(create_stmt)

    insert_stmt = sql.SQL(
        """
        INSERT INTO {} (symbol, time, open, high, low, close, volume, percent_change)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time) DO UPDATE
        SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            percent_change = EXCLUDED.percent_change;
        """
    ).format(sql.Identifier(table_name))

    df = history_df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    payload = df[["symbol", "time", "open", "high", "low", "close", "volume", "percent_change"]].values.tolist()
    cur.executemany(insert_stmt, payload)


def import_single_symbol(
    symbol: str,
    conn,
    start_date: datetime,
    end_date: datetime,
    legacy_tables: bool = False,
    max_attempts: int = 8,
) -> dict:
    """Import a single symbol into PostgreSQL. Returns a status dict.

    Can be called from the dashboard or CLI.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            company_df = info_company(symbol)
            history_df = history_quote(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if history_df.empty:
                return {"symbol": symbol, "status": "skip", "message": "no quote history", "rows": 0}

            with conn.cursor() as cur:
                upsert_company(cur, company_df)
                upsert_normalized_prices(cur, history_df)
                if legacy_tables:
                    upsert_legacy_symbol_table(cur, symbol, history_df)
            conn.commit()
            return {"symbol": symbol, "status": "ok", "message": f"{len(history_df)} rows", "rows": len(history_df)}
        except BaseException as exc:
            conn.rollback()
            if is_rate_limit_error(exc) and attempt < max_attempts:
                wait_seconds = get_retry_delay_seconds(exc)
                time.sleep(wait_seconds)
                continue

            return {"symbol": symbol, "status": "error", "message": str(exc), "rows": 0}

    return {"symbol": symbol, "status": "error", "message": "max retries exceeded", "rows": 0}


def import_symbols_to_db(
    symbols: List[str],
    conn,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    legacy_tables: bool = False,
    progress_callback=None,
) -> List[dict]:
    """Import a list of symbols into PostgreSQL.

    Args:
        symbols: list of stock ticker symbols
        conn: psycopg2 connection
        start_date: start date (default: 730 days ago)
        end_date: end date (default: today)
        legacy_tables: also create per-symbol legacy tables
        progress_callback: optional callable(index, total, result_dict)

    Returns:
        list of per-symbol result dicts
    """
    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - timedelta(days=730)

    with conn.cursor() as cur:
        ensure_schema(cur)
    conn.commit()

    results: List[dict] = []
    for idx, symbol in enumerate(symbols):
        result = import_single_symbol(symbol, conn, start_date, end_date, legacy_tables=legacy_tables)
        results.append(result)
        if progress_callback:
            progress_callback(idx, len(symbols), result)

    return results


def get_default_symbols() -> List[str]:
    """Load symbols from VN100.txt relative to this script."""
    script_dir = Path(__file__).resolve().parent
    vn100_path = script_dir / "VN100.txt"
    if vn100_path.exists():
        return [
            line.strip().upper()
            for line in vn100_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def main() -> None:
    load_env_file()
    configure_vnstock_api_key()

    parser = argparse.ArgumentParser(description="Import VN stocks into PostgreSQL (company + prices).")
    parser.add_argument("--symbols-file", default="VN100.txt", help="Path to symbol list file")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 730 days ago)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--legacy-tables",
        action="store_true",
        help="Also populate legacy per-symbol tables for backward compatibility",
    )
    args = parser.parse_args()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.today()
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else end_date - timedelta(days=730)

    symbols = load_symbols(Path(args.symbols_file))
    print(f"Loading {len(symbols)} symbols from {args.symbols_file}")
    print(f"Date range: {start_date.date()} -> {end_date.date()}")

    conn = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
    )

    def cli_progress(idx, total, result):
        status = result["status"].upper()
        print(f"[{status}] {result['symbol']}: {result['message']}")

    import_symbols_to_db(
        symbols,
        conn,
        start_date=start_date,
        end_date=end_date,
        legacy_tables=args.legacy_tables,
        progress_callback=cli_progress,
    )

    conn.close()
    print("Import complete.")


if __name__ == "__main__":
    main()
