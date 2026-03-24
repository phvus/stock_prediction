from __future__ import annotations

import pandas as pd
from vnstock import Quote
from pathlib import Path

class ApiRepository:
    """Data access layer for stock data using VNStock API directly."""

    def test_connection(self) -> dict:
        return {
            "status": "ok",
            "database": "VNStock API (Realtime)",
            "version": "latest",
        }

    def get_schema_mode(self) -> str:
        return "api"

    def get_companies(self) -> pd.DataFrame:
        project_root = Path(__file__).resolve().parents[1]
        vn100_path = project_root / "VN100.txt"
        
        symbols = []
        if vn100_path.exists():
            symbols = [line.strip().upper() for line in vn100_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            symbols = ["FPT", "VCB", "VNM", "SSI"]
            
        return pd.DataFrame({
            "symbol": symbols,
            "sector": ["API Category"] * len(symbols),
            "listing_date": [None] * len(symbols),
            "ceo_name": ["VnStock API"] * len(symbols)
        })

    def list_symbols(self) -> list[str]:
        companies = self.get_companies()
        return sorted(companies["symbol"].astype(str).str.upper().unique().tolist())

    def list_sectors(self) -> list[str]:
        return ["API Category"]

    def symbols_by_sector(self) -> dict[str, list[str]]:
        return {"API Category": self.list_symbols()}

    def get_symbol_history(self, symbol: str, min_rows: int = 60) -> pd.DataFrame:
        from datetime import datetime, timedelta
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        try:
            quote = Quote(symbol=symbol, source="VCI")
            history = quote.history(start=start_date, end=end_date, interval="1d")
        except Exception:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
            
        if history.empty:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
            
        df = history.copy()
        if "time" in df.columns:
            df = df.rename(columns={"time": "Date", "close": "Close", "volume": "Volume"})
            
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
            
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        
        if len(df) < min_rows:
            return pd.DataFrame(columns=["Date", "Close", "Volume"])
            
        return df[["Date", "Close", "Volume"]]

    def get_sector_history(self, sector: str, min_rows: int = 60) -> pd.DataFrame:
        # Returning empty to encourage using single Company mode for API
        return pd.DataFrame(columns=["Date", "Close", "Volume"])
