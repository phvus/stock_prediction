from vnstock import Listing, Quote, Company
from pandas import DataFrame
from datetime import datetime
import pandas as pd
import os
import dotenv
import psycopg2

dotenv.load_dotenv()
with open('VN100.txt', 'r') as file:
    List = file.read().splitlines()
def info_company(symbol):
    company_info_VCI = Company(symbol=symbol,source='VCI')
    company_info_KBS = Company(symbol = symbol, source = 'KBS')
    company_VCI = company_info_VCI.overview()[['symbol', 'icb_name2']]
    company_KBS = company_info_KBS.overview()[['listing_date','ceo_name']]
    company=company_VCI.reset_index(drop=True).join(company_KBS.reset_index(drop=True))
    return company

def history_quote(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%d%m%Y').strftime('%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%d%m%Y').strftime('%Y-%m-%d')
    quote=Quote(symbol = symbol, source = 'VCI')
    history_quote = quote.history(start=start_date, end=end_date, interval = '1d')
    history_quote['symbol'] = symbol
    percent_change= ((history_quote['close'] - history_quote['close'].shift(1))/history_quote['close'].shift(1))*100
    history_quote['percent_change'] = round(percent_change, 2)
    if not history_quote['percent_change'].empty:
        history_quote.loc[0, 'percent_change'] = 0
    return history_quote

conn = psycopg2.connect(
    user = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    host = os.getenv("DB_HOST"),
    port = os.getenv("DB_PORT"),
    dbname = os.getenv("DB_NAME")
)

cur = conn.cursor()
insert_query = """
INSERT INTO public.company_info (symbol, icb_name2, listing_date, ceo_name)
VALUES (%s, %s, %s, %s)
ON CONFLICT (symbol) DO NOTHING;
"""

for symbol in List:
    Create_table = f"""
    CREATE TABLE IF NOT EXISTS {symbol}(
        symbol VARCHAR(255),
        time DATE PRIMARY KEY,
        open FLOAT,  
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT,
        percent_change FLOAT
    );"""
    cur.execute(Create_table)
    conn.commit()
    insert_query = f"""
    INSERT INTO {symbol} (symbol, time, open, high, low, close, volume, percent_change)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (time) DO NOTHING;
    """
    history_quote_data = history_quote(symbol, '01012026', '01022026')
    cur.executemany(insert_query, history_quote_data[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume', 'percent_change']].values.tolist())
    conn.commit()

cur.close()
conn.close()