CREATE TABLE IF NOT EXISTS public.company_info (
    symbol VARCHAR(32) PRIMARY KEY,
    icb_name2 VARCHAR(255),
    listing_date DATE,
    ceo_name VARCHAR(255)
);

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

CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time ON public.stock_prices (symbol, time);
CREATE INDEX IF NOT EXISTS idx_stock_prices_time ON public.stock_prices (time);
