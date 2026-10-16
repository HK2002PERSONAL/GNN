import os
import connectorx as cx
# import pandas as pd

def load_stock_return():
    conn = "mysql://kai_hu%40service%3Aocean:TDk9Fe0D@192.168.55.161:2883/SmartQuant"

    sql = """
    SELECT
        SecuCode, DataDate, Return1d, Return5d, Return20d, InnerCode
    FROM ForecastReturn
    """

    cx.read_sql(conn, sql).set_index(["DataDate", "SecuCode"]).sort_index().to_parquet(os.path.join('data', 'label', "stock_ret.pqt"))

def load_stock_data():
    conn = "mysql://kai_hu%40service%3Aocean:TDk9Fe0D@192.168.55.161:2883/SmartQuant"

    sql = """
    SELECT
        SecuCode, DataDate, DailyReturn, OpenPrice, HighPrice, LowPrice, ClosePrice, PrevClosePrice,
        TurnoverVolume, TurnoverValue, MarketCap, FloatMarketCap, IfSuspended, IfSpecialTrade, SecuMarket, IndustryCodeNew, ListedSector
    FROM ReturnDaily
    WHERE ifTradingDay = 1
    """

    cx.read_sql(conn, sql).set_index(["DataDate", "SecuCode"]).sort_index().to_parquet(os.path.join('data', 'raw_data', "stock_data.pqt"))


def load_stock_edge():
    conn = "mysql://kai_hu%40service%3Aocean:TDk9Fe0D@192.168.55.161:2883/SmartQuant"

    sql = """
    SELECT
        SecuCode, DataDate, DailyReturn, FloatMarketCap, IndustryCodeNew, SecondIndustryCodeNew, IfSuspended, IfSpecialTrade
    FROM ReturnDaily
    WHERE ifTradingDay = 1
    """

    cx.read_sql(conn, sql).set_index("").to_parquet(os.path.join('data', 'raw_data', "stock_edge.pqt"))


def load_risk_data():
    conn = "mysql://kai_hu%40service%3Aocean:TDk9Fe0D@192.168.55.161:2883/SmartQuant"

    sql = """
    SELECT
        DataDate, SecuCode, FactorName, Value
    FROM riskmodel.BarraBaseFactorExpo
    WHERE StockPool = 'all'
    """

    data = cx.read_sql(conn, sql)#
    data = data.set_index(['DataDate', 'SecuCode', 'FactorName'])['Value'].unstack()

    data.to_parquet(os.path.join('data', 'raw_data', "stock_risk.pqt"))

if __name__ == "__main__":
    load_stock_return()