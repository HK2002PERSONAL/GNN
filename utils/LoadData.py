import os
import connectorx as cx
import pandas as pd

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


def load_delimit_stock_return():
    conn = "mysql://kai_hu%40service%3Aocean:TDk9Fe0D@192.168.55.161:2883/SmartQuant"

    ret_sql = """
    SELECT
        SecuCode, DataDate, Return1d, Return5d, Return20d, InnerCode
    FROM ForecastReturn
    """

    stock_ret = cx.read_sql(conn, ret_sql).set_index(["DataDate", "SecuCode"]).sort_index()

    con_sql = """
    SELECT
        SecuCode, DataDate, DailyReturn, HighPrice, LowPrice, ClosePrice, PrevClosePrice, IfSpecialTrade, ListedSector
    FROM ReturnDaily
    WHERE ifTradingDay = 1
    """

    condition = cx.read_sql(conn, con_sql).set_index(["SecuCode", "DataDate"]).sort_index()

    condition['UnQualified'] = condition["IfSpecialTrade"]
    # 主板 1
    condition.loc[
        (condition['ListedSector']==1) & 
        (condition['DailyReturn'] >=  0.095) & 
        (condition['DailyReturn'] <= -0.095), 'UnQualified'] = 1
    # 创业板 6
    # 改制前：~2020-08-21
    condition.loc[
        (
            (condition['ListedSector']==6) & 
            (condition.index.get_level_values('DataDate') <= pd.to_datetime("2020-08-21"))
        ) & (
            (condition['DailyReturn'] >=  0.095) & 
            (condition['DailyReturn'] <= -0.095)
        ), 'UnQualified'] = 1
    # 改制前：2020-08-24~
    condition.loc[
        (
            (condition['ListedSector']==6) & 
            (condition.index.get_level_values('DataDate') >= pd.to_datetime("2020-08-24"))
        ) & (
            (condition['DailyReturn'] >=  0.195) &
            (condition['DailyReturn'] <= -0.195)
        ), 'UnQualified'] = 1
    # 科创板 7
    condition.loc[
        (condition['ListedSector']==7) & 
        (condition['DailyReturn'] >=  0.195) &
        (condition['DailyReturn'] <= -0.195), 'UnQualified'] = 1
    # 删除T+1涨跌停的数据
    condition['UnQualified'] = condition.groupby("SecuCode")['UnQualified'].shift(-1).ffill()
    condition = condition.swaplevel().sort_index()
    
    stock_ret = stock_ret.join(condition['UnQualified'], how="inner")

    stock_ret = stock_ret.loc[stock_ret['UnQualified'] != 1, ['Return1d', 'Return5d', 'Return20d']]

    stock_ret.to_parquet("data/label/stock_ret_delimit.pqt")


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
    load_delimit_stock_return()