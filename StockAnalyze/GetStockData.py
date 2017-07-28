# coding = utf-8
import tushare as ts
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from OneStockAnalyze import OneStockAnalyze


def getdata():
    codes = ts.get_stock_basics()
    codelist = pd.DataFrame()
    print("股票代码" + " " * 5 + "股票名称" + " " * 10)
    result = {"code": [], "name": []}
    yearpre2 = datetime.datetime.now() - datetime.timedelta(days=712)
    yearpre2 = int(yearpre2.strftime("%Y%m%d"))  # 上市时间要超过两年
    '''
    iloc速度巨慢，是iterrows的3倍
    print(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"))
    for code in codes.index:
        if code.startswith("3") or codes.ix[code]["name"].startswith(
                "*ST") or codes.ix[code]["name"].startswith("N") or codes.ix[code]["timeToMarket"] > yearpre2:
            continue
        tmp["code"].append(code)
        tmp["name"].append(codes.ix[code]["name"])
    print(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"))
    '''
    tmpCode = result["code"]
    tmpName = result["name"]
    for val in codes.iterrows():
        if val[0].startswith("3") or val[1]["name"].startswith(
                "*ST") or val[1]["name"].startswith(
                    "N") or val[1]["timeToMarket"] > yearpre2:
            continue
        tmpCode.append(val[0])
        tmpName.append(val[1]["name"])

    # for i in range(0, len(codelist)):
    #    print(codelist.iloc[i]["code"] + " " * 7 + codelist.iloc[i]["name"])
    now = datetime.datetime.now()
    preday30 = now + datetime.timedelta(days=-200)

    data_d = None
    data_w = None
    # 股票历史数据
    good_data = []
    count = 0
    for codeStr in result["code"]:
        count += 1
        print('正在进行第' + str(count) + "个数据处理，代码为：" + codeStr)
        data_d = ts.get_hist_data(
            codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='D')
        if (data_d is None or len(data_d) == 0):
            continue
        if (handleData(data_d)):
            data_w = ts.get_hist_data(
                codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='W')
            data_w.rename(
                columns={"ma5": "ma25",
                         "ma10": "ma50",
                         "ma20": "ma100"},
                inplace=True)  # 修改列名
            if (handleWdata(
                    pd.concat(
                        [data_d, data_w], axis=1, join_axes=[data_d.index]))):
                good_data.append(codeStr)
    # 成交量大于vol(手)的股票
    # df = ts.get_sina_dd(codeStr, date=today, vol=500)
    # print(df)
    # 获取所有股票数据
    # allStock = ts.get_stock_basics()
    # print(allStock)
    # 获取即时财经新闻
    # news = ts.get_latest_news()
    # print(news)


def handleData(data):
    # 需要上升趋势
    i = 0
    if (data.iloc[i].ma5 <= data.iloc[i].ma10):
        return False
    if (data.iloc[i].ma10 <= data.iloc[i].ma20):
        return False
    sorted = data.sort_values(by="high", ascending=False)  # 降序排列

    max_high = sorted.iloc[i].high
    max_date = sorted.iloc[i].name
    sorted = data.sort_values(by="low")  # 升序排列
    min_low = sorted.iloc[i].low
    min_date = sorted.iloc[i].name
    if (min_date > max_date):
        return False
    max_time = datetime.datetime.strptime(max_date, "%Y-%m-%d")
    if (max_time + datetime.timedelta(-5) > datetime.datetime.now()):
        # 如果最高点为最近5天内，则不符合
        return False
    sorted = sorted.sort_index(ascending=False)
    sorted1 = sorted[sorted.iloc[i].name:max_date].sort_values(by="low")
    min_low1 = sorted1.iloc[i].low
    curr_price = sorted.iloc[i].close
    if (curr_price > max_high * 0.8):
        return False
    if (min_low1 < sorted1.iloc[i].ma20):
        return False
    return True


def handleWdata(data):
    # 需要上升趋势
    i = 0
    if (data.iloc[i].ma25 <= data.iloc[i].ma50):
        return False
    if (data.iloc[i].ma50 <= data.iloc[i].ma100):
        return False
    return True


'''
双塔 002481
龙力 002604
光洋 002708


code = "002708"
start = datetime.datetime.now() - datetime.timedelta(200)
data = ts.get_hist_data(code, start.strftime("%Y-%m-%d"))
data = data[::-1]
ShowCandle.pandas_candlestick_ohlc(data)

'''


'''
# getdata()
# 行业
data_industry = ts.get_industry_classified()
data_industry = data_industry[data_industry["c_name"].str.contains('汽车')]

# 地区
data_area = ts.get_area_classified()
data_area = data_area[data_area["area"].str.contains("上海")]
# 概念
data_gn = ts.get_concept_classified()
data_gn = data_gn[data_gn["c_name"].str.contains("特斯拉")]
join_df = data_gn.merge(data_area, on="code", how="inner")

print(ts.get_index())
'''
