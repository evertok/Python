import tushare as ts
import time
import datetime
import pandas as pd


def getdata():
    codes = ts.get_stock_basics()
    codelist = pd.DataFrame()
    print("股票代码" + " " * 5 + "股票名称" + " " * 10)
    tmp = {"code": [], "name": []}
    for i in range(len(codes)):
        if codes.index[i].startswith("3") or codes.name[i].startswith(
                "*ST") or codes.name[i].startswith("N"):
            continue
        tmp["code"].append(codes.iloc[i].name)
        tmp["name"].append(codes.iloc[i]["name"])

    codelist = pd.DataFrame(tmp, index=tmp["code"])
    # for i in range(0, len(codelist)):
    #    print(codelist.iloc[i]["code"] + " " * 7 + codelist.iloc[i]["name"])
    currTime = time.localtime()
    today = time.strftime("%Y-%m-%d", currTime)
    now = datetime.datetime.now()
    preday30 = now + datetime.timedelta(days=-200)

    data_d = None
    data_w = None
    # 股票历史数据
    good_data = []
    for codeStr in tmp["code"]:
        data_d = ts.get_hist_data(
            codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='D')
        data_d.dropna(inplace=True)
        if (handleData(data_d)):
            data_w = ts.get_hist_data(
                codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='W')
            data_w.rename(columns={"ma5": "ma25", "ma10": "ma50", "ma20": "ma100"}, inplace=True)  # 修改列名
            if(handleWdata(pd.concat([data_d, data_w], axis=1, join_axes=[data_d.index]))):
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
    if(curr_price > max_high * 0.8):
        return False
    if(min_low1 < sorted1.iloc[i].ma20):
        return False
    return True


def handleWdata(data):
    # 需要上升趋势
    if (data.iloc[1].ma25 <= data.iloc[1].ma50):
        return False
    if (data.iloc[1].ma50 <= data.iloc[1].ma100):
        return False
    return True


getdata()