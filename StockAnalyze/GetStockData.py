import tushare as ts
import time
import datetime
import pandas as pd


def getdata():
    codes = ts.get_stock_basics()
    codelist = pd.DataFrame()
    print("股票代码" + " " * 5 + "股票名称" + " " * 10)
    tmp = {
        "code": [],
        "name": []
    }
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
    for codeStr in tmp["code"]:
        data_d = ts.get_hist_data(codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='D')
        data_d.dropna(inplace=True)
        handleData(data_d)
        data_w = ts.get_hist_data(codeStr, start=preday30.strftime("%Y-%m-%d"), ktype='W')
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
    max_high = 0
    max_date = ""
    sorted = data.sort_values(by="high", ascending=False)  # 降序排列
    max_high = sorted.iloc[1].high
    max_date = sorted.iloc[1].name
    sorted = data.sort_values(by="low")  # 升序排列
    min_low = sorted.iloc[1].low
    min_date = sorted.iloc[1].name


getdata()