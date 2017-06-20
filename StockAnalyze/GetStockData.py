import tushare as ts
import time
code = "600029"
# 股票历史数据
data = ts.get_hist_data(code, start='2016-01-01')
print(data)
currTime = time.localtime()
today = time.strftime("%Y-%m-%d", currTime)
# 成交量大于vol的股票
df = ts.get_sina_dd(code, date=today, vol=400)
print(df)
# 获取所有股票数据
allStock = ts.get_stock_basics()
print(allStock)
# 获取即时财经新闻
news = ts.get_latest_news()
print(news)