import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
import datetime
'''
显示蜡烛图
'''


def pandas_candlestick_ohlc(stock_data, otherseries=None):
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    dayFormatter = DateFormatter("%d")
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    weekFormatter = None
    weekFormatter = DateFormatter("%b %d")
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    ax.grid(True)
    stock_array = np.array(stock_data.reset_index()[["date", "open", "high", "low", "close"]])
    tmp = []
    for i in stock_array[:, 0]:
        tmp.append(datetime.datetime.strptime(i, "%Y-%m-%d"))
    stock_array[:, 0] = date2num(tmp)

    candlestick_ohlc(ax, stock_array, colorup='red', colordown='green', width=0.4)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.show()
