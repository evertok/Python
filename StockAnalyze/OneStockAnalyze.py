# coding = utf-8
import tushare as ts
import sys


class OneStockAnalyze():
    def __init__(self, code=None):
        self.code = code

    """
    获取单个股票数据
    """

    def GetStockData(self, code=None, startTime=None):
        if code is None:
            return None
        try:
            return ts.get_sina_dd(code=code, date=startTime)
        except:
            print(sys.exc_info()[0])
            return None

    """
    分析股票数据
    """

    def AnalyzeData(self, code, start):

        data = self.GetStockData(code, start.strftime("%Y-%m-%d"))
        if data is None:
            pass
        else:
            if (len(data) < 50):
                print(
                    data.sort_values(
                        ["time", "volume"], ascending=[False, False]))
            else:
                sorted = data.sort_values(["type", "volume"], ascending=[False, False])
                for i in sorted.values:
                    print(i)

    '''
    私有方法
    '''
    def __str__(self):
        return self.code
