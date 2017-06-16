# coding = utf-8

# 爬虫技术

import urllib.request


# 方法前面要以两个空行来隔开
def getUrl(url):
    "获取页面内容并返回"
    page = urllib.request.urlopen(url)
    html = page.read()
    return html


"以两个空行结束函数"
decodeStr = "utf-8"
baseUrl = "http://192.168.1.200/api"
method = "open/login/sendsmscode"
html = getUrl(baseUrl + method).decode(decodeStr)
print(html)
