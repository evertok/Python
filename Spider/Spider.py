# coding=utf-8
import urllib.request
import re
import tool
import os
import http.cookiejar


# 爬虫抓取
class Spider:

    # 页面初始化
    def __init__(self, keyword):
        self.siteURL = "http://www.ifeng.com/"   # 'https://www.baidu.com/s'
        self.tool = tool.Tool()
        self.keyword = keyword

    # 获取索引页面的内容
    def getPage(self, pageIndex):
        url = self.siteURL + "?pn=" + str(pageIndex) + "&wd=" + str(
            self.keyword)
        head = {
            'Connection':
            'Keep-Alive',
            'Accept':
            'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language':
            'zh-CN,zh;q=0.8',
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
            "Host": "www.baidu.com"
        }

        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
        header = []
        for key, value in head.items():
            elem = (key, value)
            header.append(elem)
        opener.addheaders = header

        uop = opener.open(url, timeout=1000)
        data = uop.read()

        return data.decode("utf-8")

    # 获取索引界面所有的信息，list格式
    def getContents(self, pageIndex):
        page = self.getPage(pageIndex)
        pattern = re.compile(
            '<div .*?result c-container "><h3.*?<a .*?data-click href="(.*?)">(.*?)<em>(.*?)</em></a></h3></div>',
            re.S)
        items = re.findall(pattern, page)
        contents = []
        for item in items:
            contents.append([item[0], item[1], item[2]])
        return contents

    # 获取关键字详情页面
    def getDetailPage(self, infoURL):
        response = urllib.request.urlopen(infoURL)
        return response.read().decode('utf-8')

    # 保存内容
    def saveFile(self, item, fileName):
        data = ((item[1] + item[2], item[0]))
        f = open(fileName, 'wb')
        f.write(data)
        f.close()

    # 创建新目录
    def mkdir(self, path):
        path = path.strip()
        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            return False

    # 将一页的信息保存起来
    def savePageInfo(self, pageIndex):
        # 获取第一页的列表
        contents = self.getContents(pageIndex)
        for item in contents:
            # item[0]个人详情URL,item[1]头像URL,item[2]姓名,item[3]年龄,item[4]居住地
            print("%s%s%s" % (u"发现一条匹配内容：", item[1], item[2], u",网址为",
                              item[0]))

            self.mkdir("D:\\test\\" + self.keyword)
            # 保存信息到文件
            self.saveFile(item, "D:\\test\\", self.keyword + ".csv")

    # 传入起止页码，获取匹配内容
    def savePagesInfo(self, start):
        print("%s%s%s" % (u"正在寻找第", start, u"页，看是否有匹配内容"))
        self.savePageInfo(start)


spider = Spider("python")
spider.savePagesInfo(0)
