# -*- coding: UTF-8 -*-
import time
import calendar

currTime = time.localtime()
print(time.strftime("%Y-%m-%d %H:%M:%S", currTime))

cal = calendar.month(currTime[0], currTime[1])
print(cal)

decodeStr = "gbk"
txtFile = open("C:\\Users\\Administrator\\Desktop\\tmp\\test.txt", "w", encoding=decodeStr)
txtFile.write("icallcar\n")
txtFile.close() 


