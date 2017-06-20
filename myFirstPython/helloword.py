# -*- coding: UTF-8 -*-
import time
"""
import calendar
"""
currTime = time.time()
# print(time.strftime("%Y-%m-%d %H:%M:%S", currTime))
arr1 = []
for i in range(1, 2000000):
    arr1.append(i)
currTime1 = time.time()
print(currTime1 - currTime)
currTime = time.time()
arr2 = [i for i in range(1, 2000000)]
currTime1 = time.time()
print(currTime1 - currTime)
print(currTime)
'''
cal = calendar.month(currTime[0], currTime[1])
print(cal)

decodeStr = "gbk"
txtFile = open("C:\\Users\\Administrator\\Desktop\\tmp\\test.txt", "w", encoding=decodeStr)
txtFile.write("icallcar\n")
txtFile.close() 

'''
