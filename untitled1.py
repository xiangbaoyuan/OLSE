# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:22:07 2022

@author: 23618
"""
import json
import requests
from lxml import etree
class WeatherSpider():
    def __init__(self,key):
        self.key = key
        with open('city.json', 'r')as f:
            result = json.load(f)
            num = result[key]
            self.url = 'http://www.weather.com.cn/weather/{}.shtml'.format(num)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
        self.day_list = list()
        self.weather_list = list()
        self.temperature_list = list()
        self.wind_list = list()


    # 1 请求数据
    def get_data(self):
        data = requests.get(self.url,headers = self.headers).content.decode('utf-8')
        # print(data)
        Xpath_data = etree.HTML(data)
        return Xpath_data

    # 2 分析数据
    def analyse_data(self,data):
        self.day_list = data.xpath('//div[@id="7d"]/ul/li/h1/text()')

        self.weather_list = data.xpath('//div[@id="7d"]/ul/li/p/@title')

        self.temperature_list = list()
        count = 0
        for i in range(1,8):
            temperature_1 = data.xpath('//div[@id="7d"]/ul/li[{}]/p[@class="tem"]/span/text()'.format(i))
            temperature_2 = data.xpath('//div[@id="7d"]/ul/li[{}]/p[@class="tem"]/i/text()'.format(i))
            if temperature_1 == []:
                temperature = temperature_2[0]
            else:
                temperature = temperature_1[0] +'/' + temperature_2[0]
            self.temperature_list.append(temperature)


        self.wind_list = list()
        wind = data.xpath('//div[@id="7d"]/ul/li/p[@class="win"]/i/text()')
        for i  in wind:
            x = '风力:' + i
            self.wind_list.append(x)
    # 3 表现数据
    def print_data(self):
        print('的近七天天气如下：')
        print('|{x:^{y}s}|'.format(x='天气',
                                    y=15 - len('天气'.encode('GBK')) + len('天气'))+'{x:^{y}s}\t|'.format(x='气象',
                                    y=15 - len('气象'.encode('GBK')) + len('气象'))+'{x:^{y}s}\t|'.format(x='温度',
                                    y=15 - len('温度'.encode('GBK')) + len('温度'))+'{x:^{y}s}\t|'.format(x='风力',
                                    y=15 - len('风力'.encode('GBK')) + len('风力')))
        for i in range(7):
            print('|{x:^{y}s}\t'.format(x=self.day_list[i],
                                        y=15-len(self.day_list[i].encode('GBK'))+len(self.day_list[i])),end='')
            print('|{x:^{y}s}\t'.format(x=self.weather_list[i],
                                        y=15 - len(self.weather_list[i].encode('GBK')) + len(self.weather_list[i])), end='')
            print('|{x:^{y}s}\t'.format(x=self.temperature_list[i],
                                        y=15 - len(self.temperature_list[i].encode('GBK')) + len(self.temperature_list[i])), end='')
            print('|{x:^{y}s}\t|'.format(x=self.wind_list[i],
                                        y=15 - len(self.wind_list[i].encode('GBK')) + len(self.wind_list[i])))

    

    def Run(self):
        Xpath_data = self.get_data()
        self.analyse_data(Xpath_data)
        self.print_data()

if __name__ == '__main__':
    ans = 0
    # name = input('输出要查城市的名字：')
    try:
        WS = WeatherSpider('北京')
    except:
        print('我已经很努力了，可是还没有找到这个地方')
        ans = 1
    if ans:
        pass
    else:
        WS.Run()


