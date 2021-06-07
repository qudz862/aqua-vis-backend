from flask import Flask,json,jsonify
from flask_pymongo import PyMongo

import os
import math
import csv
import json
import datetime
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def read_city_locs ():
    # 连接mongodb数据库
    app.config["MONGO_URI"] = "mongodb://localhost:27017/city_locs"
    mongo = PyMongo(app)
    collection = mongo.db.city_locs

    file_path = './data/cities.csv'
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        data = {'name': row['NAME'], 'lat': row['lat'], 'lng': row['lng'], 'group': row['group']}
        # print(data)
        collection.insert_one(data)

def walkFiles (path):
    fileNameList = []
    for root, dirs, files in os.walk(path):
        for f in files:
            rootPath = root + '/'
            fileNameList.append(os.path.join(rootPath, f))
    return fileNameList, files

# 污染物顺序：PM2.5 PM10 SO2 NO2 CO O3
def read_air_quality ():
    app.config["MONGO_URI"] = "mongodb://localhost:27017/air_quality_2015"
    mongo = PyMongo(app)
    collection = mongo.db.air_quality_2015
    bin_num = 16

    fileNameList, pureNameList = walkFiles('./data/air_quality/')
    max_global = 500.0
    min_global = 1.0
    interval_global = (max_global - min_global) / bin_num

    # max = [500.0, 500.0, 177.917, 155.333, 166.583, 201.5]
    # min = [3.16667, 5.83333, 1.25, 2.0, 1.0, 1.125]
    # interval = [31.052083125, 30.885416875, 11.0416875, 9.5833125, 10.3489375, 12.5234375]

    for fIndex, file in enumerate(fileNameList):
        df = pd.read_csv(file, header=None)
        city_data = {}
        fileName = pureNameList[fIndex].split('.',1)[0]
        city_data['name'] = fileName
        startDate = datetime.datetime(2015, 1, 5)
        dateData = []
        for i in range(0, 364):
            date = (startDate + datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            val_list = df.iloc[i].tolist()
            bin_global_list = []
            for j in range(6):
                bin_global = math.floor((val_list[j]-min_global) / interval_global)
                bin_global_list.append(bin_global)
            dateData.append({'date': date, 'val': val_list, 'bin_global': bin_global_list})
        city_data['data'] = dateData
        collection.insert_one(city_data)

if __name__ == '__main__':
    # read_city_locs()
    read_air_quality()
