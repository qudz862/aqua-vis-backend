from flask import Flask,json,jsonify
from flask_pymongo import PyMongo
from flask_cors import *

import numpy as np
import pandas as pd
import math
import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/city_locs")
@cross_origin()
def get_city_locs ():
    app.config["MONGO_URI"] = "mongodb://localhost:27017/city_locs"
    mongo = PyMongo(app)
    collection = mongo.db.city_locs
    data = collection.find({})
    city_locs = []
    sIndex = 0
    for row in data:
        city_locs.append({'index': sIndex, 'name': row['name'], 'lat': row['lat'], 'lng': row['lng'], 'group': row['group']})
        sIndex += 1
    return jsonify(city_locs)

def compute_entropy (hist):
    entropy = 0.0
    for prob in hist:
        if (prob != 0):
            entropy -= prob * math.log(prob, 2)
    return entropy

def compute_mutual_infor (X, Y, bin_num)

# 污染物顺序：PM2.5 PM10 SO2 NO2 CO O3
# infor中给出：每个变量的信息熵、每对变量的最大、最小值、平均值、标注差
@app.route("/air_quality_infor/<space>/<time>")
@cross_origin()
def get_air_quality_infor (space, time):
    air_quality_data = {'data': {}, 'infor': {}}
    spaceJson = json.loads(space)
    timeJson = json.loads(time)

    app.config["MONGO_URI"] = "mongodb://localhost:27017/air_quality_2015"
    mongo = PyMongo(app)
    collection = mongo.db.air_quality_2015
    result = collection.find({'name': {'$in': spaceJson}})
    data = list(result[:])


    max = [-500, -500, -500, -500, -500, -500]
    min = [501, 501, 501, 501, 501, 501]
    interval = [0, 0, 0, 0, 0, 0]
    bin_num = 16
    variable_num = 6
    hist_local = [[0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)]]
    hist_global = [[0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)], [0 for n in range(16)]]
    entropy_local = [0, 0, 0, 0, 0, 0]
    entropy_global = [0, 0, 0, 0, 0, 0]
    mutual_infor = [0 for n in range(15)]
    # print(mutual_infor)

    startSplit = timeJson['start'].split('-', 2)
    endSplit = timeJson['end'].split('-', 2)
    startIndex = (datetime.datetime(int(startSplit[0]), int(startSplit[1]), int(startSplit[2])) - datetime.datetime(2015,1,5)).days
    endIndex = 363 - (datetime.datetime(2016, 1, 3) - datetime.datetime(int(endSplit[0]), int(endSplit[1]), int(endSplit[2]))).days
    item_num = len(spaceJson) * (endIndex - startIndex + 1)
    startDate = datetime.datetime(2015, 1, 5)
    dataMtx = np.zeros([variable_num, item_num])
    for row in data:
        dateData = {}
        for i in range(startIndex, endIndex + 1):
            val_list = row['data'][i]['val']
            dateData[row['data'][i]['date']] = val_list
            for j in range(variable_num):
                val = val_list[j]
                if val > max[j]:
                    max[j] = val
                if val < min[j]:
                    min[j] = val
        air_quality_data['data'][row['name']] = dateData
    for i in range(variable_num):
        interval[i] = (max[i] - min[i]) / bin_num
    # print(max)
    # print(min)
    # print(interval)
    # 计算信息熵 & 互信息
    for row in data:
        for i in range(startIndex, endIndex + 1):
            val_list = row['data'][i]['val']
            bin_global_list = row['data'][i]['bin_global']
            for j in range(variable_num):
                bin_local = math.floor((val_list[j]-min[j]) / interval[j])
                # print(bin_local,val_list[j], interval[j])
                if (bin_local == 16):
                    bin_local -= 1
                bin_global = bin_global_list[j]
                hist_local[j][bin_local] += 1
                hist_global[j][bin_global] += 1
    for i in range(variable_num):
        for j in range(bin_num):
            hist_local[i][j] = 1.0 * hist_local[i][j] / item_num
            hist_global[i][j] = 1.0 * hist_global[i][j] / item_num
    print(hist_local)
    print(hist_global)

    for i in range(variable_num):
        entropy_local[i] = compute_entropy(hist_local[i])
        entropy_global[i] = compute_entropy(hist_global[i])

    print(entropy_local)
    print(entropy_global)
    # print(air_quality_data)


    return air_quality_data

    # for city in spaceJson:
    #     file_path = './data/air_quality/' + city + '.csv'
    #     df = pd.read_csv(file_path, header=None)
    #     startSplit = timeJson['start'].split('-', 2)
    #     endSplit = timeJson['end'].split('-', 2)
    #     startIndex = (datetime.datetime(int(startSplit[0]), int(startSplit[1]), int(startSplit[2])) - datetime.datetime(2015,1,5)).days
    #     endIndex = 363 - (datetime.datetime(2016,1,3) - datetime.datetime(int(endSplit[0]), int(endSplit[1]), int(endSplit[2]))).days
    #     air_quality_data['data'][city] = {}
    #     startDate = datetime.datetime(2015,1,5)
    #     dateData = {}
    #     for i in range(startIndex, endIndex + 1):
    #         date = (startDate + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
    #         dateData[date] = df.iloc[i].tolist()
    #         # air_quality_data[city][date] = df[i].tolist()
    #     air_quality_data['data'][city] = dateData

    # 计算信息熵和互信息
    # return air_quality_data

# max = [500.0, 500.0, 177.917, 155.333, 166.583, 201.5]
    # min = [3.16667, 5.83333, 1.25, 2.0, 1.0, 1.125]
    # interval = [31.052083125, 30.885416875, 11.0416875, 9.5833125, 10.3489375, 12.5234375]
# for j in range(6):
            #     val = df.iloc[i].tolist()[j]
            #     if val > max[j]:
            #         max[j] = val
            #     if val < min[j]:
            #         min[j] = val



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1002)
