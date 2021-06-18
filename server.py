from flask import Flask,json,jsonify
from flask_pymongo import PyMongo
from flask_cors import *

import numpy as np
import pandas as pd
import math
import datetime
from sklearn.metrics import mutual_info_score
from sklearn.manifold import TSNE
from scipy.spatial import procrustes
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, OPTICS
import copy
import time

from spmf import Spmf

import umap

import subprocess
def vmsp(input_file, output_file, min_supp):
    subprocess.call(["java", "-jar", "spmf.jar", "run", "VMSP", input_file, output_file, str(min_supp),str(5),str(1)])
    lines = []
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
    except:
        print("read_output error")

    # decode
    patterns = []
    for line in lines:
        line = line.strip()
        patterns.append(line.split(" -1 "))
    return patterns

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

all_city_locs = []
cur_air_quality_data = {}
cur_city_list = []
cur_city_num = 0
cur_date_num = 0
cur_cluster_num = [0,0]
cur_global_label_patterns = []

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
    # global all_city_locs
    # all_city_locs = city_locs

    return jsonify(city_locs)

class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        m, n = data.shape
        dist = np.zeros((m, m))
        disti = np.zeros(m)
        distj = np.zeros(m)
        B = np.zeros((m, m))
        for i in range(m):
            dist[i] = np.sum(np.square(data[i] - data), axis=1).reshape(1, m)
        for i in range(m):
            disti[i] = np.mean(dist[i, :])
            distj[i] = np.mean(dist[:, i])
        distij = np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
        lamda, V = np.linalg.eigh(B)
        index = np.argsort(-lamda)[:self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        V_selected = V[:, index]
        Z = V_selected.dot(diag_lamda)
        return Z

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    # print(c_normalized)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    # print(c_normalized, H)
    return H

def compute_entropy (X, bin_num):
    hist = np.histogram(X, bin_num)[0]
    return shan_entropy(hist)

def compute_mutual_infor (X, Y, bin_num):
    c_xy = np.histogram2d(X, Y, bin_num)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def procrustes_m(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

# 污染物顺序：PM2.5 PM10 SO2 NO2 CO O3
# infor中给出：每个变量的信息熵、每对变量的最大、最小值、平均值、标注差
@app.route("/air_quality_infor/<space>/<time>")
@cross_origin()
def get_air_quality_infor (space, time):
    air_quality_data = {'data': {}, 'infor': {}, 'coord': {}}
    spaceJson = json.loads(space)
    global cur_city_list
    cur_city_list = spaceJson
    timeJson = json.loads(time)
    global cur_city_num
    global cur_date_num
    cur_city_num = len(spaceJson)

    app.config["MONGO_URI"] = "mongodb://localhost:27017/air_quality_2015"
    mongo = PyMongo(app)
    collection = mongo.db.air_quality_2015
    result = collection.find({'name': {'$in': spaceJson}})
    data = list(result[:])
    global cur_city_locs
    cur_city_locs = data

    bin_num = 16
    variable_num = 6
    hist_local = np.zeros((6, bin_num))
    hist_global = np.zeros((6, bin_num))
    hist_global_joint = np.zeros((15, bin_num, bin_num))
    hist_local_joint = np.zeros((15, bin_num, bin_num))
    # print(np.array(hist_global_joint).size)
    # print(hist_global_joint)
    entropy_local = [0, 0, 0, 0, 0, 0]
    entropy_global = [0, 0, 0, 0, 0, 0]
    joint_entropy_global = [0 for n in range(15)]
    mutual_infor_local = [0 for n in range(15)]
    mutual_infor_global = [0 for n in range(15)]
    relative_entropy_local = [[ 0 for i in range(variable_num)] for j in range(variable_num)]
    relative_entropy_global = [[ 0 for i in range(variable_num)] for j in range(variable_num)]
    avg_air_quality = [0, 0, 0, 0, 0, 0]
    max_air_quality = [0, 0, 0, 0, 0, 0]

    startSplit = timeJson['start'].split('-', 2)
    endSplit = timeJson['end'].split('-', 2)
    startIndex = (datetime.datetime(int(startSplit[0]), int(startSplit[1]), int(startSplit[2])) - datetime.datetime(2015,1,5)).days
    endIndex = 363 - (datetime.datetime(2016, 1, 3) - datetime.datetime(int(endSplit[0]), int(endSplit[1]), int(endSplit[2]))).days
    time_dim_num = endIndex - startIndex + 1
    item_num = len(spaceJson) * time_dim_num
    startDate = datetime.datetime(2015, 1, 5)
    dataMtx = np.zeros([variable_num, item_num])
    cur_date_num = time_dim_num

    for k, row in enumerate(data):
        dateData = {}
        for i in range(startIndex, endIndex + 1):
            val_list = row['data'][i]['val']
            bin_global_list = row['data'][i]['bin_global']
            dateData[row['data'][i]['date']] = val_list
            for j in range(variable_num):
                dataMtx[j,k*time_dim_num+i-startIndex] = val_list[j]
                bin_global = bin_global_list[j]
                if (bin_global > 15):
                    bin_global = 15
                hist_global[j][bin_global] += 1
                avg_air_quality[j] += val_list[j]
                if val_list[j] > max_air_quality[j]:
                    max_air_quality[j] = val_list[j]
            idx = 0
            for m in range(variable_num):
                for r in range(m+1, variable_num):
                    if bin_global_list[m] > 15:
                        bin_global_list[m] = 15
                    if bin_global_list[r] > 15:
                        bin_global_list[r] = 15
                    hist_global_joint[idx][bin_global_list[m]][bin_global_list[r]] += 1
                    idx += 1
        air_quality_data['data'][row['name']] = dateData
    global cur_air_quality_data
    cur_air_quality_data = air_quality_data['data']
    # print(cur_air_quality_data)
    for i in range(variable_num):
        avg_air_quality[i] = avg_air_quality[i] / item_num
    # 计算信息熵 & 互信息
    for i in range(variable_num):
        hist_local[i] = np.histogram(dataMtx[i], bin_num)[0]
        entropy_local[i] = compute_entropy(dataMtx[i], bin_num)
        entropy_global[i] = shan_entropy(hist_global[i])
    idx = 0
    for i in range(variable_num):
        for j in range(i+1, variable_num):
            hist_local_joint[idx] = np.histogram2d(dataMtx[i], dataMtx[j], bin_num)[0]
            mutual_infor_local[idx] = mutual_info_score(None, None, contingency=hist_local_joint[idx])
            joint_entropy_global[idx] = shan_entropy(hist_global_joint[idx])
            mutual_infor_global[idx] = entropy_global[i] + entropy_global[j] - joint_entropy_global[idx]
            # print(np.array(hist_global_joint[idx]).size)
            # print(np.sum(np.array(hist_global_joint[idx])))
            idx += 1
    idx = 0
    for i in range(variable_num):
        for j in range(i+1, variable_num):
            relative_entropy_global[i][j] = joint_entropy_global[idx] - entropy_global[i]
            relative_entropy_global[j][i] = joint_entropy_global[idx] - entropy_global[j]
            relative_entropy_local[i][j] = shan_entropy(hist_local_joint[idx]) - entropy_local[i]
            relative_entropy_local[j][i] = shan_entropy(hist_local_joint[idx]) - entropy_local[j]
            idx += 1

    air_quality_data['infor']['avg_air_quality'] = avg_air_quality
    air_quality_data['infor']['max_air_quality'] = max_air_quality
    air_quality_data['infor']['entropy_local'] = entropy_local
    air_quality_data['infor']['entropy_global'] = entropy_global
    air_quality_data['infor']['mutual_infor_local'] = mutual_infor_local
    air_quality_data['infor']['mutual_infor_global'] = mutual_infor_global
    air_quality_data['infor']['relative_entropy_local'] = relative_entropy_local
    air_quality_data['infor']['relative_entropy_global'] = relative_entropy_global

    # 对变量的数据分布，用tsne进行降维
    tsne = TSNE(n_components=2, learning_rate=100)
    mds = MyMDS(2)
    local_coord = tsne.fit_transform(hist_local)
    global_coord = tsne.fit_transform(hist_global)
    local_coord_mds = mds.fit(hist_local)
    global_coord_mds = mds.fit(hist_global)

    # d, Z, tform = procrustes_m(local_coord, global_coord)
    air_quality_data['coord']['coord_local'] = local_coord.tolist()
    air_quality_data['coord']['coord_global'] = global_coord.tolist()
    air_quality_data['coord']['coord_local_mds'] = local_coord_mds.tolist()
    air_quality_data['coord']['coord_global_mds'] = local_coord_mds.tolist()
    # print(d, Z, tform)
    # mtx1, mtx2, disparity = procrustes(local_coord, global_coord)
    # air_quality_data['coord']['coord_local'] = mtx1.tolist()
    # air_quality_data['coord']['coord_global'] = mtx2.tolist()
    return air_quality_data


# # 设计一个Canopy类
# class Canopy:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.t1 = 0
#         self.t2 = 0
#     # 设置初始阈值t1 和 t2
#     def setThreshold(self, t1, t2):
#         if t1 > t2:
#             self.t1 = t1
#             self.t2 = t2
#         else:
#             print("t1 needs to be larger than t2!")
#     # 使用欧式距离进行距离计算
#     def euclideanDistance(self, vec1, vec2):
#         return math.sqrt(((vec1 - vec2) ** 2).sum())
#     # 根据当前dataset的长度随机选择一个下标
#     def getRandIndex(self):
#         return np.random.randint(len(self.dataset))
#         # return random.randint(0, len(self.dataset) - 1)
#     # 获取任意两点间距离的平均值
#     def getAvgDistance(self):
#         avgDist = 0
#         cnt = 0
#         for i in range(len(self.dataset)):
#             for j in range(i+1, len(self.dataset)):
#                 avgDist += self.euclideanDistance(self.dataset[i], self.dataset[j])
#                 cnt += 1
#         avgDist = avgDist / cnt
#         return avgDist
#     # 核心算法
#     def clustering(self):
#         if self.t1 == 0:
#             print('Please set the threshold t1 and t2!')
#         else:
#             canopies = []  # 用于存放最终归类的结果
#             while len(self.dataset) != 0:
#                 # 获取一个随机下标
#                 rand_index = self.getRandIndex()
#                 # 随机获取一个中心点，定为P点
#                 current_center = self.dataset[rand_index]
#                 # 初始化P点的canopy类容器
#                 current_center_list = []
#                 # 初始化P点的删除容器
#                 delete_list = []
#                 # 删除随机选择的中心点P
#                 self.dataset = np.delete(self.dataset, rand_index, 0)
#                 for datum_j in range(len(self.dataset)):
#                     datum = self.dataset[datum_j]
#                     # 计算选取的中心点P到每个点之间的距离
#                     distance = self.euclideanDistance(current_center, datum)
#                     if distance < self.t1:
#                         # 若距离小于t1，则将点归入P点的canopy类
#                         current_center_list.append(datum)
#                     if distance < self.t2:
#                         # 若小于t2则归入删除容器
#                         delete_list.append(datum_j)
#                 self.dataset = np.delete(self.dataset, delete_list, 0)
#                 canopies.append((current_center, current_center_list))
#             return canopies
#
# def canopy_kmeans(data):
#     canopy = Canopy(data)
#     canopy.t1 = canopy.getAvgDistance() + 20
#     canopy.t2 = canopy.getAvgDistance() + 10
#     print(canopy.t1, canopy.t2)
#     canopy_result = canopy.clustering()
#     print(len(canopy_result))
#     print(canopy_result[0])
#     return ""

def hierarchical_clustering(data, k):
    h_clustering = AgglomerativeClustering().fit(data)
    labels = h_clustering.labels_
    return labels

def all_points_projection_mds(data):
    mds = MyMDS(2)
    point_locs_mds = mds.fit(data)
    return point_locs_mds

def all_points_projection_tsne(data):
    tsne = TSNE(n_components=2, learning_rate=100)
    point_locs_tsne = tsne.fit_transform(data)
    return point_locs_tsne

def all_points_projection_umap(data):
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.001)
    embedding = reducer.fit_transform(data)
    return embedding

def kmeans(data, k):
    # mkmeans = MiniBatchKMeans(n_clusters=k, random_state = 0, batch_size = 6).fit(data)
    mkmeans = KMeans(n_clusters=k, random_state = 0).fit(data)
    centers = mkmeans.cluster_centers_
    mkmeans.predict(data)
    labels = mkmeans.labels_
    return centers, labels

def optics(data):
    clustering = OPTICS(min_samples=13).fit(data)
    print(clustering.labels_)
    print(max(clustering.labels_.tolist()))
    return clustering.labels_

def st_conn_judge(tmp_dis, j, k, tmp_conn_mtx):
    # 如果该点已经被考虑过，不再重复计算
    if tmp_conn_mtx[j][j] == 0:
        return
    # 判断当前城市在时间上是否连续
    tmp_k_plus = k
    tmp_k_minus = k
    while tmp_dis[j][tmp_k_plus] > 0:
        tmp_dis[j][tmp_k_plus] = 0
        tmp_k_plus += 1
        if tmp_k_plus == tmp_dis.shape[1]:
            break
    while tmp_dis[j][tmp_k_minus] > 0:
        tmp_dis[j][tmp_k_minus] = 0
        tmp_k_minus -= 1
        if tmp_k_minus == -1:
            break
    # 判断当前城市在空间上是否连续，空间上延展的城市需要继续进行时空连续性的判断，是一个迭代的过程（需要写一个函数进行迭代调用）
    tmp_conn_mtx[j][j] = 0
    for i in range(0, tmp_conn_mtx[j].size):
        if tmp_conn_mtx[j][i] == 0:
            continue
        tmp_conn_mtx[j][i] = 0
        tmp_conn_mtx[i][j] = 0
        st_conn_judge(tmp_dis, i, k, tmp_conn_mtx)

@app.route("/clustering_result/<var_flag>/<cluster_num>")
@cross_origin()
def get_clustering_result (var_flag, cluster_num):
    var_flag = json.loads(var_flag)
    cluster_num_list = json.loads(cluster_num)
    # print(cluster_num_list)
    cluster_num_from = cluster_num_list['from']
    cluster_num_to = cluster_num_list['to']
    global cur_air_quality_data
    global cur_city_num
    global cur_date_num
    global cur_cluster_num
    cur_cluster_num = cluster_num_list

    air_quality_data = []
    pure_data = []
    var_num = 0
    for item in var_flag:
        if item == 1:
            var_num += 1
    for key, value in cur_air_quality_data.items():  # for (key,value) in girl_dict.items() 这样加上括号也可以
        for key2, value2 in value.items():
            valList = []
            for i in range(len(value2)):
                if var_flag[i] == 1:
                    valList.append(value2[i])
            item = {'city':key, 'date':key2, 'value':valList, 'label':-1}
            pure_data.append(valList)
            air_quality_data.append(item)
    # print(air_quality_data)
    pure_data = np.array(pure_data)
    pure_data_list = list(pure_data)
    centers_list = []
    labels_list = []
    for i in range(cluster_num_from, cluster_num_to+1):
        centers, labels = kmeans(pure_data, i)
        center_list = list(centers)
        label_list = list(labels)
        centers_list.append(center_list)
        labels_list.append(label_list)
        pure_data_list.extend(center_list)

    # 原始数据和所有的中心点
    pure_data_and_centers = np.array(pure_data_list)

    # 使用umap降维，感觉umap好像好用一点....不加其他的方法了
    start_umap = time.time()
    all_points_locs_umap = all_points_projection_umap(pure_data_and_centers)
    elapsed_umap = time.time() - start_umap
    print(elapsed_umap)

    # OPTICS对参数也很敏感啊，不好用！！！
    # start_optics = time.time()
    # labels = optics(pure_data)
    # elapsed_optics = time.time() - start_optics
    # print(elapsed_optics)
    # cluster_num_int = max(labels.tolist()) + 1

    # start_mds = time.time()
    # all_points_locs_mds = all_points_projection_mds(pure_data_and_centers)
    # elapsed_mds = time.time() - start_mds
    #
    # start_tsne = time.time()
    # all_points_locs_tsne = all_points_projection_tsne(pure_data_and_centers)
    # elapsed_tsne = time.time() - start_tsne

    # 为每条数据赋予聚类标签（数组）

    for i in range(len(air_quality_data)):
        air_quality_data[i]['label'] = []
        for j in range(cluster_num_from, cluster_num_to+1):
            air_quality_data[i]['label'].append(int(labels_list[j-cluster_num_from][i]))

    # 计算浓度和连续性
    # concentration = np.zeros([cluster_num_int])
    # continuity = np.zeros([cluster_num_int])
    concentration = []
    continuity = []
    iaqi_bin_num = 64
    iaqi_distribution = []
    for i in range(cluster_num_from, cluster_num_to+1):
        concentration.append(np.zeros([i]))
        continuity.append(np.zeros([i]))
        iaqi_distribution.append(np.zeros([i, var_num, iaqi_bin_num]))

    # 相关参数：时间段分割大小、关键bin比例（隔离噪音）
    daySep = 7
    cityNum = cur_city_num
    weekNum = math.ceil(1.0 * cur_date_num / daySep)

    for i in range(cluster_num_from, cluster_num_to+1):
        st_distribution = np.zeros([i, cur_city_num, weekNum])
        totalNum = len(air_quality_data)
        for j in range(totalNum):
            cluster_id = air_quality_data[j]['label'][i-cluster_num_from]
            city_id = math.floor(1.0 * j / cur_date_num)
            date_id = j % cur_date_num
            week_id = math.floor(1.0 * date_id / daySep)
            st_distribution[cluster_id][city_id][week_id] += 1

        # 计算浓度 & 构建significant分布
        sBinNum = np.zeros(i)
        sig_st_distribution = np.zeros([i, cur_city_num, weekNum])
        for j in range(i):
            tmp_dis = st_distribution[j].copy()
            itemNum = np.sum(tmp_dis)
            # pRate = math.sqrt(2 * math.log(totalNum, math.e))
            pRate = 0.8
            tmpItemNum = 0
            while tmpItemNum < itemNum * pRate:
                cur_max = np.max(tmp_dis)
                cur_max_pos = np.unravel_index(np.argmax(tmp_dis),tmp_dis.shape)
                sBinNum[j] += 1
                tmpItemNum += cur_max
                sig_st_distribution[j][cur_max_pos] = tmp_dis[cur_max_pos]
                tmp_dis[cur_max_pos] = 0
            concentration[i-cluster_num_from][j] = tmpItemNum / sBinNum[j]

    # 计算连续性
    # 构建城市的地理坐标列表
    global cur_city_list
    app.config["MONGO_URI"] = "mongodb://localhost:27017/city_locs"
    mongo = PyMongo(app)
    collection = mongo.db.city_locs
    data = collection.find({'name': {'$in': cur_city_list}})
    cur_city_locs = list(data[:])

    points = np.zeros([cur_city_num, 2])
    for i in range(len(cur_city_locs)):
        points[i][0] = cur_city_locs[i]['lng']
        points[i][1] = cur_city_locs[i]['lat']

    mds = MyMDS(1)
    tsne = TSNE(n_components=1, learning_rate=100)
    tsne_city_locs = tsne.fit_transform(points)
    mds_city_locs = mds.fit(points)
    # print(mds_city_locs)
    sel_city_locs = []
    for i in range(len(cur_city_locs)):
        sel_city_locs.append({'city': cur_city_locs[i]['name'], 'real_loc': points[i].tolist(), 'tsne_loc': str(tsne_city_locs[i][0]), 'mds_loc': str(mds_city_locs[i][0]) })
    # print(tsne_city_locs)
    # print(sel_city_locs)

    vor = Voronoi(points)
    vor_ridge_points = vor.ridge_points
    # print(vor_ridge_points)

    # 构建城市间相邻关系的邻接矩阵
    space_conn_mtx = np.zeros([cur_city_num, cur_city_num])
    for i in range(cur_city_num):
        space_conn_mtx[i][i] = 1
    for item in vor_ridge_points:
        space_conn_mtx[item[0]][item[1]] = 1
        space_conn_mtx[item[1]][item[0]] = 1
    # print(space_conn_mtx)

    for cNum in range(cluster_num_from, cluster_num_to+1):
        # 计算连续性
        for i in range(cNum):
            tmp_dis = sig_st_distribution[i].copy()
            conn_region_num = 0
            for j in range(tmp_dis.shape[0]):
                for k in range(tmp_dis.shape[1]):
                    if tmp_dis[j][k] == 0:
                        continue
                    tmp_conn_mtx = space_conn_mtx.copy()
                    st_conn_judge(tmp_dis, j, k, tmp_conn_mtx)
                    conn_region_num += 1
            # print(conn_region_num, sBinNum[i])
            continuity[cNum-cluster_num_from][i] = 1.0 - 1.0 * conn_region_num / sBinNum[i]

        # 构建iaqi分布
        iaqi_max = np.zeros([cNum, var_num])
        bin_range = 1.0 * 500 / iaqi_bin_num
        for item in air_quality_data:
            tmp_clsuter_id = item['label']
            for i in range(len(item['value'])):
                if item['value'][i] > iaqi_max[tmp_clsuter_id][i]:
                    iaqi_max[tmp_clsuter_id][i] = item['value'][i]
                bin_id = math.floor(item['value'][i] / bin_range)
                if bin_id == iaqi_bin_num:
                    bin_id = iaqi_bin_num - 1
                iaqi_distribution[cNum-cluster_num_from][tmp_clsuter_id][i][bin_id] += 1
        label_num = st_distribution.sum(axis=(1,2)).tolist()
        for i in range(cNum):
            iaqi_distribution[cNum-cluster_num_from][i] = iaqi_distribution[cNum-cluster_num_from][i] / label_num[i]

    # print(iaqi_distribution)
    # print(iaqi_distribution.sum(axis=2))

    # clustering_result = {'cluster_info': {'centers':centers.tolist(), 'label_num': label_num, 'iaqi_distribution': iaqi_distribution.tolist(), 'iaqi_max': iaqi_max.tolist(), 'concentration': concentration.tolist(), 'continuity': continuity.tolist()}, 'data': air_quality_data, 'st_distribution': st_distribution.tolist(), 'tsne_city_locs': sel_city_locs, 'tsne_point_locs': all_points_locs_tsne.tolist(), 'mds_point_locs': all_points_locs_mds.tolist(), 'umap_point_locs': all_points_locs_umap.tolist()}
    clustering_result = {'cluster_info': {'centers': centers.tolist(), 'label_num': label_num,
                                          'iaqi_distribution': iaqi_distribution.tolist(),
                                          'iaqi_max': iaqi_max.tolist(), 'concentration': concentration.tolist(),
                                          'continuity': continuity.tolist()}, 'data': air_quality_data,
                         'st_distribution': st_distribution.tolist(), 'tsne_city_locs': sel_city_locs,
                         'umap_point_locs': all_points_locs_umap.tolist()}
    return clustering_result

def findSubSeq (Seq, subSeq):
    last_cur = 0
    count = 0
    posList = []
    while (1):
        where = Seq.find(subSeq)
        if (not where == -1):
            # print(last_cur + where)
            posList.append(last_cur + where)
            Seq = Seq[where + len(subSeq):]
            last_cur = last_cur + where + len(subSeq)
            count += 1
        else:
            break
    return count, posList

def sortByPatternLen(item):
    return len(item['pattern'])

def Node(label, contain_pattern_list):
    return {'label': label, 'pattern_num': len(contain_pattern_list), 'pattern_list': contain_pattern_list, 'children': []}

def rank_top_events (seq_collection):
    # print(seq_collection)
    rank_list = []
    for i in range(cur_cluster_num):
        rank_list.append({'seq_num': 0, 'fisrt_pos': []})
        for seq in seq_collection:
            str_index = seq['pattern'].find(str(i))
            if str_index != -1:
                rank_list[i]['seq_num'] += 1
                rank_list[i]['fisrt_pos'].append(str_index)
    for i in range(cur_cluster_num):
        rank_list[i]['avg_first_pos'] = np.mean(rank_list[i]['fisrt_pos'])

    max_index = -1
    max_seq_num = -1
    max_first_pos = -1
    for i in range(cur_cluster_num):
        if rank_list[i]['seq_num'] > max_seq_num:
            max_index = i
            max_seq_num = rank_list[i]['seq_num']
            max_first_pos = rank_list[i]['avg_first_pos']
        elif rank_list[i]['seq_num'] == max_seq_num:
            if rank_list[i]['avg_first_pos'] < max_first_pos:
                max_index = i
                max_seq_num = rank_list[i]['seq_num']
                max_first_pos = rank_list[i]['avg_first_pos']

    return {'max_index': max_index, 'max_seq_num': max_seq_num, 'max_avg_pos': max_first_pos}

def pattern_is_valid (seq):
    if seq['pattern'] == '':
        return False
    else:
        return True

def core_flow(seq_collection, seq_index_list, tree_node, min_support, define_list):
    if len(define_list) != 0:
        print('have define list...')
    if len(seq_index_list) < min_support:
        # add exit as a child of tree_node
        if (len(seq_index_list) != 0):
            exit_node = Node(-1, seq_index_list)
            tree_node['children'].append(exit_node)
        return
    else:
        top_event = rank_top_events(seq_collection)
        seq_not_contain = []
        index_not_contain = []
        seq_contain = []
        index_contain = []
        for i in range(len(seq_index_list)):
            if seq_collection[i]['pattern'].find(str(top_event['max_index'])) != -1:
                seq_contain.append(seq_collection[i])
                index_contain.append(seq_index_list[i])
            else:
                seq_not_contain.append(seq_collection[i])
                index_not_contain.append(seq_index_list[i])
        # contain_patterns = []
        # for index in index_contain:
        #     contain_patterns.append(cur_global_label_patterns[index])
        if len(seq_contain) >= min_support:
            cur_node = Node(top_event['max_index'], index_contain)
            tree_node['children'].append(cur_node)
            pattern_null_list = []
            # for seq in seq_contain:
            #     first_pos = seq['pattern'].find(str(top_event['max_index'])
            #     seq['pattern'] = seq['pattern'][(first_pos+1):]
            #     if seq['pattern'] == '':
            #         seq_contain.remove(seq)
            seq_contain_copy = copy.deepcopy(seq_contain)
            for i in range(len(seq_contain_copy)):
                first_pos = seq_contain[i]['pattern'].find(str(top_event['max_index']))
                seq_contain[i]['pattern'] = seq_contain[i]['pattern'][(first_pos+1):]
            # seq_contain = list(filter(pattern_is_valid, seq_contain))
            # index_contain = []
            # for seq in seq_contain:
            #     index_contain.append
            # print(len(seq_index_list), len(index_not_contain), len(index_contain))
            core_flow(seq_not_contain, index_not_contain, tree_node, min_support, define_list)
            core_flow(seq_contain, index_contain, cur_node, min_support, define_list)
        else:
            if len(seq_index_list) != 0:
                exit_node = Node(-1, seq_index_list)
                tree_node['children'].append(exit_node)
            # if (len(index_contain) != 0):
            #     exit_node = Node(-1, index_contain)
            #     tree_node['children'].append(exit_node)
            # if (len(index_not_contain) != 0):
            #     exit_node = Node(-1, index_not_contain)
            #     tree_node['children'].append(exit_node)
            return


@app.route("/vmsp_pattern/<sequential_list>/<cluster_num>")
@cross_origin()
def get_vmsp_pattern (sequential_list, cluster_num):
    temporal_list = json.loads(sequential_list)
    seqStrList = []    #字符串形式的序列
    tmp_list_index = 0
    inputFile = open('./input.txt', mode='w')
    for key in temporal_list:
        tList = temporal_list[key]
        seqStrList.append("")
        for item in tList:
            seqStrList[tmp_list_index] += str(item)
            inputFile.write(str(item)+' ')
            inputFile.write('-1 ')
        tmp_list_index += 1
        inputFile.write('-2'+'\n')
    inputFile.close()
    # print(seqStrList)
    # vmsp('./input.txt', './output.txt', 0.5)
    # input_list = []
    # for key in temporal_list:
    #     tList = temporal_list[key]
    #     tList_input = []
    #     for item in tList:
    #         tList_input.append([item])
    #     input_list.append(tList_input)
    # print(input_list)
    minSupThreshold = 2
    maxSupThreshold = 10
    supStep = 1
    raw_patterns = []
    for sup in range(minSupThreshold, maxSupThreshold, supStep):
        sup_float = sup / 10
        spmf = Spmf("VMSP", input_filename='./input.txt', output_filename="./output.txt", arguments=[sup_float, 300, 1, True])
        spmf.run()
        if not spmf.patterns_:
            spmf.parse_output()
        # print(spmf.patterns_)
        patterns_dict_list = []
        for pattern_sup in spmf.patterns_:
            pattern = pattern_sup[:-1]
            pattern_str = ""
            for s in pattern:
                pattern_str += s
            sup = pattern_sup[-1:][0]
            sup = sup.strip()
            if not sup.startswith("#SUP"):
                print("support extraction failed")
            sid_index = sup.find('#SID')
            sup_val = sup[:sid_index].split()[1]
            sid = sup[sid_index+5:]
            patterns_dict_list.append({'pattern': pattern_str, 'sup': int(sup_val), 'sid': sid})
        # result = spmf.to_pandas_dataframe(pickle=True).to_dict(orient='records')
        raw_patterns.extend(patterns_dict_list)
    #
    # for pattern in raw_patterns:
    #     tmpStr = ""
    #     for item in pattern['pattern']:
    #         tmpStr += item
    #     pattern['pattern'] = tmpStr
    # print(len(raw_patterns), raw_patterns)
    # 模式的合并、约简（删除重复的和长度为1的模式）
    non_redundant_patterns = [dict(t) for t in set([tuple(d.items()) for d in raw_patterns])]
    non_redundant_patterns.sort(key=sortByPatternLen)
    for item in non_redundant_patterns:
        if len(item['pattern']) == 1:
            non_redundant_patterns.remove(item)
        else:
            break
    for i in range(len(non_redundant_patterns)):
        non_redundant_patterns[i]['index'] = i
    # print(non_redundant_patterns)

    # 为每个不重复的模式计算出现频次
    # findSubSeq
    for item in non_redundant_patterns:
        subseq = item['pattern']
        occurFre = []
        for seq in seqStrList:
            curCnt, curPosList = findSubSeq(seq, subseq)
            occurFre.append({'count': curCnt, 'pos_list': curPosList})
        item['occur'] = occurFre

    # 将模式按照其实类别分成多组
    start_label_patters = []
    for i in range(int(cluster_num)):
        start_label_patters.append([])
    for item in non_redundant_patterns:
        cur_pattern = item['pattern']
        for i in range(int(cluster_num)):
            if cur_pattern[0] == str(i) and len(cur_pattern) > 1:
                start_label_patters[i].append(item)
                break
    # print(start_label_patters)
    all_trees = []
    # global cur_global_label_patterns
    for i in range(int(cluster_num)):
        cur_label_pattern = start_label_patters[i]
        # cur_global_label_patterns = cur_label_pattern
        # print(cur_label_pattern)
        # 模式树构建
        cur_label_pattern_copy = copy.deepcopy(cur_label_pattern)
        cur_label_index_list = []
        for j in range(len(cur_label_pattern)):
            cur_label_index_list.append(j)
            cur_label_pattern_copy[j]['pattern'] = cur_label_pattern_copy[j]['pattern'][1:]
        # print(cur_label_pattern)
        rootNode = Node(i, cur_label_index_list)
        core_flow(cur_label_pattern_copy, cur_label_index_list, rootNode, 2, [])
        # print(rootNode)
        all_trees.append(rootNode)
    # print(all_trees)
    pattern_obj = {'label_patterns': start_label_patters, 'label_trees': all_trees}

    return pattern_obj

@app.route("/flow_importance/<flow_list>/<stack_list>/<city_num>/<cluster_num>")
@cross_origin()
def get_flow_importance (flow_list, stack_list, city_num, cluster_num):
    flow_list_json = json.loads(flow_list)
    stack_list_json = json.loads(stack_list)
    city_num = int(city_num)
    cluster_num = int(cluster_num)
    # print(flow_list_json)
    # print(stack_list_json)
    for item in flow_list_json:
        change_score = (math.fabs(item['source'] - item['target'])) / (cluster_num - 1)
        level_score = max(item['source'], item['target']) / cluster_num
        count_score = item['count'] / city_num
        importance = change_score + level_score + count_score

        cur_dis = stack_list_json[item['dateIndex']]
        cur_dis_next = stack_list_json[item['dateIndex']+1]
        source_rate = cur_dis[item['source']]['cnt'] / city_num
        target_rate = cur_dis_next[item['target']]['cnt'] / city_num
        count_rate = item['count'] / city_num
        source_rate_score = 0
        target_rate_score = 0
        if source_rate != 0:
            source_rate_score = 1.0 / source_rate
        if target_rate != 0:
            target_rate_score = 1.0 / target_rate
        surprise = (source_rate_score + target_rate_score) * count_rate
        item['importance'] = importance
        item['surprise'] = surprise
    # print(flow_list_json)
    return jsonify(flow_list_json)

if __name__ == '__main__':
 #    points = [[108.989923,34.304985],
 # [111.513333,36.076483],
 # [112.421429,34.659   ],

 # [111.1715,34.79065 ],
 # [108.714,34.34795 ],
 # [109.00725,34.987   ],
 # [107.189875,34.354125],
 # [109.48125,34.5014  ],
 # [112.729,37.6992  ],
 # [110.9934,35.07002 ],
 # [111.359,37.380467]]
 #    cur_vor = Voronoi(points)
 #    # print(cur_vor.vertices)
 #    # print(cur_vor.regions)
 #    print(cur_vor.points)
 #    print(cur_vor.ridge_points)
 #    fig = voronoi_plot_2d(cur_vor)
 #    plt.show()
    app.run(host='0.0.0.0', port=1002)
