'''
@Descripttion: 
@Author: Ian
@LastEditors: Ian
@Contact: yier_demon@163.com
@LastEditTime: 2020-05-04 09:43:13
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import json

# 针对标称属性，统计频数
def Nominal(df1, filename):
    resdic = {}
    col1 = df1.country.unique()     #country
    col6 = df1.province.unique()    #province
    col7 = df1.region_1.unique()    #region_1
    col8 = df1.region_2.unique()    #region_2
    col9 = df1.variety.unique()      #variety
    col10 = df1.winery.unique()      #winery
    resdic["country"] = {}
    resdic["province"] = {}
    resdic["region_1"] = {}
    resdic["region_2"] = {}
    resdic["variety"] = {}
    resdic["winery"] = {}

    for each in col1:
        resdic["country"][each] = list(df1.country).count(each)
    for each in col6:
        resdic["province"][each] = list(df1.province).count(each)
    for each in col7:
        resdic["region_1"][each] = list(df1.region_1).count(each)
    for each in col8:
        resdic["region_2"][each] = list(df1.region_2).count(each)
    for each in col9:
        resdic["variety"][each] = list(df1.variety).count(each)
    for each in col10:
        resdic["winery"][each] = list(df1.winery).count(each)
    # print(resdic)

    # with open(filename, 'w', encoding="utf-8") as wf:
    #     wf.writelines(resdic)
    jsObj = json.dumps(resdic)
    fileObject = open(filename, 'w', encoding="utf-8")
    fileObject.write(jsObj)
    fileObject.close()


# 可视化
def Numeric(Ndf):
    resdic = {}
    describe = Ndf.describe()
    resdic["col4"] = {}
    resdic["col5"] = {}
    print(describe)
    resdic["col4"]["Max"] = describe.points[7]
    resdic["col4"]["Min"] = describe.points[3]
    resdic["col4"]["Mean"] = describe.points[1]
    resdic["col4"]["Mid"] = describe.points[5]
    resdic["col4"]["25"] = describe.points[4]
    resdic["col4"]["75"] = describe.points[6]
    resdic["col4"]["Nan"] = Ndf.points.isna().sum()
    print("col4_nan:{}".format(resdic["col4"]["Nan"])) # 缺失值数量

    resdic["col5"]["Max"] = describe.price[7]
    resdic["col5"]["Min"] = describe.price[3]
    resdic["col5"]["Mean"] = describe.price[1]
    resdic["col5"]["Mid"] = describe.price[5]
    resdic["col5"]["25"] = describe.price[4]
    resdic["col5"]["75"] = describe.price[6]
    resdic["col5"]["Nan"] = Ndf.price.isna().sum()
    print("col5_nan:{}".format(resdic["col5"]["Nan"])) # 缺失值数量

    # 直方图
    Ndf = Ndf.dropna(subset=["points"]) # 去空值
    plt.hist(Ndf["points"], bins=100)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("points--Frequency distribution histogram")
    plt.show()

    Ndf = Ndf.dropna(subset=["price"]) # 去空值
    plt.hist(Ndf["price"], bins=100)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("price--Frequency distribution histogram")
    plt.show()


    # qq图
    points = Ndf["points"]
    price = Ndf["price"]
    sort_points = np.sort(points)
    sort_price = np.sort(price)
    y_points = np.arange(len(sort_points))/float(len(sort_points))
    y_price = np.arange(len(sort_price))/float(len(sort_price))
    trans_y_points = stats.norm.ppf(y_points)
    trans_y_price = stats.norm.ppf(y_price)
    plt.scatter(sort_points, trans_y_points)
    plt.xlabel("Ordered values")
    plt.ylabel("Theoretical quantile")
    plt.title("points--quantile-quantile plot")
    plt.show()
    plt.scatter(sort_price, trans_y_price)
    plt.xlabel("Ordered values")
    plt.ylabel("Theoretical quantile")
    plt.title("price--quantile-quantile plot")
    plt.show()

    # 盒图
    plt.boxplot(Ndf["points"])
    plt.ylabel("points")
    plt.show()
    plt.boxplot(Ndf["price"])
    plt.ylabel("price")
    plt.show()


# 用最高频率值来填补缺失值
def high_feq_process(df1):
    Ndf = pd.DataFrame(df1, columns=["points", "price"])

    # 用value_counts()方法计算指定数值属性不同取值的频率
    feq_points = Ndf["points"].value_counts()
    feq_price = Ndf["price"].value_counts()
    
    # 通过下标得到频率最高的取值
    fill_value = {
        "points": list(dict(feq_points))[0],
        "price": list(dict(feq_price))[0]
    }
    print(fill_value)
    # 用fillna()方法填补对应列的缺失值，并调用之前的可视化函数
    Ndf = Ndf.fillna(value=fill_value) 
    print(Ndf)
    Numeric(Ndf)


# 通过属性的相关关系来填补缺失值
def relation_process(df1):
    Ndf = pd.DataFrame(df1, columns=["points", "price"])
    Ndf.interpolate(method="values")
    Numeric(Ndf)


# 通过数据对象之间的相似性来填补缺失值
def similarity_process(df1, k_num):
    OriNdf = pd.DataFrame(df1, columns=["points", "price"])
    Ndf = pd.DataFrame(df1, columns=["points", "price"])
    Ndf = Ndf.dropna(axis=0, how="any")
    clf = KNeighborsRegressor(n_neighbors=k_num, weights="distance")
    clf.fit(np.array(list(Ndf["points"])).reshape(-1, 1), np.array(list(Ndf["price"])).reshape(-1, 1)) 

    for i in range(0, len(OriNdf)):
        if pd.isna(OriNdf.iloc[i]["price"]):
            new_value = clf.predict(np.array([OriNdf.iloc[i]["points"]]).reshape(-1, 1))
            OriNdf.set_value(i, "price", new_value)
    Numeric(OriNdf)


if __name__ == "__main__":
    filelist = ["winemag-data_first150k.csv", "winemag-data-130k-v2.csv"]
    for id, input_file in enumerate(filelist):
        dataframe = pd.read_csv('data\\wine-reviews\\'+input_file)
        Nominal(dataframe, "data\\wine-reviews\\result-Nominal-{}.json".format(id))
        # print("原始数据")
        # Numeric(pd.DataFrame(dataframe, columns=["points", "price"]))
        # print("用最高频率值来填补缺失值")
        # high_feq_process(dataframe)
        # print("通过属性的相关关系来填补缺失值")
        # relation_process(dataframe)
        # print("通过数据对象之间的相似性来填补缺失值")
        # similarity_process(dataframe, k_num=3)
