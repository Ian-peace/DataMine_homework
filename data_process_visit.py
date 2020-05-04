'''
@Descripttion: 
@Author: Ian
@LastEditors: Ian
@Contact: yier_demon@163.com
@LastEditTime: 2020-05-04 10:05:51
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import json
# 4 5 8

def Nominal(df1, filename):
    resdic = {}
    col1 = df1.census_block_group.unique()
    col2 = df1.date_range_start.unique()
    col3 = df1.date_range_end.unique()
    col6 = df1.visitor_home_cbgs.unique()
    col7 = df1.visitor_work_cbgs.unique()
    col9 = df1.related_same_day_brand.unique()
    col10 = df1.related_same_month_brand.unique()
    col11 = df1.top_brands.unique()
    col12 = df1.popularity_by_hour.unique()
    col13 = df1.popularity_by_day.unique()

    resdic["census_block_group"] = {}
    resdic["date_range_start"] = {}
    resdic["date_range_end"] = {}
    resdic["visitor_home_cbgs"] = {}
    resdic["visitor_work_cbgs"] = {}
    resdic["related_same_day_brand"] = {}
    resdic["related_same_month_brand"] = {}
    resdic["top_brands"] = {}
    resdic["popularity_by_hour"] = {}
    resdic["popularity_by_day"] = {}

    for each in col1:
        resdic["census_block_group"][each] = list(df1.census_block_group).count(each)
    for each in col2:
        resdic["date_range_start"][each] = list(df1.date_range_start).count(each)
    for each in col3:
        resdic["date_range_end"][each] = list(df1.date_range_end).count(each)
    for each in col6:
        resdic["visitor_home_cbgs"][each] = list(df1.visitor_home_cbgs).count(each)
    for each in col7:
        resdic["visitor_work_cbgs"][each] = list(df1.visitor_work_cbgs).count(each)
    for each in col9:
        resdic["related_same_day_brand"][each] = list(df1.related_same_day_brand).count(each)
    for each in col10:
        resdic["related_same_month_brand"][each] = list(df1.related_same_month_brand).count(each)
    for each in col11:
        resdic["top_brands"][each] = list(df1.top_brands).count(each)
    for each in col12:
        resdic["popularity_by_hour"][each] = list(df1.popularity_by_hour).count(each)
    for each in col13:
        resdic["popularity_by_day"][each] = list(df1.popularity_by_day).count(each)
    
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
    resdic["col8"] = {}
    print(describe)
    resdic["col4"]["Max"] = describe.raw_visit_count[7]
    resdic["col4"]["Min"] = describe.raw_visit_count[3]
    resdic["col4"]["Mean"] = describe.raw_visit_count[1]
    resdic["col4"]["Mid"] = describe.raw_visit_count[5]
    resdic["col4"]["25"] = describe.raw_visit_count[4]
    resdic["col4"]["75"] = describe.raw_visit_count[6]
    resdic["col4"]["Nan"] = Ndf.raw_visit_count.isna().sum()
    print("col4_nan:{}".format(resdic["col4"]["Nan"])) # 缺失值数量

    resdic["col5"]["Max"] = describe.raw_visitor_count[7]
    resdic["col5"]["Min"] = describe.raw_visitor_count[3]
    resdic["col5"]["Mean"] = describe.raw_visitor_count[1]
    resdic["col5"]["Mid"] = describe.raw_visitor_count[5]
    resdic["col5"]["25"] = describe.raw_visitor_count[4]
    resdic["col5"]["75"] = describe.raw_visitor_count[6]
    resdic["col5"]["Nan"] = Ndf.raw_visitor_count.isna().sum()
    print("col5_nan:{}".format(resdic["col5"]["Nan"])) # 缺失值数量

    resdic["col8"]["Max"] = describe.distance_from_homepoints[7]
    resdic["col8"]["Min"] = describe.distance_from_homepoints[3]
    resdic["col8"]["Mean"] = describe.distance_from_homepoints[1]
    resdic["col8"]["Mid"] = describe.distance_from_homepoints[5]
    resdic["col8"]["25"] = describe.distance_from_homepoints[4]
    resdic["col8"]["75"] = describe.distance_from_homepoints[6]
    resdic["col8"]["Nan"] = Ndf.distance_from_homepoints.isna().sum()
    print("col8_nan:{}".format(resdic["col8"]["Nan"])) # 缺失值数量


    # 直方图
    Ndf = Ndf.dropna(subset=["raw_visit_count"]) # 去空值
    plt.hist(Ndf["raw_visit_count"], bins=100)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("raw_visit_count--Frequency distribution histogram")
    plt.show()

    Ndf = Ndf.dropna(subset=["raw_visitor_count"]) # 去空值
    plt.hist(Ndf["raw_visitor_count"], bins=100)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("raw_visitor_count--Frequency distribution histogram")
    plt.show()

    Ndf = Ndf.dropna(subset=["distance_from_homepoints"]) # 去空值
    plt.hist(Ndf["distance_from_homepoints"], bins=100)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("distance_from_homepoints--Frequency distribution histogram")
    plt.show()


    # qq图
    raw_visit_count = Ndf["raw_visit_count"]
    raw_visitor_count = Ndf["raw_visitor_count"]
    distance_from_homepoints = Ndf["distance_from_homepoints"]

    sort_raw_visit_count = np.sort(raw_visit_count)
    sort_raw_visitor_count = np.sort(raw_visitor_count)
    sort_distance_from_homepoints = np.sort(distance_from_homepoints)

    y_raw_visit_count = np.arange(len(sort_raw_visit_count))/float(len(sort_raw_visit_count))
    y_raw_visitor_count = np.arange(len(sort_raw_visitor_count))/float(len(sort_raw_visitor_count))
    y_distance_from_homepoints = np.arange(len(sort_distance_from_homepoints))/float(len(sort_distance_from_homepoints))

    trans_y_raw_visit_count = stats.norm.ppf(y_raw_visit_count)
    trans_y_raw_visitor_count = stats.norm.ppf(y_raw_visitor_count)
    trans_y_distance_from_homepoints = stats.norm.ppf(y_distance_from_homepoints)

    plt.scatter(sort_raw_visit_count, trans_y_raw_visit_count)
    plt.xlabel("Ordered values")
    plt.ylabel("Theoretical quantile")
    plt.title("raw_visit_count--quantile-quantile plot")
    plt.show()
    plt.scatter(sort_raw_visitor_count, trans_y_raw_visitor_count)
    plt.xlabel("Ordered values")
    plt.ylabel("Theoretical quantile")
    plt.title("raw_visitor_count--quantile-quantile plot")
    plt.show()
    plt.scatter(sort_distance_from_homepoints, trans_y_distance_from_homepoints)
    plt.xlabel("Ordered values")
    plt.ylabel("Theoretical quantile")
    plt.title("distance_from_homepoints--quantile-quantile plot")
    plt.show()

    # 盒图
    plt.boxplot(Ndf["raw_visit_count"])
    plt.ylabel("raw_visit_count")
    plt.show()
    plt.boxplot(Ndf["raw_visitor_count"])
    plt.ylabel("raw_visitor_count")
    plt.show()
    plt.boxplot(Ndf["distance_from_homepoints"])
    plt.ylabel("distance_from_homepoints")
    plt.show()


# 用最高频率值来填补缺失值
def high_feq_process(df1):
    Ndf = pd.DataFrame(df1, columns=["raw_visit_count", "raw_visitor_count", "distance_from_homepoints"])

    # 用value_counts()方法计算指定数值属性不同取值的频率
    feq_raw_visit_count = Ndf["raw_visit_count"].value_counts()
    feq_raw_visitor_count = Ndf["raw_visitor_count"].value_counts()
    feq_distance_from_homepoints = Ndf["distance_from_homepoints"].value_counts()
    
    # 通过下标得到频率最高的取值
    fill_value = {
        "raw_visit_count": list(dict(feq_raw_visit_count))[0],
        "raw_visitor_count": list(dict(feq_raw_visitor_count))[0],
        "distance_from_homepoints": list(dict(feq_distance_from_homepoints))[0]
    }
    print(fill_value)
    # 用fillna()方法填补对应列的缺失值，并调用之前的可视化函数
    Ndf = Ndf.fillna(value=fill_value) 
    print(Ndf)
    Numeric(Ndf)


# 通过属性的相关关系来填补缺失值
def relation_process(df1):
    Ndf = pd.DataFrame(df1, columns=["raw_visit_count", "raw_visitor_count", "distance_from_homepoints"])
    Ndf.interpolate(method="values")
    Numeric(Ndf)


# 通过数据对象之间的相似性来填补缺失值
def similarity_process(df1, k_num):
    OriNdf = pd.DataFrame(df1, columns=["raw_visit_count", "raw_visitor_count", "distance_from_homepoints"])
    Ndf = pd.DataFrame(df1, columns=["raw_visit_count", "raw_visitor_count", "distance_from_homepoints"])
    Ndf = Ndf.dropna(axis=0, how="any")
    clf = KNeighborsRegressor(n_neighbors=k_num, weights="distance")
    clf.fit(np.array(list(Ndf["raw_visit_count"])).reshape(-1, 1), 
            np.array(list(Ndf["raw_visitor_count"])).reshape(-1, 1),
            np.array(list(Ndf["distance_from_homepoints"])).reshape(-1, 1)) 

    for i in range(0, len(OriNdf)):
        if pd.isna(OriNdf.iloc[i]["raw_visitor_count"]):
            new_value = clf.predict(np.array([OriNdf.iloc[i]["raw_visit_count"]]).reshape(-1, 1))
            OriNdf.set_value(i, "raw_visitor_count", new_value)
        if pd.isna(OriNdf.iloc[i]["distance_from_homepoints"]):
            new_value = clf.predict(np.array([OriNdf.iloc[i]["raw_visit_count"]]).reshape(-1, 1))
            OriNdf.set_value(i, "distance_from_homepoints", new_value)
    Numeric(OriNdf)


if __name__ == "main":
    print("test")
    dataframe = pd.read_csv('data\\visit-patterns-by-census-block-group\\cbg_patterns.csv')
    
    # 针对标称属性
    print("对标称属性统计频数")
    Nominal(dataframe, "data\\visit-patterns-by-census-block-group\\result-Nominal-{}.json".format(id))
    
    # 针对数值属性
    print("原始数据(将缺失部分剔除)")
    Numeric(pd.DataFrame(dataframe, columns=["raw_visit_count", "raw_visitor_count", "distance_from_homepoints"]))
    print("用最高频率值来填补缺失值")
    high_feq_process(dataframe)
    print("通过属性的相关关系来填补缺失值")
    relation_process(dataframe)
    print("通过数据对象之间的相似性来填补缺失值")
    similarity_process(dataframe, k_num=3)