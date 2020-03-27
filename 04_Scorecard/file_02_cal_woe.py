#---------------------------------------------------------------------------------
#   计算各箱的WOE并映射到数据中
#  我们现在已经有了我们的箱子，接下来我们要做的是计算各箱的WOE，并且把WOE替换到我们的原始数据model_data中，因为我们将使用WOE覆盖后的数据来建模，
#  我们希望获取的是”各个箱”的分类结果，即评分卡上各个评分项目的分类结果。
#--------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
pd.set_option('display.float_format',lambda x: '%.4f'%x)
import numpy  as np
from numpy import inf
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
import seaborn as sns

model_data = pd.read_csv(r'D:\CreditScorecard\Data\model_data.csv',encoding='gbk')
vali_data = pd.read_csv(r'D:\CreditScorecard\Data\vali_data.csv',encoding='gbk')
#读取字典文件
f = open(r'D:\CreditScorecard\Data\bins_of_final.txt','r')
bins_of_final = eval(f.read())
f.close()

def get_woe(df, col, y, bins):
    df = df[[col, y]].copy()
    df["cut"] = pd.cut(df[col], bins)
    bins_df = df.groupby("cut")[y].value_counts().unstack()
    woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
    return woe


# 将所有特征的WOE存储到字典当中
woeall = {}
for col in bins_of_final:
    woeall[col] = get_woe(model_data, col, "target", bins_of_final[col])

#保存woeall字典文件到本地
f = open(r'D:\CreditScorecard\Data\woeall.txt','w')
f.write(str(woeall))
f.close()

#把所有WOE映射到原始数据中
model_woe = pd.DataFrame(index=model_data.index)
for col in bins_of_final:
    model_woe[col] = pd.cut(model_data[col],bins_of_final[col]).map(woeall[col])

model_woe['target'] = model_data['target']
model_woe.to_csv(r'D:\CreditScorecard\Data\model_woe.csv',encoding='gbk',index=False)

#处理测试集，把计算好的WOE映射到测试集中
vali_woe = pd.DataFrame(index=vali_data.index)
for col in bins_of_final:
    vali_woe[col] = pd.cut(vali_data[col],bins_of_final[col]).map(woeall[col])

vali_woe['target'] = vali_data['target']
vali_woe.to_csv(r'D:\CreditScorecard\Data\vali_woe.csv',encoding='gbk',index=False)

