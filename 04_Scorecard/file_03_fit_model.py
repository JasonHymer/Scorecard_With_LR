#-----------------------------------------------------------------------
#   训练模型，并调整参数
#----------------------------------------------------------------------
import pandas  as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import scikitplot as skplt
import matplotlib.pyplot as plt
from numpy import inf
from sklearn import metrics
import seaborn as sns

#读取训练集(model_woe)和测试集(vali_woe)
model_woe = pd.read_csv(r'D:\CreditScorecard\Data\model_woe.csv',encoding='gbk')
X = model_woe.iloc[:,:-1]
y = model_woe.iloc[:,-1]

vali_woe = pd.read_csv(r'D:\CreditScorecard\Data\vali_woe.csv',encoding='gbk')
vali_X = vali_woe.iloc[:,:-1]
vali_y = vali_woe.iloc[:,-1]

'''
#读取字典文件
f = open(r'D:\Scorecard\woeall.txt','r')
woeall = eval(f.read())
f.close()
'''


#建模，逻辑回归模型
#实例化
lr = LR()
#用训练数据拟合模型
lr = lr.fit(X,y)
lr.score(vali_X,vali_y)

'''
lr = LR(penalty='l1',solver='liblinear',C=0.5,max_iter=100)
#用训练数据拟合模型
lr = lr.fit(X,y)
lr.score(vali_X,vali_y)
'''

'''
#尝试使用C和max_iter的学习曲线把逻辑回归的效果调上去
c_1 = np.linspace(0.01,10,20)
score = []
for i in c_1:
    lr = LR(solver='liblinear',C=i).fit(X,y)
    score.append(lr.score(vali_X,vali_y))
plt.figure()
plt.plot(c_1,score)
plt.show()
'''

#看模型在ROC曲线上的效果
vali_proba_df = pd.DataFrame(lr.predict_proba(vali_X))
skplt.metrics.plot_roc(vali_y, vali_proba_df,
                        plot_micro=False,figsize=(6,6),
                        plot_macro=False)


def plot_roc(y_label, y_pred):
    """
    绘制roc曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        roc曲线
    """
    tpr, fpr, threshold = metrics.roc_curve(y_label, y_pred)
    AUC = metrics.roc_auc_score(y_label, y_pred)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tpr, fpr, color='blue', label='AUC=%.3f' % AUC)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show()


def plot_model_ks(y_label, y_pred):
    """
    绘制ks曲线
    param:
        y_label -- 真实的y值 list/array
        y_pred -- 预测的y值 list/array
    return:
        ks曲线
    """
    pred_list = list(y_pred)
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list) - total_bad
    items = sorted(zip(pred_list, label_list), key=lambda x: x[0])
    step = (max(pred_list) - min(pred_list)) / 200

    pred_bin = []
    good_rate = []
    bad_rate = []
    ks_list = []
    for i in range(1, 201):
        idx = min(pred_list) + i * step
        pred_bin.append(idx)
        label_bin = [x[1] for x in items if x[0] < idx]
        bad_num = sum(label_bin)
        good_num = len(label_bin) - bad_num
        goodrate = good_num / total_good
        badrate = bad_num / total_bad
        ks = abs(goodrate - badrate)
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pred_bin, good_rate, color='green', label='good_rate')
    ax.plot(pred_bin, bad_rate, color='red', label='bad_rate')
    ax.plot(pred_bin, ks_list, color='blue', label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show()

y_pred = lr.predict_proba(vali_X)[:,1]
plot_roc(vali_y, y_pred=y_pred)
plot_model_ks(vali_y, y_pred=y_pred)



#---------------------------------------------------------------------------------------------------------------------
#  制作评分卡
#---------------------------------------------------------------------------------------------------------------------
B = 20 / np.log(2)
A = 600 + B * np.log(1/60)
base_score = A - B*lr.intercept_#lr.intercept_：截距

score_df = pd.DataFrame()
score_df1 = pd.DataFrame()
col = model_woe.columns[:-1]
coe_dict = dict(zip(list(col),list(lr.coef_[0,:])))
for cc in model_woe.columns[:-1]:
    score_df[cc] = model_woe[cc].apply(lambda x : x * (-B * coe_dict[cc]))
    score_df1[cc] = vali_woe[cc].apply(lambda x: x * (-B * coe_dict[cc]))

score_df['base_score'] = base_score[0,]
score_df['Score'] = score_df.apply(lambda x : x.sum(), axis=1)
score_df['Score'] = score_df['Score'].apply(lambda x : round(x,0))
score_df['target'] = model_woe['target']
score_df1['base_score'] = base_score[0,]
score_df1['Score'] = score_df1.apply(lambda x : x.sum(), axis=1)
score_df1['Score'] = score_df1['Score'].apply(lambda x : round(x,0))
score_df1['target']= vali_woe['target']

#绘制好坏用户得分分布图
def plot_score_hist(df,target,score_col,title):
    """
    绘制好坏用户得分分布图
    param:
        df -- 数据集 Dataframe
        target -- 标签字段名 string
        score_col -- 模型分的字段名 string
        plt_size -- 绘图尺寸 tuple
        title -- 图表标题 string
    return:
        好坏用户得分分布图
    """
    plt.figure(figsize=(6,6))
    plt.title(title)
    x1 = df[df[target]==1][score_col]
    x2 = df[df[target]==0][score_col]
    sns.kdeplot(x1,shade=True,label='bad',color='hotpink')
    sns.kdeplot(x2,shade=True,label='good',color ='seagreen')
    plt.legend()
    return plt.show()

plot_score_hist(score_df,'target','Score','train_score')
plot_score_hist(score_df1,'target','Score','vali_score')

'''
file = r"D:\Scorecard\ScoreData.csv"
with open(file,"w") as fdata:
    fdata.write("base_score,{}\n".format(base_score))
for i,col in enumerate(X.columns):#[*enumerate(X.columns)]
    score = woeall[col] * (-B*lr.coef_[0][i])
    score.name = "Score"
    score.index.name = col
    score.to_csv(file,header=True,mode="a")

'''



