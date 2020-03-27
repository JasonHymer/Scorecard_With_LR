"""
分箱:
    在评分卡制作过程中，一个重要的步骤就是分箱。可以说，分箱是评分卡最难，也是最核心的思路，分箱的本质，其实就是离散化连续变量，
好让拥有不同属性的人被分成不同的类别（打上不同的分数），其实本质比较类似于聚类。
    我们希望不同属性的人有不同的分数，因此我们希望在同一个箱子内的人的属性是尽量相似的，而不同箱子的人的属性是尽量不同的，
即业界常说的 组间差异大，组内差异小。对于评分卡来说，就是说我们希望一个箱子内的人违约概率是类似的，而不同箱子的人的违约概率差距很大，
即WOE差距要大，并且每个箱子中坏客户所占的比重（）也要不同。那我们，可以使用卡方检验来对比两个箱子之间的相似性，如果两个箱子之间卡方检验的P值很
大，则说明他们非常相似，那我们就可以将这两个箱子合并为一个箱子。基于这样的思想，我们总结出我们对一个特征进行分箱的步骤：
1）我们首先把连续型变量分成一组数量较多的分类型变量，比如，将几万个样本分成100组，或50组
2）确保每一组中都要包含两种类别的样本，否则IV值会无法计算
3）我们对相邻的组进行卡方检验，卡方检验的P值很大的组进行合并，直到数据中的组数小于设定的N箱为止
4）我们让一个特征分别分成[2,3,4.....20]箱，观察每个分箱个数下的IV值如何变化，找出最适合的分箱个数
5）分箱完毕后，我们计算每个箱的WOE值， ，观察分箱效果
这些步骤都完成后，我们可以对各个特征都进行分箱，然后观察每个特征的IV值，以此来挑选特征。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

train_data = pd.read_csv(r'D:\CreditScorecard\Data\train_data.csv')
test_data = pd.read_csv(r'D:\CreditScorecard\Data\test_data.csv')

#----------------------------------------------
#   根据变量唯一数据的个数划分类别变量和数值变量
#----------------------------------------------
cate_col_list = []
num_col_list= []

for col in train_data.columns[1:-1]:
    if train_data[col].unique().shape[0] <= 10: #变量唯一数据小于10的，划分为类别变量，根据类别分箱
        cate_col_list.append(col)
    else:
        num_col_list.append(col)
'''
#类别型变量的分箱
def binning_cate(df, col_list, target):
    """
    df:数据集
    col_list:变量list集合
    target:目标变量的字段名

    return:
    bin_df :list形式，里面存储每个变量的分箱结果
    iv_value:list形式，里面存储每个变量的IV值
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    all_odds = good * 1.0 / bad
    bin_df = []
    iv_value = []
    eps= 0.000001  # eps用于防止计算IV时出现inf
    for col in col_list:
        d1 = df.groupby([col], as_index=True)
        d2 = pd.DataFrame()
        d2['min_bin'] = d1[col].min()
        d2['max_bin'] = d1[col].max()
        d2['total'] = d1[target].count()
        d2['totalrate'] = d2['total'] / total
        d2['bad'] = d1[target].sum() + eps
        d2['badrate'] = d2['bad'] / d2['total']
        d2['good'] = d2['total'] - d2['bad']
        d2['goodrate'] = d2['good'] / d2['total']
        d2['badattr'] = d2['bad'] / bad
        d2['goodattr'] = (d2['total'] - d2['bad']) / good
        d2['odds'] = d2['good'] / d2['bad']
        d2['woe'] = np.log(d2['badattr'] / d2['goodattr'])
        d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['woe']
        d2['IV'] = d2['bin_iv'].sum()
        iv = d2['bin_iv'].sum().round(3)
        print('变量名:{}'.format(col))
        print('IV:{}'.format(iv))
        print('\t')
        bin_df.append(d2)
        iv_value.append(iv)
    return bin_df, iv_value

#类别型变量IV的明细表
def iv_cate(df,col_list,target):
    """
    :param df: 数据集
    :param col_list: 变量List集合
    :param target: 目标变量的字段名
    :return: 变量的IV明细表
    """
    bin_df,iv_value = binning_cate(df,col_list,target)
    iv_df = pd.DataFrame({'col':col_list,'iv':iv_value})
    iv_df = iv_df.sort_values('iv',ascending=False)
    return iv_df

iv_df = iv_cate(train_data,cate_col_list,'target')

bin_df,iv_value = binning_cate(train_data,cate_col_list,'target')
IV_all=pd.DataFrame()
for bin1 in bin_df:
    col=bin1.index.name
    bin1=bin1.reset_index(drop=True)
    bin1=pd.DataFrame(bin1)
    #bin1=bin1.rename(columns={bin1.columns[0]:'col'},inplace=True)
    bin1['col']=col
    IV_all=IV_all.append(bin1)

candidate_col = list(IV_all[IV_all.IV>0.01].col.unique())
'''

# 自定义分箱
def binning_self(df, col, target, cut=None, right_border=True):
    """
    df: 数据集
    col:分箱的单个变量名
    cut:划分区间的list
    right_border：设定左开右闭、左闭右开

    return:
    bin_df: df形式，单个变量的分箱结果
    iv_value: 单个变量的iv
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    all_odds = good / bad
    bucket = pd.cut(df[col], cut, right=right_border)
    d1 = df.groupby(bucket)
    d2 = pd.DataFrame()
    d2['min_bin'] = d1[col].min()
    d2['max_bin'] = d1[col].max()
    d2['total'] = d1[target].count()
    d2['totalrate'] = d2['total'] / total
    d2['bad'] = d1[target].sum()
    d2['badrate'] = d2['bad'] / d2['total']
    d2['good'] = d2['total'] - d2['bad']
    d2['goodrate'] = d2['good'] / d2['total']
    d2['badattr'] = d2['bad'] / bad
    d2['goodattr'] = (d2['total'] - d2['bad']) / good
    d2['odds'] = d2['good'] / d2['bad']
    GB_list = []
    for i in d2.odds:
        if i >= all_odds:
            GB_index = str(round((i / all_odds) * 100, 0)) + str('G')
        else:
            GB_index = str(round((all_odds / i) * 100, 0)) + str('B')
        GB_list.append(GB_index)
    d2['GB_index'] = GB_list
    d2['woe'] = np.log(d2['badattr'] / d2['goodattr'])
    d2['bin_iv'] = (d2['badattr'] - d2['goodattr']) * d2['woe']
    d2['IV'] = d2['bin_iv'].sum()
    iv_value = d2['bin_iv'].sum().round(3)
    print('变量名:{}'.format(col))
    print('IV:{}'.format(iv_value))
    bin_df = d2.copy()
    return bin_df, iv_value


#不能使用自动分箱的变量
hand_bins = {
 'FREQUENCY_DETAIL_SCORE':[-1,0,6,20],
 'GREY_LIST_SCORE':[-1,2,30],
 'FREQUENCY_DETAIL_SCORE.1':[0,10,1000],
 'ALD_INB_OTH_ORGNUM':[1,2,10],
 'ALD_CNB_OTH_ORGNUM':[1,2,10],
 'IR_ID_X_MAIL_CNT':[0,1,5],
 'IR_ID_X_NAME_CNT':[1,2,6],
 'IR_CELL_X_MAIL_CNT':[0,1,5],
 'IR_CELL_X_NAME_CNT':[1,2,6],
 'IR_M3_ID_X_CELL_CNT':[0,1,4],
 'IR_M3_ID_X_HADDR_CNT':[-9999,0,3],
 'IR_M3_CELL_X_HADDR_CNT':[-9999,0,3],
 'IR_M6_ID_X_CELL_CNT':[0,1,4],
 'IR_M6_CELL_X_HADDR_CNT':[-9999,0,5],
 'IR_M6_CELL_X_BIZ_ADDR_CNT':[-9999,0,6],
 'IR_M6_LMAN_CELL_X_ID_CNT':[-9999,0,3],
 'IR_M6_LMAN_CELL_X_CELL_CNT':[-9999,0,3],
 'IR_M12_ID_X_CELL_CNT':[1,5],
 'IR_M12_ID_X_NAME_CNT':[1,5],
 'IR_M12_CELL_X_NAME_CNT':[1,5],
 'IR_M12_ID_X_HADDR_CNT':[0,1,7],
 'IR_M12_ID_X_BIZ_ADDR_CNT':[0,1,10],
 'IR_M12_CELL_X_HADDR_CNT':[0,1,7],
 'IR_M12_CELL_X_BIZ_ADDR_CNT':[0,1,10],
 'IR_M12_LMAN_CELL_X_ID_CNT':[0,3],
 'IR_M12_LMAN_CELL_X_CELL_CNT':[0,3]
}

hand_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()}

def cal_iv(df,target,hand_bins):
    """
    :param df: 数据集 eg:train_data
    :param target: Y值  eg:'target'
    :param hand_bins: 需要手动分箱的变量列表字典  eg : hand_col_list
    :return:  返回所有变量的IV信息
    """
    bin_df = []
    iv_value = []
    for col in hand_bins:
        df, iv = binning_self(train_data, col, 'target', cut=hand_bins[col])
        bin_df.append(df)
        iv_value.append(iv_value)

    IV_all = pd.DataFrame()
    for bin1 in bin_df:
        col = bin1.index.name
        bin1 = bin1.reset_index(drop=True)
        bin1 = pd.DataFrame(bin1)
        # bin1=bin1.rename(columns={bin1.columns[0]:'col'},inplace=True)
        bin1['col'] = col
        IV_all = IV_all.append(bin1)
    return IV_all

'''
bin_df = []
iv_value = []
for col in hand_bins:
    df, iv = binning_self(train_data, col, 'target', cut=hand_bins[col])
    bin_df.append(df)
    iv_value.append(iv_value)

IV_all=pd.DataFrame()
for bin1 in bin_df:
    col=bin1.index.name
    bin1=bin1.reset_index(drop=True)
    bin1=pd.DataFrame(bin1)
    #bin1=bin1.rename(columns={bin1.columns[0]:'col'},inplace=True)
    bin1['col']=col
    IV_all=IV_all.append(bin1)
'''

#卡方分箱，对于数值变量用卡方分箱
def graphforbestbin(DF, X, Y, n=5, q=20, graph=False):
    '''
    自动最优分箱函数，基于卡方检验的分箱

    参数：
    DF: 需要输入的数据
    X: 需要分箱的列名
    Y: 分箱数据对应的标签 Y 列名
    n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像

    区间为前开后闭 (]

    '''

    DF = DF[[X, Y]].copy()
    bins_df = pd.DataFrame()
    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]

    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
            continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]
                break
        else:
            break

    def get_woe(num_bins):
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    def get_iv(df):
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    IV = []
    axisx = []
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3])]

        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))

    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.show()
    return bins_df


for i in num_col_list:
    print(i)
    graphforbestbin(train_data,i,"target",n=2,q=10)

#对于无法卡方分箱的变量，查看类别变量分箱情况
cate_col_list2 = ['ALD_ID_X_CELL_NUM','NBANK_ORGNUM12','NBANK_ORGNUM11','NBANK_ORGNUM015','NBANK_ALLNUM12',
                  'IR_ID_X_CELL_CNT','ALD_INB_ALLNUM','ALD_INB_OTH_ALLNUM','ALD_INB_ORGNUM','ALD_CNB_ALLNUM','ALD_CNB_OTH_ALLNUM','ALD_CNB_ORGNUM'
                  ]

auto_col_bins = {'NBANK_ALLNUM11':4,'NBANK_ALLNUM015':4, 'PLATFORM_COUNT7天内申请人在多个平台申请借款':3,
                 'PLATFORM_COUNT3个月内申请人在多个平台申请借款':5,'PLATFORM_COUNT1个月内申请人在多个平台申请借款':3,
                'IR_ALLMATCH_DAYS':5,'SCORECONSOFF':5,'SCORECASHOFF':3,'SCORECASHON':5,'SCOREREVOLOAN':4
                 }

hand_col_bins = {'ALD_ID_X_CELL_NUM':[1,2,15],
                 'NBANK_ORGNUM12':[1,2,3,19],
                'NBANK_ORGNUM11':[1,2,3,4,6,19],
                'NBANK_ORGNUM015':[0,1,2,3,17.5],
                'NBANK_ALLNUM12':[0,1,2,3,22],
                  'IR_ID_X_CELL_CNT':[1,2,3,29],
                'ALD_INB_ALLNUM':[1,2,15],
                'ALD_INB_OTH_ALLNUM':[1,2,11],
                'ALD_INB_ORGNUM':[1,2,15],
                'ALD_CNB_ALLNUM':[1,2,15],
                'ALD_CNB_OTH_ALLNUM':[1,2,11],
                'ALD_CNB_ORGNUM':[1,2,15]
}
hand_col_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_col_bins.items()}
hand_col_bins.update(hand_bins)



#计算所有需要手动分箱的变量的IV信息
iv_all2 = cal_iv(df='train_data',target='target',hand_bins=hand_col_bins)


# 生成自动分箱的分箱区间和分箱后的 IV 值
bins_of_col = {}
for col in auto_col_bins:
    bins_df = graphforbestbin(train_data,col
                             ,"target"
                             ,n=auto_col_bins[col]
                             #使用字典的性质来取出每个特征所对应的箱的数量
                             ,q=20
                             ,graph=False)
    bins_list = sorted(set(bins_df["min"]).union(bins_df["max"]))
    #保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
    bins_list[0],bins_list[-1] = -np.inf,np.inf
    bins_of_col[col] = bins_list

iv_all3 = cal_iv(df=train_data,target='target',hand_bins=bins_of_col)
iv_final = iv_all2.append(iv_all3)

#筛选出IV值大于0.05的变量作为训练集和测试集的最终变量
col_final = list(iv_final[iv_final.IV > 0.05].col.unique()) + ['target']

#筛选出最终变量的分箱情况
bins_of_col.update(hand_col_bins)
bins_of_final = {k: v for k, v in bins_of_col.items() if k in col_final}



#筛选最终的测试集和训练集
model_data = train_data[col_final]
vali_data =  test_data[col_final]

model_data.to_csv(r'D:\CreditScorecard\Data\model_data.csv',encoding='gbk',index=False)
vali_data.to_csv(r'D:\CreditScorecard\Data\vali_data.csv',encoding='gbk',index=False)



#-------------------------------------------------------------------------
#调整WOE单调性
#-------------------------------------------------------------------------
def judge_increasing(L):
    """
    :param L: list
    :return: 判断一个List是否单调递增
    """
    return all(x < y for x, y in zip(L,L[1:]))

def judge_decreasing(L):
    """
    :param L: list
    :return: 判断一个list是否单调递减
    """
    return all(x > y for x, y in zip(L, L[1:]))


col = list(iv_final[iv_final.IV > 0.05].col.unique())
bins_of_final_rewoe = bins_of_final.copy()
for cc in col:
    cut = bins_of_final[cc]
    #print(cut)
    woe_lst = iv_final[iv_final['col']==cc].woe
    #print('woe_lst of {} is {}'.format(cc,woe_lst))
    if woe_lst[0] > 0:
        while not judge_decreasing(woe_lst):
            judge_list = [x > y for x, y in zip(woe_lst, woe_lst[1:])]
            #print(judge_list)
            index_list = [i+1 for i, j in enumerate(judge_list) if j == False]
            #print(index_list)
            #new_cut = cut
            new_cut = [j for i,j in enumerate(cut) if i not in index_list]
            #print('new_cut of {} is {}'.format(cc,new_cut))
            bins_of_final_rewoe[cc] = new_cut
            bin_df_cc,iv_value_cc = binning_self(train_data,cc,'target',cut=new_cut)
            woe_lst = bin_df_cc['woe'].tolist()
            cut = new_cut
            #print('new_woe_lst of {} if {}'.format(cc,woe_lst))
    elif woe_lst[0] < 0:
        while not judge_increasing(woe_lst):
            judge_list = [x < y for x, y in zip(woe_lst, woe_lst[1:])]
            #print(judge_list)
            index_list = [i+1 for i, j in enumerate(judge_list) if j == False]
            #print(index_list)
            #new_cut = cut
            new_cut = [j for i,j in enumerate(cut) if i not in index_list]
            #print('new_cut of {} is {}'.format(cc,new_cut))
            bins_of_final_rewoe[cc] = new_cut
            bin_df_cc,iv_value_cc = binning_self(train_data,cc,'target',cut=new_cut)
            woe_lst = bin_df_cc['woe'].tolist()
            cut = new_cut
            #print('new_woe_lst of {} if {}'.format(cc,woe_lst))


iv_rewoe = cal_iv(df='train_data',target='target',hand_bins=bins_of_final_rewoe)


#保存字典文件到本地
f = open(r'D:\CreditScorecard\Data\bins_of_final.txt','w')
f.write(str(bins_of_final_rewoe))
f.close()


#iv_final[iv_final['col']=='SCORECASHON'].woe