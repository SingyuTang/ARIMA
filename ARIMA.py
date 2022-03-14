import numpy as np
import pandas as pd
import akshare as ak
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import draw_acf_pacf as dr
import warnings
warnings.filterwarnings("ignore")#不显示警告信息

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
datapath=r"G:\python\ARIMA\time_series_grace_2002_2017_bj_original.xls"
# datapath=r'G:\GRACE_processing\hbpy2002_2017\time_series_grace_2002_2017.xls'
write_excel_path_predicted=r'G:\python\ARIMA\grace_2002_2017_bj_predict.xls'
# write_excel_path_predicted=r'G:\GRACE_processing\hbpy2002_2017\grace_2002_2017_predict.xls'
##data数据的第一列为x轴，第二列为y轴
data_index=u'时间'
data_value=u'等效水高(cm)'
data=pd.read_excel(datapath,header=0,index_col=data_index)
data_unindex=pd.read_excel(datapath,header=0,index_col=None)#索引不算数据列，故该数据只有一列，这里不设置索引（自动添加自然数）
# data_conindex=data_unindex.set_index(data_unindex[data_index])
# print(data_conindex)
# data1=data_conindex[data_value]
# data2=data_unindex[data_value]
# data11=data1.to_frame()
# column = data11.columns[0]
# data11['index'] = data11.index.tolist()#将索引转为数据列

# data_conindex.plot(data_conindex[data_index],data_conindex[data_value])
# plt.show()
df_data=data
dr.ADF1(df_data)
print('************************************************************************')
dr.ADF1(df_data.diff().dropna())
# print('************************************************************************')
# dr.PP(df_data)
# dr.draw_acf_pacf(df_data)#画图确定p(pacf)，q(acf)
p=2;q=2;d=1


sheet_name='predict result'
pred_time_interval=30/365
start,end=0,173
dr.plot_ARIMA1(datapath,data_index,p,d,q,write_excel_path_predicted,sheet_name,pred_time_interval,start,end)
# print(type(df_data),df_data)
# print(df_data)
# dr.decompose(df_data)

# dataframe=pd.DataFrame({'A':[9.3, 4.3, 4.1, 5.0, 7.0], 'B':[2.5, 4.1, 2.7, 8.8, 1.0]})
# dataframe.plot()
# data_unindex.plot.scatter(x=data_index,y=data_value)
# plt.show()

