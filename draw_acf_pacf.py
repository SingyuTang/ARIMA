import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
import xlwt

#绘制acf和pacf图
def draw_acf_pacf(data):
    """
    输入需要求解ACF\PACF的数据,datafreme格式,共两列，一列时间一列值
    """
    num=len(data.index)#获取数据个数
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 模型的平稳性检验
    """时序图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig,ax=plt.subplots(2,2)
    fig.subplots_adjust(hspace=0.5)

    #一阶差分
    print('正在一阶差分')
    head=data.columns
    df1=data[head[0]].diff().dropna()

    # print(df1)
    #原始时序图
    print('正在绘制时序图')
    data.plot(ax=ax[0][0])
    plt.title("时序图")

    #一阶差分图
    print('正在绘制一阶差分图')
    df1.plot(ax=ax[0][1])
    plt.title('一阶差分图')
    print('正在绘制pacf')
    """data PACF"""
    plot_pacf(data, lags=num / 2 - 1, ax=ax[1][0])
    ax[1][0].xaxis.set_ticks_position('bottom')
    print('正在绘制acf')
    """data残差ACF"""
    plot_acf(df1, lags=num/2-1, ax=ax[1][1])
    ax[1][1].xaxis.set_ticks_position('bottom')

    fig.tight_layout();
    plt.show()
#adf和pp检验
def ADF1(data):
    adf=ADF(data)
    print(adf.summary().as_text())
def ADF2(data):
    adf=adfuller(data)
    print(adf)
def PP(data):
    pp=PhillipsPerron(data)
    print(pp.summary().as_text())
#白噪声检验
def ACORR(data):
    pass
#绘制ARIMA时序图
def plot_ARIMA1(datapath,index_key,p,d,q,write_excel_path,sheet_name,pred_time_interval,start,end):
    '''

    :param datapath: 要预测的数据，必须是excel类型，xls后缀
    :param index_key: 索引列列名，如‘时间’
    :param p: 自回归阶数
    :param d: 差分次数
    :param q: 移动平均阶数
    :param write_excel_path: 预测结果保存的文件路径，excel文件，xls后缀，包含三列数据，第一列序号（0-N），第二列时间，第三列拟合或者预测数据
    :param sheet_name: sheet表表名
    :param pred_time_interval: 原始数据中的时间间隔，比如月数据该值就为30/365，单位都为归一化的年
    :param start: 预测数据开始时间
    :param end: 预测数据结束时间，需大于原始数据，如原始数据有100个，则该值需要大于等于100（从0开始算，第100个刚好是第一个预测值），设置99不行
    :return:
    '''
    # pred_time_interval=30/365
    # start,end=0,173
    data=pd.read_excel(datapath,header=0,index_col=index_key)#指定index_col为索引
    data_unIndex = pd.read_excel(datapath, header=0, index_col=None)#未指定索引，默认1-N为索引值
    data_Index_name,data_value_name=data_unIndex.columns[0],data_unIndex.columns[1]#索引列列名和数据值列名
    print(len(data_unIndex[data_Index_name]))
    if len(data_unIndex[data_Index_name])<=end:
        model = ARIMA(data, order=(p, d, q))  # order=(p,d,q)
        model_fit = model.fit()
        # print(model_fit.summary())#统计信息
        coreFrame=model_fit.predict().to_frame()#预测值和原始数据
        coreFrame['original data']=data[data_value_name] #向frame中添加数据列时要保证其索引值相同，这里data和coreFrame的索引都是日期
        r_2=r2_score(coreFrame['original data'],coreFrame['predicted_mean'])
        R=format(math.sqrt(r_2),'.3f')#format函数设置保留小数位数
        RMSE=format(mean_squared_error(coreFrame['original data'],coreFrame['predicted_mean']),'.6f')
        # print(mean_square_error)
        # print('原始数据\n',data_unIndex[data_value])
        # print('预测数据\n',model_fit.forecast(10))#参数为预测步长
        fig, ax = plt.subplots(2, 2)
        fig01=coreFrame.plot.scatter(y='predicted_mean', x='original data',ax=ax[0][1],fontsize=10,title='scatterplot')
        # model_fit.forecast(10).plot()
        # model_fit.predict(0,173).plot()
        # print(plt.setp(fig01))
        scatter_text='R={}\nRMSE={}'.format(R,RMSE)
        ax[0][1].text(0.2,-0.1,scatter_text)#计算决定系数和均方根误差后在散点图上标注
        model_fit.predict().plot(color='r',ax=ax[0][0],legend='predict value',fontsize=10)
        data.plot(color='b',ax=ax[0][0],title='time series',legend='real value',fontsize=10)
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot(title="Residuals", ax=ax[1][0],fontsize=10,legend=None)
        residuals.plot(kind='kde', title='Density', ax=ax[1][1],fontsize=10,legend=None)
        fig.tight_layout()
        plt.show()
        predicted_value=model_fit.predict(start, end)
        predicted_time=data_unIndex[data_Index_name]
        diff_time_value=len(predicted_value)-len(predicted_time)#预测值的value比time多出的元素个数
        print('---------------------------------------------------------------------------------------------------')
        # print(predicted_time)#data中的时间当成了索引值不能输出，只能用data_Index中的数据导出时间
        # print(predicted_value)
        pred_cat=pd.concat([predicted_time,predicted_value],axis=1)#按照索引的并集方式拼接，按列拼接
        # mean_time_step=predicted_time.diff().dropna().mean()
        # print(predicted_time.diff().dropna())
        nan_loc=np.where(np.isnan(pred_cat))#NAN的位置，时间缺失值的位置（预测时间的位置缺失）
        nan_row,nan_col=nan_loc[0],nan_loc[1]
        count_data=len(predicted_time)#原始数据个数
        len_nan=len(nan_row)
        predicted_time_2_list=[]
        original_last_data_time=predicted_time.iloc[len(predicted_time)-1]#原始数据最后一个数据的时间
        for i in range(diff_time_value):
            time_tmp=original_last_data_time+(i+1)*pred_time_interval
            predicted_time_2_list.append(time_tmp)
        predicted_time_2_index=[]
        for tmp in range(len_nan):
            index_tmp=tmp+count_data
            predicted_time_2_index.append(str(index_tmp))
        predicted_time_2=pd.Series(predicted_time_2_list,index=predicted_time_2_index)#预测值的时间，即在原始数据后面追加的数据
        predicted_time_sum=pd.concat([predicted_time,predicted_time_2])#行拼接
        predicted_value_list=predicted_value.tolist()
        predicted_time_sum_list=predicted_time_sum.tolist()
        predicted_time_value_list=[predicted_time_sum_list]+[predicted_value_list]
        # print(predicted_time_value_list)
        print(data_Index_name,data_value_name)
        predicted_time_sum_list_str = [str(x) for x in predicted_time_sum_list]#将数字列表转为字符串列表
        predicted_time_value_sum_df=pd.DataFrame(pd.DataFrame(data=predicted_time_value_list).values.T)#先将两行列表转置为两列（时间，等效水高），此时为numpy.ndarray类型，再用pd.DataFrame函数转为DataFrame类型方便写入
        predicted_time_value_sum_df.columns=[data_Index_name,data_value_name]
        print(predicted_time_value_sum_df)
        predicted_time_value_sum_df.to_excel(write_excel_path,sheet_name=sheet_name)
        # predicted_value.plot()
        # plt.show()
        print('预测值文件写入完成。')
    else:
        print('输入的end必须大于等于数据的个数，请重新设置')
#季节性评价
def decompose(data):
    seasonal_decompose(data).plot()
    plt.show()
    decomposition = seasonal_decompose(data) # 画出分解后时序图
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # fig = plt.figure()
    # ax1 = fig.add_subplot(411)
    # ax1.plot(data, label='Original')
    # ax1.legend(loc='best')
    # ax2 = fig.add_subplot(412)
    # ax2.plot(trend, label='Trend')
    # ax2.legend(loc='best')
    # ax3 = fig.add_subplot(413)
    # ax3.plot(seasonal, label='Seasonality')
    # ax3.legend(loc='best')
    # ax4 = fig.add_subplot(414)
    # ax4.plot(residual, label='Residuals')
    # ax4.legend(loc='best')
    # fig.tight_layout()
    # plt.show()
    return trend,seasonal,residual
