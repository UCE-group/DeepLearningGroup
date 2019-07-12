import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from numba import jit

# @jit
def test():
    '''读取数据'''
    data = pd.read_csv('train.csv')   #DataFrame类型
    # print(data)
    # # del data[:3]
    del data['datetime']
    del data['stations']
    del data['observations']
    data.drop([0])
    # print(data)
    data = data.replace('NR', 0)
    # print(data)
    # print(data)


    '''整理训练集合'''
    ItemNum=18
    X_Train=[]        #训练样本features集合
    Y_Train=[]        #训练样本目标PM2.5集合
    for i in range(int(len(data)/ItemNum)):
        day = data[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
        for j in range(15):
            x = day.iloc[:, j:j + 9]
            y = int(day.iloc[9,j+9])
            X_Train.append(x)
            Y_Train.append(y)


    '''绘制散点图'''
    x_AMB_TEMP=[]
    x_CH4=[]
    x_CO=[]
    x_NMHC=[]
    y=[]
    for i in range(len(Y_Train)):
        y.append(Y_Train[i])
        x=X_Train[i]
        #求各测项的平均值
        x_AMB_TEMP_sum=0
        x_CH4_sum=0
        x_CO_sum=0
        x_NMHC_sum=0
        for j in range(9):
            # print(x.iloc[0,j])
            x_AMB_TEMP_sum=x_AMB_TEMP_sum+float(x.iloc[0,j])
            # print(x.iloc[1, j])
            x_CH4_sum = x_CH4_sum + float(x.iloc[1, j])
            x_CO_sum = x_CO_sum + float(x.iloc[2, j])
            x_NMHC_sum = x_NMHC_sum + float(x.iloc[3, j])
        x_AMB_TEMP.append(x_AMB_TEMP_sum / 9)
        x_CH4.append(x_CH4_sum / 9)
        x_CO.append(x_CO_sum / 9)
        x_NMHC.append(x_NMHC_sum / 9)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.title('AMB_TEMP')
    plt.scatter(x_AMB_TEMP, y)
    plt.subplot(2, 2, 2)
    plt.title('CH4')
    plt.scatter(x_CH4, y)
    plt.subplot(2, 2, 3)
    plt.title('CO')
    plt.scatter(x_CO, y)
    plt.subplot(2, 2, 4)
    plt.title('NMHC')
    plt.scatter(x_NMHC, y)
    plt.show()

    '''小批量梯度下降'''
    dict={0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:8,8:8,9:9,10:9,11:9,12:9,13:9,14:9,15:9,16:9,17:9,18:12,19:12,20:12,21:12,22:12,23:12,24:12,25:12,26:12}
    iteration_count = 10   #迭代次数
    learning_rate = 0.000001  #学习速率
    b=0.0001    #初始化偏移项
    parameters=[0.001]*27     #初始化27个参数
    loss_history=[]
    for i in range(iteration_count):
        loss=0
        b_grad=0
        w_grad=[0]*27
        examples=list(randint(0, len(X_Train)-1) for index in range(100))
        print("标记1",i,"\n")
        for j in range(100):
            index=examples.pop()
            day = X_Train[index]
            # print(day)
            # # print(parameters[0])
            # print(day.iloc[8,0])
            # print(day.iloc[8,1])
            # print(type(day.iloc[8,1]))

            
            # print(day.iloc[8,:])
            # print(day.iloc[9,:])
            # print(day.iloc[12,:])
            # # print(day.iloc[8,1])
            # # print(day.iloc[8,2])
            # # print(day.iloc[8,1])

            partsum = b+parameters[0]*float(day.iloc[8,0])+parameters[1]*float(day.iloc[8,1])+parameters[2]*float(day.iloc[8,2])+parameters[3]*float(day.iloc[8,3])+parameters[4]*float(day.iloc[8,4])+parameters[5]*float(day.iloc[8,5])+parameters[6]*float(day.iloc[8,6])+parameters[7]*float(day.iloc[8,7])+parameters[8]*float(day.iloc[8,8])+parameters[9]*float(day.iloc[9,0])+parameters[10]*float(day.iloc[9,1])+parameters[11]*float(day.iloc[9,2])+parameters[12]*float(day.iloc[9,3])+parameters[13]*float(day.iloc[9,4])+parameters[14]*float(day.iloc[9,5])+parameters[15]*float(day.iloc[9,6])+parameters[16]*float(day.iloc[9,7])+parameters[17]*float(day.iloc[9,8])+parameters[18]*float(day.iloc[12,0])+parameters[19]*float(day.iloc[12,1])+parameters[20]*float(day.iloc[12,2])+parameters[21]*float(day.iloc[12,3])+parameters[22]*float(day.iloc[12,4])+parameters[23]*float(day.iloc[12,5])+parameters[24]*float(day.iloc[12,6])+parameters[25]*float(day.iloc[12,7])+parameters[26]*float(day.iloc[12,8])-Y_Train[index]
            loss=loss + partsum * partsum
            b_grad = b_grad + partsum
            for k in range(27):
                # print(day.iloc[dict[k],k % 9])
                w_grad[k]=w_grad[k]+ partsum * float(day.iloc[dict[k],k % 9])
        loss_history.append(loss/2)
        #更新参数
        b = b - learning_rate * b_grad
        for t in range(27):
            parameters[t] = parameters[t] - learning_rate * w_grad[t]
    # print("jieshu ")

    '''小批量梯度下降'''
    dict={0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:8,8:8,9:9,10:9,11:9,12:9,13:9,14:9,15:9,16:9,17:9,18:12,19:12,20:12,21:12,22:12,23:12,24:12,25:12,26:12}
    iteration_count = 10   #迭代次数
    learning_rate = 0.000001  #学习速率
    b=0.0001    #初始化偏移项
    parameters=[0.001]*27     #初始化27个参数
    loss_history=[]
    for i in range(iteration_count):
        print("标记2",i,"\n")
        loss=0
        b_grad=0
        w_grad=[0]*27
        examples=list(randint(0, len(X_Train)-1) for index in range(100))
        for j in range(100):
            index=examples.pop()
            day = X_Train[index]
            partsum = b+parameters[0]*float(day.iloc[8,0])+parameters[1]*float(day.iloc[8,1])+parameters[2]*float(day.iloc[8,2])+parameters[3]*float(day.iloc[8,3])+parameters[4]*float(day.iloc[8,4])+parameters[5]*float(day.iloc[8,5])+parameters[6]*float(day.iloc[8,6])+parameters[7]*float(day.iloc[8,7])+parameters[8]*float(day.iloc[8,8])+parameters[9]*float(day.iloc[9,0])+parameters[10]*float(day.iloc[9,1])+parameters[11]*float(day.iloc[9,2])+parameters[12]*float(day.iloc[9,3])+parameters[13]*float(day.iloc[9,4])+parameters[14]*float(day.iloc[9,5])+parameters[15]*float(day.iloc[9,6])+parameters[16]*float(day.iloc[9,7])+parameters[17]*float(day.iloc[9,8])+parameters[18]*float(day.iloc[12,0])+parameters[19]*float(day.iloc[12,1])+parameters[20]*float(day.iloc[12,2])+parameters[21]*float(day.iloc[12,3])+parameters[22]*float(day.iloc[12,4])+parameters[23]*float(day.iloc[12,5])+parameters[24]*float(day.iloc[12,6])+parameters[25]*float(day.iloc[12,7])+parameters[26]*float(day.iloc[12,8])-Y_Train[index]
            loss=loss + partsum * partsum
            b_grad = b_grad + partsum
            for k in range(27):
                w_grad[k]=w_grad[k]+ partsum * float(day.iloc[dict[k],k % 9])
        loss_history.append(loss/2)
        #更新参数
        b = b - learning_rate * b_grad
        for t in range(27):
            parameters[t] = parameters[t] - learning_rate * w_grad[t]

    '''评价模型'''
    data1 = pd.read_csv('test.csv')
    del data1['id']
    del data1['item']
    # data.drop([0], axis=1)


    X_Test=[]
    ItemNum=18
    for i in range(int(len(data1)/ItemNum)):
        day = data1[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
        X_Test.append(day)
    Y_Test=[]
    data2 = pd.read_csv('answer.csv')
    for i in range(len(data2)):
        Y_Test.append(data2.iloc[i,1])
    b=0.00371301266193
    parameters=[-0.0024696993501677625, 0.0042664323568029619, -0.0086174899917209787, -0.017547874680980298, -0.01836289806786489, -0.0046459546176775678, -0.031425910733080147, 0.018037490234208024, 0.17448898242705385, 0.037982590870111861, 0.025666115101346722, 0.02295437149703404, 0.014272058968395849, 0.011573452230087483, 0.010984971346586308, -0.0061003639742210781, 0.19310213021199321, 0.45973205224805752, -0.0034995637680653086, 0.00094072189075279807, 0.00069329550591916357, 0.002966257320079194, 0.0050690506276038138, 0.007559004246038563, 0.013296350700555241, 0.027251049329127801, 0.039423988570899793]
    Y_predict=[]
    for i in range(len(X_Test)):
        day=X_Test[i]
        p=b+parameters[0]*float(day.iloc[8,0])+parameters[1]*float(day.iloc[8,1])+parameters[2]*float(day.iloc[8,2])+parameters[3]*float(day.iloc[8,3])+parameters[4]*float(day.iloc[8,4])+parameters[5]*float(day.iloc[8,5])+parameters[6]*float(day.iloc[8,6])+parameters[7]*float(day.iloc[8,7])+parameters[8]*float(day.iloc[8,8])+parameters[9]*float(day.iloc[9,0])+parameters[10]*float(day.iloc[9,1])+parameters[11]*float(day.iloc[9,2])+parameters[12]*float(day.iloc[9,3])+parameters[13]*float(day.iloc[9,4])+parameters[14]*float(day.iloc[9,5])+parameters[15]*float(day.iloc[9,6])+parameters[16]*float(day.iloc[9,7])+parameters[17]*float(day.iloc[9,8])+parameters[18]*float(day.iloc[12,0])+parameters[19]*float(day.iloc[12,1])+parameters[20]*float(day.iloc[12,2])+parameters[21]*float(day.iloc[12,3])+parameters[22]*float(day.iloc[12,4])+parameters[23]*float(day.iloc[12,5])+parameters[24]*float(day.iloc[12,6])+parameters[25]*float(day.iloc[12,7])+parameters[26]*float(day.iloc[12,8])
        Y_predict.append(p)
    def dev_degree(y_true,y_predict):    #评价函数
        sum=0
        for i in range(len(y_predict)):
            sum=sum+(y_true[i]-y_predict[i])*(y_true[i]-y_predict[i])
        return sum/len(y_predict)
    print(dev_degree(Y_Test,Y_predict))

def main():
    test()

if __name__ == '__main__':
    main()