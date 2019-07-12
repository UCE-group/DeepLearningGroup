import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv', encoding='gbk')

pm2_5 = data[data['class']=='PM2.5'].ix[:, 3:]

tempxlist = []
tempylist = []
for i in range(15):
    tempx = pm2_5.iloc[:, i:i+9]
    tempx.columns = np.array(range(9))
    tempy = pm2_5.iloc[:, i+9]
    tempy.columns = ['1']
    tempxlist.append(tempx)
    tempylist.append(tempy)
xdata = pd.concat(tempxlist)
x = np.array(xdata, float)
ydata = pd.concat(tempylist)
y = np.array(ydata, float)

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

w = np.zeros((len(x[0])))
lr = 10
iteration = 10000
s_grad = np.zeros(len(x[0]))
for i in range(iteration):
    tem = np.dot(x, w)
    loss = y - tem
    grad = np.dot(x.transpose(), loss)*(-2)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - lr*grad/ada

testdata = pd.read_csv('test.csv', encoding='gbk')
pm2_5_test = testdata[testdata['class']=='PM2.5'].ix[:, 2:]
x_test = np.array(pm2_5_test, float)
x_test_b = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
y_star = np.dot(x_test_b, w)
y_pre = pd.read_csv('sampleSubmission.csv', encoding='gbk')
y_pre.value = y_star

real = pd.read_csv('ans.csv', encoding='gbk')
erro = abs(y_pre.value - real.value).sum()/len(real.value)
print(erro)

