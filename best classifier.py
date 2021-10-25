# 二分类采样画图
# -*- coding:utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

Prior_p = 0.1
Prior_n = 0.9
conv = np.array([[1, 0], [0, 1]])    # 协方差
risk = np.array([[0, 1], [1000, 0]])  # 风险矩阵


# 绘制散点
def show_scatter(pos, neg, size):
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(pos.data.numpy()[:, 0], pos.data.numpy()[:, 1], marker='*', color='r', s=50,
                label='类别1：非携带者')
    plt.scatter(neg.data.numpy()[:, 0], neg.data.numpy()[:, 1], marker='+', color='b', s=50,
                label='类别-1：携带者')
    plt.legend()

def show_onecircle(x_p, y_p,r,setcolor):
    x_p_temp1 = np.arange(x_p - r, x_p + r, 0.0001)
    y_p_temp1 = np.sqrt(np.power(r, 2) - np.power((x_p_temp1 - x_p), 2)) + y_p
    x_p_temp2 = np.arange(x_p - r, x_p + r, 0.0001)
    y_p_temp2 = -1*np.sqrt(np.power(r, 2) - np.power((x_p_temp2 - x_p), 2)) + y_p
    plt.plot(x_p_temp1, y_p_temp1, x_p_temp2, y_p_temp2, color=setcolor, linestyle=':')
    plt.axis('equal')
    
# 绘制一倍，二倍， 三倍标准差圆环
def show_circle(x_p, y_p, x_n, y_n, r):
    x = [x_p, x_n]
    y = [y_p, y_n]
    for i in range(1, r + 1):
        x_p_temp1 = np.arange(x_p - i, x_p + i, 0.0001)
        y_p_temp1 = np.sqrt(np.power(i, 2) - np.power((x_p_temp1 - x_p), 2)) + y_p
        x_p_temp2 = np.arange(x_p - i, x_p + i, 0.0001)
        y_p_temp2 = -1*np.sqrt(np.power(i, 2) - np.power((x_p_temp2 - x_p), 2)) + y_p

        x_n_temp1 = np.arange(x_n - i, x_n + i, 0.0001)
        y_n_temp1 = np.sqrt(np.power(i, 2) - np.power((x_n_temp1 - x_n), 2)) + y_n
        x_n_temp2 = np.arange(x_n - i, x_n + i, 0.0001)
        y_n_temp2 = -1*np.sqrt(np.power(i, 2) - np.power((x_n_temp2 - x_n), 2)) + y_n
        plt.plot(x_p_temp1, y_p_temp1, x_p_temp2, y_p_temp2, color="#808080", linestyle=':')
        plt.plot(x_n_temp1, y_n_temp1, x_n_temp2, y_n_temp2, color="#808080", linestyle=':')
    plt.plot(x, y, 'o',color='#808080')
    plt.axis('equal')

# 绘制最小错误曲线和最小风险曲线
def decision_line(ui,uj,sigma,pwi,setcolor,setlinestyle,settext=''):
    pwj=1-pwi;
    #最小错误曲线
    w=ui-uj;
    uiuj2=math.pow(np.linalg.norm(w, 2),2);
    x0=0.5*(ui+uj)-1/uiuj2*math.log(pwi/pwj)*w;

    hor_x1 = np.linspace(-2, 10, 500)
    hor_x2 = x0[1]-1.0*w[0]/w[1]*(hor_x1-x0[0])
    print("%lfx+y %lf"%(1.0*w[0]/w[1],-x0[1]-1.0*w[0]/w[1]*x0[0]))
    plt.plot(hor_x1, hor_x2, color=setcolor, linewidth=1.0,linestyle=setlinestyle,label=settext)
    plt.axis('equal')
    
# 绘制最小错误曲线和最小风险曲线
def decision_plane(u_p, u_n, cov, risk_matrix):
    cov_inv = np.linalg.inv(cov)
    # 求ln( P(w1) / P(w2) )
    ln_pw = math.log(Prior_p / Prior_n, math.e)
    ln_risk = math.log((risk_matrix[0][1] - risk_matrix[1][1]) / (risk_matrix[1][0] - risk_matrix[0][0]), math.e)
    x0 = 0.5*(u_p + u_n) - ln_pw*((u_p - u_n)/np.dot(np.dot(np.transpose(u_p - u_n), cov_inv), u_p - u_n))
    w = np.dot(cov_inv, u_p - u_n)
    k = -w[0]/w[1]
    b_error = x0[1] - k*x0[0]
    b_risk = x0[1] - k*x0[0] + ln_risk / w[1]
    hor_x1 = np.linspace(-2, 10, 500)
    plt.plot(hor_x1, k*hor_x1 + b_error, color='green', linewidth=1.0)
    print("最小错误率分类边界 function: y=%lf*x+%lf" % (k, b_error))
    plt.plot(hor_x1, k*hor_x1 + b_risk, color='orange', linewidth=1.0)
    print("最小风险分类边界 function: y=%lf*x+%lf" % (k, b_risk))
    plt.text(0, 9.7, r'最小错误率分类边界', fontdict={'size': '10'})
    plt.text(-2.4, 6.2, r'最小风险分类边界', fontdict={'size': '10'})


# 根据样本点估计期望
def cal_mean(pos, neg, size_p, size_n):
    sum_p = pos.data.numpy()[0]
    sum_n = neg.data.numpy()[0]
    for i in range(1, size_p):
        sum_p += pos.data.numpy()[i]
    for j in range(1, size_n):
        sum_n += neg.data.numpy()[j]
    cal_mean_p = sum_p / size_p
    cal_mean_n = sum_n / size_n
    return cal_mean_p, cal_mean_n

###################################### 概率分布已知情况下的贝叶斯决策 ##############################################


data_size = 100                      # 每类样本数量
mean_p = np.array([6, 7])            # 正类（非携带者）样本均值
mean_n = np.array([2, 3])            # 负类（携带者）样本均值

data_size_p = np.random.binomial(data_size,Prior_p,1)[0]
data_size_p = (int)(Prior_p*data_size)+2#人工设定
data_size_n = data_size-data_size_p
print("mean_p=%d,mean_n=%d"%(data_size_p,data_size_n))
axis_p = np.random.multivariate_normal(mean=mean_p, cov=conv, size=data_size_p)
axis_n = np.random.multivariate_normal(mean=mean_n, cov=conv, size=data_size_n)
axis_p = torch.from_numpy(axis_p)
axis_n = torch.from_numpy(axis_n)

f1 = plt.figure(1)
show_circle(mean_p[0], mean_p[1], mean_n[0], mean_n[1], 3)
show_scatter(axis_p, axis_n, 50)
plt.xlim(1, 10)
plt.ylim(0, 12)
decision_plane(mean_p, mean_n, conv, risk)
#decision_line(mean_p,mean_n,1,Prior_p,'#808080','-')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.show()

################################### 概率分布(期望)未知情况下的贝叶斯决策 ######################################

mean = cal_mean(axis_p, axis_n, data_size_p, data_size_n)
mean_p_new = mean[0]
mean_n_new = mean[1]

f2 = plt.figure(2)
#show_circle(mean_p[0], mean_p[1], mean_n[0], mean_n[1], 3)
show_scatter(axis_p, axis_n, 50)
x = [mean_p[0], mean_n[0]]
y = [mean_p[1], mean_n[1]]
plt.plot(x, y, 'o',color='#808080')
#show_onecircle(x_p, y_p,r,setcolor)
decision_line(mean_p,mean_n,1,Prior_p,'#808080',':','最小错误率分类边界')
x = [mean_p_new[0], mean_n_new[0]]
y = [mean_p_new[1], mean_n_new[1]]
plt.plot(x, y, 'o',color='#000000')
print("均值1：%lf,%lf，均值2：%lf,%lf"%(mean_p_new[0],mean_p_new[1],mean_n_new[0],mean_n_new[1]))
decision_line(mean_p_new,mean_n_new,1,1.0*data_size_p/data_size,'#000000','-','学习出的分类边界')
plt.legend()
plt.xlim(1, 10)
plt.ylim(0, 12)
plt.show()
################################### 直线分类曲线 ######################################
f3 = plt.figure(3)
#show_circle(mean_p[0], mean_p[1], mean_n[0], mean_n[1], 3)
show_scatter(axis_p, axis_n, 50)
#x = [mean_p[0], mean_n[0]]
#y = [mean_p[1], mean_n[1]]
#plt.plot(x, y, 'o',color='#808080')
#show_onecircle(x_p, y_p,r,setcolor)
decision_line(mean_p,mean_n,1,Prior_p,'#808080',':','学习出的分类边界')
#x = [mean_p_new[0], mean_n_new[0]]
#y = [mean_p_new[1], mean_n_new[1]]
#plt.plot(x, y, 'o',color='#000000')
#decision_line(mean_p_new,mean_n_new,1,Prior_p,'#000000','-')

x = np.linspace(0, 20, 1000)
y_f = -0.14 * (x**2) + 0.6 * (x) + 5.4
plt.plot(x, y_f, color='purple', linewidth=1, label="九次分类边界")
plt.legend()

plt.xlim(1, 10)
plt.ylim(0, 12)
plt.show()

