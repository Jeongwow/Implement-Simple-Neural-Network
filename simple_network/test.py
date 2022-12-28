import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import OrderedDict


# def init_xt():
#     input_data=np.empty((0,3))
#     t=np.empty((0,3))
#     for i in range(500):
#         input_data=np.append(input_data, np.array([[random.randrange(0,300) for j in range(3)]]),axis=0)
#         x=input_data[-1,0]
#         if x<100:
#             t=np.append(t,np.array([[1, 0, 0]]),axis=0)
#         elif x<200:
#             t=np.append(t,np.array([[0, 1, 0]]),axis=0)
#         elif x<300:
#             t=np.append(t,np.array([[0, 0, 1]]),axis=0)
#         # print(x)

#     return input_data,t

# def init_W(): # 나중에 파일에 w값 저장해놓자.
#     W1=np.empty((0,4))
#     W2=np.empty((0,4))
#     W3=np.empty((0,3))
    
#     for i in range(3):
#         W1=np.append(W1, np.array([np.random.normal(0,1,4)]),axis=0)  #정규분포 랜덤수로 W1생성
    
#     for i in range(4):
#         W2=np.append(W2, np.array([np.random.normal(0,1,4)]),axis=0)  #정규분포 랜덤수로 W2생성
        
#     for i in range(4):
#         W3=np.append(W3, np.array([np.random.normal(0,1,3)]),axis=0)  #정규분포 랜덤수로 W3생성
    
#     # print(W1)
#     # print(W2)
#     return W1,W2,W3



weight_init_std=0.01
params=OrderedDict()
params['W1']=weight_init_std*np.random.randn(3,4)
params['B1']=np.zeros(4)
params['W2']=weight_init_std*np.random.randn(4,4)
params['B2']=np.zeros(4)
params['W3']=weight_init_std*np.random.randn(4,3)
params['B3']=np.zeros(3)

# print(params)


# xyz,t=init_xt()
# w1,w2,w3=init_W()

# xyz_train_set=xyz[0:449,:]
# t_train_set=t[0:499,:]

# xyz_test_set=xyz[450:499,:]
# t_test_set=t[450:499,:]

# print(xyz_train_set.shape[0])
# # print(w3.T)

# a=np.array([[1,2,3,4],[5,6,7,8]])
# b=np.array([[10,20,30],[40,50,60]])
# print(np.dot(a.T,b))

# print(len(a.T))


# c=np.array([1,2,3,4])
# d=np.array([10,20,30])
# # 이게 진짜 히트다.... 차원수가 1이면 reshape해주면 됨.
# if c.ndim==1:
#     c=c.reshape(1,c.size)
# if d.ndim==1:
#     d=d.reshape(1,d.size)
    
# print(np.dot(c.T,d))


# batch_size=5

# batch_mask=np.random.choice(xyz_train_set.shape[0],batch_size)  #랜덤으로 trian_set에서 뽑아옴
# xyz_batch=xyz_train_set[batch_mask]
# t_batch=t_train_set[batch_mask]

# print(xyz_batch)


# xdata = [1,2, 3,4]
# ydata = [10 ,5 ,20 ,35]
# #그래프
# plt.plot(xdata , ydata)
# plt.show()




# Orderdict() 사용 이유
# d = {}
 
# d['z'] = 300
 
# d['y'] = 200
 
# d['x'] = 100
 
# v = {}
 
# v['y'] = 200
 
# v['x'] = 100
 
# v['z'] = 300
 
# print(d)
 
# print(v)
 
# if d == v :
#     print(" Dictionary가 동일")



# print(params)


        
# 변수 파일 저장
# np.savez("simple_network/Data/file.npz", **params)

# 변수 불러오기
# my_dictionary = np.load("simple_network/Data/file.npz", allow_pickle=True)
# paaa=dict(my_dictionary)
# print(paaa)

    
