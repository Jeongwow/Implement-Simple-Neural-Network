import numpy as np
import random

class Affin:
    def __init__(self,W,B):
        self.W=W
        self.B=B
        self.X=None
        self.dW=None
        self.dB=None
        self.A=0.01
    
    def forward(self,X):
        self.X=X
        out=np.dot(X,self.W)+self.B
        return out
    
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        if self.X.ndim==1 and dout.ndim==1:
            tmp=np.empty((int(len(self.X.T)), int(len(dout.T))))
            for i in range(len(self.X.T)):
                for j in range(len(dout)):
                    tmp[i,j]=self.X.T[i]*dout[j]
            self.dW=tmp
            # print(self.dW)
            
        else:
            self.dW=np.dot(self.X.T,dout)
        self.dB=np.sum(dout,axis=0)
        self.W = self.W - self.A*self.dW
        return dx    

class Relu:
    def __init__(self):
        self.mask=None
        
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        
        return out
    
    def backward(self, dout):
        dout[self.mask]=0
        dx=dout
        
        return dx

class Sigmoid:
    def __init__(self):
        self.out=None
        
    def forward(self,X):
        self.out=1/(1+np.exp(-X))
        return self.out
    
    def backward(self, dout):
        return dout*(1.0-self.out)*self.out
    
class softmax_with_loss:
    def __init__(self):
        self.y=None
        self.loss=None
        self.t=None
        
    def forward(self,X,T):
        self.y=softmax(X)  #소프트맥스 시키기
        self.t=T  # 정답 레이블
        self.loss=cross_entropy_error(self.t,self.y)
        return self.loss
    
    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        return dx
    
        
def init_xt():
    input_data=np.empty((0,3))
    t=np.empty((0,3))
    for i in range(500):
        input_data=np.append(input_data, np.array([[random.randrange(0,300) for j in range(3)]]),axis=0)
        x=input_data[-1,0]
        if x<100:
            t=np.append(t,np.array([[1, 0, 0]]),axis=0)
        elif x<200:
            t=np.append(t,np.array([[0, 1, 0]]),axis=0)
        elif x<300:
            t=np.append(t,np.array([[0, 0, 1]]),axis=0)
        # print(x)

    return input_data,t

def init_W(): # 나중에 파일에 w값 저장해놓자.
    W1=np.empty((0,4))
    W2=np.empty((0,4))
    W3=np.empty((0,3))
    
    for i in range(3):
        W1=np.append(W1, np.array([np.random.normal(0,1,4)]),axis=0)  #정규분포 랜덤수로 W1생성
    
    for i in range(4):
        W2=np.append(W2, np.array([np.random.normal(0,1,4)]),axis=0)  #정규분포 랜덤수로 W2생성
        
    for i in range(4):
        W3=np.append(W3, np.array([np.random.normal(0,1,3)]),axis=0)  #정규분포 랜덤수로 W3생성
    
    # print(W1)
    # print(W2)
    return W1,W2,W3

def softmax(x):
    c=np.max(x)     # 오버플로 대책 딥러닝책 p.94
    exp_x=np.exp(x-c)
    sum_exp_x=np.sum(exp_x)
    y=exp_x/sum_exp_x
    return y

def cross_entropy_error(t,y):
    delta=1e-7  # np.log에 0을 넣으면  계산을 할 수 없음 log(0)은 무한대이므로
    return -np.sum(t*np.log(y+delta))
              
def accuracy(y,t):
    p=np.argmax(y)
    if t[p]==1:
        print("성공 : ",y[p])
    else:
        print("실패 : ",y[p])


def main():
    XYZ,t=init_xt() #데이터, 정답 초기화
    W1,W2,W3=init_W()  #가중치 초기화
    
    train_set=XYZ[0:449,:]
    test_set=XYZ[450:499,:]
    # print(train_set)
    # print(test_set)
    # print(XYZ)
    # print(t)

    Layer=[]
    Layer.append(Affin(W1,0))
    Layer.append(Sigmoid())
    Layer.append(Affin(W2,0))
    Layer.append(Sigmoid())
    Layer.append(Affin(W3,0))
    
    LastLayer=softmax_with_loss()  #softmax, loss 레이어
    
    # 처음 50번 돌려서 totalLoss계산
    totalLoss=0
    cnt=0
    print(totalLoss)
    for j in test_set:
        input=j
        now_t=t[cnt]
        for i in range(5):
            input=Layer[i].forward(input)
        
        L=LastLayer.forward(input,now_t)
        totalLoss+=L
    print(totalLoss)
    # 테스트 끝
    
    
    #학습 시작
    cnt=0
    # 순전파 시작
    for j in train_set:
        input=j
        now_t=t[cnt]
        for i in range(5):
            input=Layer[i].forward(input)
        
        L=LastLayer.forward(input,now_t)
        # print(L)
        # 여기까지가 순전파
        
        #역전파
        dout=1 
        dout=LastLayer.backward(dout)
        for i in reversed(range(5)):
            dout=Layer[i].backward(dout)
        cnt+=1
        #역전파 끝
    #학습 끝
    
    # 처음 50번 돌려서 totalLoss계산
    cnt=0
    totalLoss=0
    print(totalLoss)
    for j in test_set:
        input=j
        now_t=t[cnt]
        for i in range(5):
            input=Layer[i].forward(input)
        
        L=LastLayer.forward(input,now_t)
        totalLoss+=L
    print(totalLoss)
    # 테스트 끝
    
    
    
    
    


if __name__ == '__main__':
    main()
