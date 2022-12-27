import numpy as np
import random

class Affin:
    def __init__(self,W,B):
        self.W=W
        self.X=None
        self.dW=None
    
    def forward(self,X):
        self.X=X
        return np.dot(X,self.W)
    
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.X.T,dout)
        return dx    
    
    def update(self,a=0.01):
        self.W=a*self.dW

class Sigmoid:
    def __init__(self):
        self.out=None
        
    def forward(self,X):
        self.out=1/(1+np.exp(-X))
        return self.out
    
    def backward(self, dout):
        return dout*(1-self.out)*self.out
    
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
              

def main():
    XYZ,t=init_xt() #데이터, 정답 초기화
    W1,W2,W3=init_W()  #가중치 초기화
    
    dout=1     # 순전파때 backward에 넣을 변수 ??????? 아닌듯.. 이해못한듯
    # 일단 dw(local gradient)를 저장해놓으려고 
    #순전파때는 1넣고 ㄱㄱ
    
    AFF=[]
    AFF.append(Affin(W1))
    AFF.append(Affin(W2))
    AFF.append(Affin(W3))
    
    SIG=[]
    SIG.append(Sigmoid())
    SIG.append(Sigmoid())
    
    LastLayer=softmax_with_loss()  #softmax, loss 레이어

    
    Aff1=Affin(W1,0) # affine 레이어
    Aff2=Affin(W2,0)
    Aff3=Affin(W3,0)
    Sig1=Sigmoid()  # sigmoid 레이어
    Sig2=Sigmoid()
    LastLayer=softmax_with_loss()  #softmax, loss 레이어
    
    # 순전파 시작
    A1=Aff1.forward(XYZ)
    Z1=Sig1.forward(A1)
    
    A2=Aff2.forward(Z1)
    Z2=Sig2.forward(A2)
    
    A3=Aff3.forward(Z2)
    L=LastLayer.forward(A3,t)
    print(LastLayer.loss)
    # 여기까지가 순전파
    
    #역전파 연습
    dout=1
    dout=LastLayer.backward(dout)
    dout=Aff3.backward(dout)
    dout=Sig2.backward(dout)
    dout=Aff2.backward(dout)
    dout=Sig1.backward(dout)
    dout=Aff1.backward(dout)
    
    Aff1.update()
    Aff2.update()
    Aff3.update()

      # 순전파 시작
    A1=Aff1.forward(XYZ)
    Z1=Sig1.forward(A1)
    
    A2=Aff2.forward(Z1)
    Z2=Sig2.forward(A2)
    
    A3=Aff3.forward(Z2)
    L=LastLayer.forward(A3,t)
    print(LastLayer.loss)
    # 여기까지가 순전파
    
    #--------------------------------------------------------
    
    
    
    
    
    
    

    
    


if __name__ == '__main__':
    main()
