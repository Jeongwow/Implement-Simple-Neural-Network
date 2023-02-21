# Neural-Network

간단한 Neural Network모델을 구현.

활동 내용 요약
1.	Neural Network 모델 개발 과정 및 문제점 분석.
2.	Hyper Parameter 및 가중치 초기화 등을 조정하여 정확도 및 Loss 결과 분석.
3.	세미나를 진행하여 학습 내용을 공유하였고, 부족한 부분에 대해 피드백을 받음.


<본문>
1	Neural Network 모델 개발 과정 및 문제점 분석.
1.1	문제 및 모델 
<img width="500" alt="image" src="https://user-images.githubusercontent.com/26295029/220339730-6d3c1632-447a-417e-a4c6-cb79cc0fdc45.png">

사진 14. 문제 및 모델 설명

간단한 문제를 해결할 수 있는 Neural Network 모델을 만들어 보라는 과제를 받아 진행함. 
(손으로 먼저 진행 과정을 정리해보고 코드화하기)

<img width="500" alt="image" src="https://user-images.githubusercontent.com/26295029/220339808-d511c184-e09d-4ac6-93d1-6ac79f7c2970.png">

사진 15. 모델 구현 전 진행 과정 정리

1.2	위 문제를 해결하는 Neural Network 모델 개발.  22.12.26~22.12.28

이전에 공부한 지식을 바탕으로 간단한 Neural Network 모델 구현했음.
초기 모델의 정확도 73%에서 수렴하였음.
<img width="340" alt="image" src="https://user-images.githubusercontent.com/26295029/220339880-6c5b38a3-e9f7-4a8a-89f9-2d01c50b3218.png">

사진 16. 최초 모델 구현 성공 후 결과 확인

이전에 공부한 딥러닝 지식들을 바탕으로 직접 간단한 Neural Network 모델을 구현하는 과정을 진행함. 아무 자료 없이 혼자의 힘으로 구현하는 것이 목표였지만 학습 모델을 직접 구현하는 것이 처음이었기 때문에 다소 어려움을 겪음.

구현 과정에서 마주친 문제점

1. 지금까지 딥러닝에 대해 개념적으로만 공부하였고, 실제 코드에서는 어떻게 구현하는지에 대해서는 깊게 공부하지 않았기 때문에 코드의 틀을 잡는 것에서 어려움이 있었음.
2. 역전파 개념에 대해 잘 이해했다고 생각했지만 손실함수의 미분과 softmax, sigmoid의 미분을 코드로써 연결하는 것에서 어려움이 있었음.
3. 파이썬 언어를 자주 사용하지 않았기 때문에 코드 구현에 다소 시간이 걸렸음.

모델을 구현하며 느낀점 : 
간단한 Neural Network 모델을 만드는 과정을 진행해보며 이미 알고 있다고 생각했던 부분들도 직접 코드로 구현하는 것은 쉽지 않은 일이고, 간단한 모델이라 하더라도 구현하며 고려해야할 부분이 생각 보다 많다는 것을 깨닫는 계기가 되었음


2	Hyper Parameter 및 가중치 초기화 등을 조정하여 정확도 및 Loss 결과 분석
2.1	모델 구현 후 추가 미션
간단한 학습 모델 구현에 성공한 후 처음 측정한 정확도는 대략 73%에서 수렴하는 결과를 보였음. 이후 임지혜 연구생이 추가 연구 사항을 전달해 주었음. 전달받은 내용은 구현한 코드를 통해 실험을 진행함.

추가 미션
1. Data set 고정
2. Input Data 정규화
3. 가중치 초기화를 조정하여 결과 분석(상수, 정규분포를 따르는 랜덤수, 균일분포를 따르는 랜덤수로 초기화)
4. Batch Size에 따른 결과 분
5. Epoch를 조정하여 under fitting, over fitting 확인하기.
6. Data set을 Train, Validation, Test set으로 구분하여 결과 분석

추가 미션에 대한 결과 분석을 위해 파이썬으로 구현한 코드를 Jupyter Notebook을 통해 정리하여 연구 및 분석을 진행함.


2.2	Hyper Parameter 및 가중치 초기화 등을 조정하여 정확도 및 Loss 분석. 23.1.2
직접 구현한 Neural Network 모델을 통해 딥러닝 과정에서 Hyper Parameter의 설정 및 Input Data의 형태에 따른 결과를 분석하였음.
아래 세가지 작업을 통해 결과를 분석하는 과정을 진행함.

1. 모델 데이터 셋 증가(1000개의 랜덤 수  5000개의 랜덤 수)
2. Input Data 정규화 이유 찾기(학습 전/후 가중치 분포를 비교하여)
3. Input Data를 정답 별로 순서대로 정렬하여 학습시키고 결과 분석

먼저 이전의 모델에서는 코드에서의 실수로 인해 정규화가 적용되지 않았던 것을 확인하였음. 정규화는 min-max normalization 방식을 사용하였고, 정규화를 적용하여 모델의 정확도가 대략 73%에서 98%로 비약적인 상승을 한 것을 확인하였음. 이 과정을 통해 정규화의 중요성을 몸소 깨달을 수 있었음. 다음 정규화의 이유를 이전 진행하였던 과정에서 느낀대로 정리를 하였으며, Input Data의 순서를 정렬하여 학습시키는 것에 대해서는 다소 어려움이 있어 추후에 다시 진행하기로 결정하였음.

결과 : 
Hyper Parameter 및 가중치 초기화 등을 조정하여 정확도 및 Loss 그래프의 노이즈를. 최소화 시켰고, Input Data 정규화를 진행하여 정확도 대략 98%에서 수렴하는 모델을 만들었음.
직접 만든 모델을 통해 결과를 확인하고 연구를 진행하는 과정을 통해 큰 성취감을 느낄 수 있었음.


3	세미나를 진행하여 학습 내용을 공유하였고, 부족한 부분에 대해 피드백을 받음
3.1	임지혜 연구생과 5차 세미나 전 자체 세미나 진행.  22.12.29

교수님과 진행할 세미나 전 임지혜 연구생과 함께 자체 세미나를 진행하였음. PPT 발표 자료 및 학습 내용에 대한 피드백을 받았고, 궁금했던 개념 및 앞으로 알아가야 할 개념들에 대한 논의하는 시간을 가졌음.
다음 임지혜 연구생에게 ResNet(Residual Neural Network) 모델에 대해 배우는 시간을 가짐.

3.2	5차 세미나 주제 : Simple Neural Network 구현.  22.12.30

교수님과 함께 직접 구현한 Simple Neural Network에 대해 세미나를 진행하였음. 모델은 앞서 설명했듯 이전에 공부한 딥러닝 지식들을 토대로 간단한 모델을 구현하였음. 문제 설명 사진 14 참고.

<img width="345" alt="image" src="https://user-images.githubusercontent.com/26295029/220339958-92b7e80e-3b48-4649-acaa-a04825c1c8f5.png">

사진 17. Neural Network 모델의 Architecture Computational Graph로 표현

<img width="337" alt="image" src="https://user-images.githubusercontent.com/26295029/220339985-7904aca1-a2ec-4a5a-aa1e-4d6f78310a95.png">

사진 18. 입력 데이터 초기화 분포

<img width="325" alt="image" src="https://user-images.githubusercontent.com/26295029/220340022-78cf4a7b-87bb-4182-affc-dacaf72b035d.png">

사진 19. 모델의 Accuracy, Loss 결과 그래프

위와 같은 문제를 해결하는 Neural Network 모델을 구현하였음. 모델에 대한 설명은 세미나 발표 자료로 대체함. 모델의 정확도(정규화 진행 전)는 대략 73%에서 수렴하는 것을 확인 하였고 주어진 추가 연구 사항들에 대한 분석과 함께 세미나를 진행함.

5차 세미나를 통해 직접 구현한 모델의 성능을 끌어올리고 정확도 및 Loss를 안정화 시키기 위한 여러 연구 및 분석을 진행하며 딥러닝이라는 분야를 공부하는 것에 큰 흥미를 느낄 수 있었음. 다음 RA활동 참여 학생(이지호)의 세미나를 듣고 CNN (Convolution Neural Network) 모델의 개념 지식을 습득하였음. 앞으로 CNN 모델도 직접 구현해 볼 계획을 갖고 있음.


3.3	6차 세미나 직접 구현한 Simple Neural Network를 사용한 추가 연구 발표.  23.1.6

5차 세미나를 통해 발표했던 Neural Network 모델을 토대로 연구 및 분석한 내용을 세미나를 통해 교수님과 공유하였음.

- 정규분포를 따르는 랜덤 수로 가중치를 초기화했을 때 정규분포의 표준편차가 결과에 미치는 영향을 학습 후 가중치 분포와 학습 전 가중치 분포의 비교를 통해 역추적하는 과정을 가졌음. Matlab의 histogram을 사용하여 가중치 분포도를 확인하였음.


<img width="323" alt="image" src="https://user-images.githubusercontent.com/26295029/220340070-c50ac660-a11d-45e4-b844-7f326dff5c3b.png">

사진 20. 가중치 초기화 방식에 따라 달리지는 결과 분석

- 정규화 적용 전의 예측 결과와 정규화 후 예측 결과를 Matlab의 Scratter3를 통해 시각화하여 비교함.

<정규화 전/후 결과 비교>
<img width="376" alt="image" src="https://user-images.githubusercontent.com/26295029/220340112-8edd983d-fc4b-40b6-9c2d-f872cdfceafb.png">

사진 21. 정규화 전 결과 분포

 <img width="355" alt="image" src="https://user-images.githubusercontent.com/26295029/220340130-0f18aaed-0da0-4844-81a8-c3740764d6ce.png">
 
사진 22. 정규화 후 결과 분포

- 다음 정규화 전 파란색의 점에 대한 예측이 부정확한 이유에 대해 분석하여 발표함.
