
    # #학습 시작
    # cnt=0
    # # 순전파 시작
    # for j in range(iteration_num):
        
    #     batch_mask=np.random.choice(xyz_train_set.shape[0],batch_size)  #랜덤으로 trian_set에서 뽑아옴
    #     xyz_batch=xyz_train_set[batch_mask]
    #     t_batch=t_train_set[batch_mask]
        
    #     input=xyz_batch
    #     now_t=t_batch
    #     for i in range(5):
    #         input=Layer[i].forward(input)
        
    #     L=LastLayer.forward(input,now_t)
    #     # print(L)
    #     # 여기까지가 순전파
        
    #     #역전파
    #     dout=1 
    #     dout=LastLayer.backward(dout)
    #     for i in reversed(range(5)):
    #         dout=Layer[i].backward(dout)
    #     cnt+=1
    #     #역전파 끝
    # #학습 끝
        
    # # 처음 50번 돌려서 totalLoss계산
    # cnt=0
    # totalLoss=0
    # print(totalLoss)
    # for j in xyz_test_set:
    #     input=j
    #     now_t=t[cnt]
    #     for i in range(5):
    #         input=Layer[i].forward(input)
        
    #     L=LastLayer.forward(input,now_t)
    #     totalLoss+=L
    # print(totalLoss)
    # # 테스트 끝
    