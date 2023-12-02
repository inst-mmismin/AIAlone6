# 필요한 패키지를 import 

# hyper-parameter 선언 
# 선언한 hparam을 저장 

# 데이터 불러오기 
# dataset 만들기 & 전처리하는 코드도 같이 작성 
# dataloader 만들기 

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기 

# AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
# Loss 객체 만들고 
# Optimizer 객체도 만들고 

# -------- 준비단계 -------
# -------- 학습단계 -------

# for loop를 기반으로 학습이 시작됨 
    # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야 함 
    # 이 과정이 for loop로 진행 
        # dataloader가 넘겨주는 데이터를 받아서 

        # AI 모델에게 넘겨주고 
        # 출력물을 기반으로 Loss를 구하고 
        # Loss를 바탕으로 Optimize를 진행  

        # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인 
            # 평가를 진행 
            # 보고 싶은 수치 확인 (Loss, 평가 결과 값, 이미지와 같은 meta-data)
            # 만약 평가 결과가 나쁘지 않으면 
                # 모델을 저장 

