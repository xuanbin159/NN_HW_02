# NN_HW_02

## 22522015 현빈   



# MNIST 숫자 분류: LeNet-5 및 사용자 정의 MLP 모델

이 프로젝트는 MNIST 데이터셋의 손글씨 숫자 분류를 위해 고전적인 LeNet-5 컨볼루션 네트워크와 사용자 정의 다층 퍼셉트론(MLP) 모델을 구현합니다.

## 구현 세부 사항

### LeNet-5
- `LeNet-5` 모델은 LeCun 등이 제안한 원래 아키텍처를 기반으로 하며, MNIST 이미지의 입력 크기에 맞게 약간의 조정이 이루어졌습니다.
- 파라미터 수: 
  - C1: (5*5*1+1)*6 = 156
  - C3: (5*5*6+1)*16 = 2,416
  - C5: (16*5*5+1)*120 = 48,120
  - F6: (120+1)*84 = 10,164
  - OUTPUT: (84+1)*10 = 850
  - Total: 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706


### 사용자 정의 MLP
- `CustomMLP` 모델은 `LeNet-5`와 유사한 파라미터 수를 가지도록 설계.
- 파라미터 수: 
    Custom Model Summary
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                   [-1, 82]          64,370
                Linear-2                   [-1, 56]           4,648
                Linear-3                   [-1, 28]           1,596
                Linear-4                   [-1, 10]             290
    ================================================================
    Total params: 70,904
    Trainable params: 70,904

## 통계 플로팅
- 훈련 및 테스트 데이터셋에 대한 평균 손실값과 정확도를 그래프로 표현합니다.
- 각 모델에 대해 네 개의 그래프가 제공됩니다: 훈련 손실, 훈련 정확도, 테스트 손실, 테스트 정확도.

- LeNet-5
  - Average training accuracy = [0.2642, 0.5606, 0.7907, 0.8566, 0.9006, 0.9125, 0.9216, 0.9287, 0.9352, 0.9398]
  - Average training loss = [0.0089, 0.0079, 0.0067, 0.0064, 0.0062, 0.0061, 0.0060, 0.0060, 0.0060, 0.0060]
  - Average validation accuracy = [0.5041, 0.7138, 0.8299, 0.8984, 0.9118, 0.9196, 0.9276, 0.9337, 0.9383, 0.9442]
  - Average validation loss = [0.0089, 0.0072, 0.0066, 0.0063, 0.0062, 0.0062, 0.0061, 0.0061, 0.0061, 0.0061]
- CustomMLP
  - Average training accuracy = [0.8029, 0.9269, 0.9488, 0.9617, 0.9703, 0.9759, 0.9801, 0.9841, 0.9868, 0.9888]
  - Average training loss = [0.0029, 0.0010, 0.0007, 0.0005, 0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001]
  - Average validation accuracy = [0.9167, 0.9416, 0.9566, 0.9622, 0.9671, 0.9701, 0.9712, 0.9728, 0.9737, 0.9743]
  - Average validation loss = [0.0012, 0.0008, 0.0006, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003]


## 성능 비교
- `LeNet-5`와 `CustomMLP` 모델의 예측 성능을 비교합니다.
- `LeNet-5`의 정확도가 알려진 정확도와 유사한지 확인합니다.

## 정규화 기법 적용
- `LeNet-5` 모델의 성능을 향상시키기 위해 두 가지 이상의 정규화 기술을 적용
- 함수 이름은 LeNet52를 사용
- LeNet52는 기존의 LeNet5에 컨볼루션 레이어 블록에 nn.Dropout(0.2)가 적용
- 플릭커넥티드 레이어 블록에 nn.Dropout(0.5)가 적용
- 또한 LeNet52의 최적화는 SGD에서 Adam으로 변화해서 적용

## 파일 설명
- `model.py`: LeNet-5, CustomMLP, LeNet52 모델 정의
- `dataset.py`: MNIST 데이터 로딩 및 전처리
- `main.py`: 모델 훈련 및 평가 실행 스크립트

## 사용 방법
- 설치 필요 라이브러리: PyTorch, Matplotlib, torchvision 등
- 실행 명령어: `python main.py`
