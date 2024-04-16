# NN_HW_02

## 22522015 현빈   



# MNIST 숫자 분류: LeNet-5 및 사용자 정의 MLP 모델

이 프로젝트는 MNIST 데이터셋의 손글씨 숫자 분류를 위해 고전적인 LeNet-5 컨볼루션 네트워크와 사용자 정의 다층 퍼셉트론(MLP) 모델을 구현합니다.

## 구현 세부 사항

### LeNet-5
- `LeNet-5` 모델은 LeCun 등이 제안한 원래 아키텍처를 기반으로 하며, MNIST 이미지의 입력 크기에 맞게 약간의 조정이 이루어졌습니다.
- 파라미터 수: 
  - LeNet-5 Model Summary
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  
              Conv2d-1            [-1, 6, 28, 28]             156
                Tanh-2            [-1, 6, 28, 28]               0
           AvgPool2d-3            [-1, 6, 14, 14]               0
              Conv2d-4           [-1, 16, 10, 10]           2,416
                Tanh-5           [-1, 16, 10, 10]               0
           AvgPool2d-6             [-1, 16, 5, 5]               0
              Linear-7                  [-1, 120]          48,120
                Tanh-8                  [-1, 120]               0
              Linear-9                   [-1, 84]          10,164
               Tanh-10                   [-1, 84]               0
             Linear-11                   [-1, 10]             850
            Softmax-12                   [-1, 10]               0
  ----------------------------------------------------------------
    - Total params: 61,706
    - Trainable params: 61,706
    - LeNet-5 model execution time : 118.63630151748657



### 사용자 정의 MLP
- `CustomMLP` 모델은 `LeNet-5`와 유사한 파라미터 수를 가지도록 설계.
- 파라미터 수: 
    Custom Model Summary
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
                Linear-1                   [-1, 82]          64,370
                Linear-2                   [-1, 56]           4,648
                Linear-3                   [-1, 28]           1,596
                Linear-4                   [-1, 10]             290
    ----------------------------------------------------------------
    - Total params: 70,904
    - Trainable params: 70,904
    - Custom model execution time : 123.68551778793335


## 정규화 기법 적용(LeNet52)
- `LeNet-5` 모델의 성능을 향상시키기 위해 두 가지 이상의 정규화 기술을 적용
- 함수 이름은 LeNet52를 사용
- LeNet52는 기존의 LeNet5에 컨볼루션 레이어 블록에 nn.Dropout(0.2)가 적용
- 플릭커넥티드 레이어 블록에 nn.Dropout(0.5)가 적용
- 또한 LeNet52의 최적화는 SGD에서 Adam으로 변화해서 적용
## 통계 플로팅
- 훈련 및 테스트 데이터셋에 대한 평균 손실값과 정확도를 그래프로 표현합니다.
- 각 모델에 대해 네 개의 그래프가 제공됩니다: 훈련 손실, 훈련 정확도, 테스트 손실, 테스트 정확도.

- LeNet-5
  - Average training accuracy = [0.7786, 0.9246, 0.9467, 0.9600, 0.9683, 0.9743, 0.9783, 0.9813, 0.9844, 0.9864]
  - Average training loss = [0.0035, 0.0010, 0.0007, 0.0005, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002]
  - Average validation accuracy = [0.9117, 0.9356, 0.9502, 0.9588, 0.9644, 0.9668, 0.9686, 0.9711, 0.9702, 0.971]
  - Average validation loss = [0.0013, 0.0008, 0.0006, 0.0005, 0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003]
- CustomMLP
  - Average training accuracy = [0.3471, 0.5342, 0.7858, 0.8933, 0.9340, 0.9473, 0.9554, 0.9611, 0.9652, 0.9690]
  - Average training loss = [0.0089, 0.0077, 0.0067, 0.0063, 0.0060, 0.0059, 0.0059, 0.0059, 0.0058, 0.0058]
  - Average validation accuracy = [0.4087, 0.7132, 0.8394, 0.927, 0.9457, 0.9557, 0.9618, 0.9663, 0.97, 0.973]
  - Average validation loss = [0.0089, 0.0071, 0.0066, 0.0062, 0.0061, 0.0060, 0.0060, 0.0060, 0.0059, 0.0059]
- LeNet52
  - Average training accuracy: [0.8850, 0.9377, 0.9396, 0.9432, 0.9448, 0.9432, 0.9462, 0.9510, 0.9520, 0.9511]
  - Average training loss = [0.0062, 0.0060, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059]
  - Average validation accuracy = [0.9493, 0.9577, 0.9334, 0.9434, 0.955, 0.9397, 0.9586, 0.9487, 0.9612, 0.9699]
  - Average validation loss = [0.0060, 0.0060, 0.0061, 0.0060, 0.0060, 0.0061, 0.0060, 0.0060, 0.0060, 0.0059]

## 성능 비교
- `LeNet-5`, `CustomMLP`, `LeNet52` 모델의 예측 성능을 비교.
- `LeNet-5`의 정확도가 알려진 정확도와 유사한지 확인.



## 파일 설명
- `model.py`: `LeNet-5`, `CustomMLP`, `LeNet52` 모델 정의
- `dataset.py`: MNIST 데이터 로딩 및 전처리
- `main.py`: 모델 훈련 및 평가 실행 스크립트

## 사용 방법
- 설치 필요 라이브러리: PyTorch, Matplotlib, torchvision 등
- 실행 명령어: `python main.py`
