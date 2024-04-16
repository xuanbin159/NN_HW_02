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
- `CustomMLP` 모델은 `LeNet-5`와 유사한 파라미터 수를 가지도록 설계되었으므로 공정한 비교가 가능합니다.
- 파라미터 수: 
   - FC1: (28*28+1)*120 = 94,200
   - FC2: (120+1)*84 = 10,164
   - FC3: (84+1)*50 = 4,250
   - FC4: (50+1)*10 = 510
   - Total: 94,200 + 10,164 + 4,250 + 510 = 109,124

## 통계 플로팅
- 훈련 및 테스트 데이터셋에 대한 평균 손실값과 정확도를 그래프로 표현합니다.
- 각 모델에 대해 네 개의 그래프가 제공됩니다: 훈련 손실, 훈련 정확도, 테스트 손실, 테스트 정확도.

## 성능 비교
- `LeNet-5`와 `CustomMLP` 모델의 예측 성능을 비교합니다.
- `LeNet-5`의 정확도가 알려진 정확도와 유사한지 확인합니다.

## 정규화 기법 적용
- `LeNet-5` 모델의 성능을 향상시키기 위해 두 가지 이상의 정규화 기술을 적용
- 함수 이름은 LeNet52를 사용
- LeNet52는 기존의 LeNet5에 컨볼루션 레이어 블록에 nn.Dropout(0.2)가 적용
- 플릭커넥티드 레이어 블록에 nn.Dropout(0.5)가 적용

## 파일 설명
- `model.py`: LeNet-5, CustomMLP, LeNet52 모델 정의
- `dataset.py`: MNIST 데이터 로딩 및 전처리
- `main.py`: 모델 훈련 및 평가 실행 스크립트

## 사용 방법
- 설치 필요 라이브러리: PyTorch, Matplotlib, torchvision 등
- 실행 명령어: `python main.py`
