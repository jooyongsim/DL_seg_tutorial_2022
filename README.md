# 숙명여자대학교 Deep Learning 2022 튜토리얼
튜토리얼 2 Segmentation 실습을 위한 페이지 입니다.

### 학습가능한 모델:
- UNET
- Deeplabv3 ResNet50
- Deeplabv3 ResNet101
- Deeplabv3 MobileNet

## 과제 설명
- Semantic Segmentation Task의 대표적인 [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 데이터로 과제를 수행합니다.

## 데이터셋 설명
- 데이터셋에서 주어진 Annotation Class는 다음과 같습니다.
```
Person: person
Animal: bird, cat, cow, dog, horse, sheep
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
```
![](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21.jpg)
![](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/21_object.png)

데이터셋은 train/validation으로 나눠집니다. 
- train: 1464개
- validation: 1449개

# 과제 제출 방법
- evaluation.py 를 실행하여 나온 결과 파일을 Submission Prediction을 통해 제출합니다.
- Prediction **결과 제출 6/3 11:59PM**까지 (Leaderboard 통해 제출)
- **보고서 제출 6/6 11:59PM** 까지 (Snowboard를 통해 제출)

## 과제 평가 방법
- 리더보드의 순위 점수 25%에 점수와 보고서(또는 발표 중선택) 점수(코드 제출 포함) 75%를 합산하여 과제 점수를 부여합니다.
- 제출한 코드가 standalone으로 동작하여야 하며 다른 API를 설치하는 것은 허용하지 않습니다.
단, 함수를 일부 가지고 와서 사용하는 것은 가능합니다.
- backbone이외에 pretrained weight를 가지고 오는 경우 총 점수의 70%만 반영합니다.
- 제출한 코드의 재현이 되지 않을 경우 점수의 60%만 반영합니다.

## 과제 보고서
과제 보고서에는 다음과 같은 내용이 포함되어야합니다.
- 실험하면서 변경한 parameter 혹은 변경 내용과 해당 결과에 대한 표 (mIoU 등의 Metric)
- 실험이 여러개면 여러개 표 나누는 것 가능
- 실험한 결과의 Learning Curve (Loss Curve, Accuracy (예, mIoU))
- 왜 위와 같은 결과가 나왔는 지에 대한 고찰
- 최종 결과에 대한 최종 코드 (Kaggle과 보고서에 모두 제출)
