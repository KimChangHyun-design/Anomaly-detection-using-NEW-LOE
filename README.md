# 설명

![image](https://github.com/KimChangHyun-design/Anomaly-detection-using-NEW-LOE/assets/127087508/1fdb40f3-bdf7-4273-be7e-ea2a5c6acb36)

- 크게 두 block으로 구성, neural transformation(data변환) / encoder(특징 추출)
- 기존 transformation은 형식이 있었지만( ex) 미분 or 정규화) 여기서는 transformation 자체를 학습

# 학습 방식
![image](https://github.com/KimChangHyun-design/Anomaly-detection-using-NEW-LOE/assets/127087508/d1a714e1-54d1-4719-b190-568e12ccddc9)

![image](https://github.com/KimChangHyun-design/Anomaly-detection-using-NEW-LOE/assets/127087508/a6d29e34-305c-4403-8d68-511c148efb38)

1. 모델이 스스로 Strain score를 계산하여 label 판정(정상 or 비정상)
'정상이라고 판단'했을 때 => yi=0
-> A VS A' 비슷한 정도 크게
-> A' VS A'' 비슷한 정도 작게
(A는 원래 DATA , A' A''는 transform 된 DATA)

'비정상이라고 판단' 했을 때 => yi=1
-> 정상이라고 판단 했을 때 와 반대로.

2. label(yi)를 가지고 전체 loss 줄어드는 방향으로 파라미터 업데이트

# 기존 논문과 다른 점
몇 가지 알고 있는 "비정상" 데이터에 대한 loss를 기존 DCL loss 에 추가
-> 파라미터 성능 향상

# Base Thesis
- Neural Transformation Learning for Deep Anomaly Detection Beyond Images(Chen Qiu et al. 2020)
https://arxiv.org/abs/2103.16440
- Latent Outlier Exposure for Anomaly Detection with Contaminated Data(Chen Qiu et al. 2022)
https://arxiv.org/abs/2202.08088

# 코드 실행 방법
LOE_semi_0724_v1\LOE_semi_v1\LOE_semi\requirements.txt , README.md 참고
