---
layout: post
title:  "CV & NLP 정리"
subtitle:   "Computer Vision"
categories: cslog
tags: cslog
comments: true
use_math: true
---

배경
- (x, y, r, g, b)
- (x, y, ...) HSV

## OpenCV
- BGR (RGB 순서를 바꿔놓은 것)
- 

### 주요 라이브러리
- 이진화: color -> gray scale; (R,G,B) -> (BW)
  - 차원 축소
    - RGB 평균
    - 기타 등등
  - threshold: black & white 바꾸기
    - global thresholding: 잘안씀
    - Adaptive:
      - Adaptive Mean thresholding
      - Adaptive Gaussian thresholding
      - kernel size: nxn (홀수로)
- ROI: (사각형, 원)
- 형태학 연산
- 외곽선 검출
- 노이즈 제거

## Morphology
- D(x), Dilation(팽창) : Smoothing
- E(x), Erosion(침식) : Sharpening
- D(x) - E(x)
- D(E(x))
- E(D(x))
- 보고 싶은 영역을 강조할때나 깨끗하게 처리할때 사용

## contour
- 영역 따기

## Text Detection
- 좌표가 캐릭터 중심일 확률
- 좌표가 캐리거 사이의 중심일 확률
- 좌표레벨의 라벨링인듯?!

