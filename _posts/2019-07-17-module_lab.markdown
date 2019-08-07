---
layout: post
title:  "모두의 연구소 DeepNLP Lab"
subtitle:   "Logs"
categories: paper
tags: deeplearning
comments: true
use_math: true
---

## 19.7.24
### MT-DNN
- ref: https://arxiv.org/pdf/1901.11504.pdf
Ranking을 배우게하는 네트워크를 학습시킬때,

P = exp(정답에 대한 스코어) / Sum of 모든 candidate의 exp(스코어) 라고 할 때,
loss = - P 로 해놓고 loss 를 줄이면, 정답에 대한 P가 올라감!

### XL-Net

단어 예측할때 단어 순서를 Perm 해서 모든 단어 순서를 다보겠다는 뜻...흠..



## 19.7.17

### 공부할 것
- einsum 
- ex) Torch, tf, numpy einsum
- ref: https://ita9naiwa.github.io/numeric%20calculation/2018/11/10/Einsum.html
- tip: Index, batch, seq, dim

### 정보이론
- 강의
  - 스탠포드: https://stanford.edu/class/stats311/?fbclid=IwAR3FCY3j0N3uQb4bQE3L__g_LlPfRmQ7j-v1Sg3T1FOZA79onC2T3l7Iaeg
- 책
  - http://staff.ustc.edu.cn/~cgong821/Wiley.Interscience.Elements.of.Information.Theory.Jul.2006.eBook-DDU.pdf?fbclid=IwAR37ArinclIuVqtPp4u366nQPWIPj3OGw5opB7asgAu56EV4_Kw3y8h8AKU  (앞쪽에 두세 장만 읽어보시는 것)
  - Bishop의 Pattern Recognition and Machine Learning의 Chapter 1의 Information theory section


Leetcode 코드인터뷰
https://www.milemoa.com/bbs/board/5085929

https://gmlwjd9405.github.io/2018/05/14/how-to-study-algorithms.html

#### Soynlp
ref: https://lovit.github.io/nlp/2018/04/09/branching_entropy_accessor_variety/

단어 추출 (LM 의 확률이 떨어질때 단어로보자 -> word boundary)

한국어는 cohesion score 사용 (character n-gram based on count)
한국어 특성상 형태소로 쪼개지면
L+R 구조중 R쪽에 의미가 없는 친구들이 더 많이 나올 확률이 높음
LM에서 확률이 가장 많이 떨어지는 구간이 토큰화되는 구간일 수 있음

L-R Graph도 결국 r_score 계산을 위해 학습 데이터가 필요하니 요즘의 NN기반이 좀 더 좋을수도있다?!

conjugator
- 어간과 어미의 원형이 주어졌을때 적절한 모양으로 용언을 바꿔주는것?!
  - 규칙 활용(어근과 어미의 합)
  - 불규칙 활용 - 14개

conjugator <-> lemmatizer

lemmatizer
- 원형 복원

stemming
- 그냥 짜르는 것
