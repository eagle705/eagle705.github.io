---
layout: post
title:  "GPT Understands, Too "
categories: paper
comments: true
date: 2021-12-13 12:00:00
tags: nlp
toc: true
---

## Author
- 저자:
    - Xiao Liu* 1 2 Yanan Zheng* 1 2 Zhengxiao Du1 2
![image](https://user-images.githubusercontent.com/7252598/145810635-fed565cc-3a42-4a88-b65c-87ffad8199af.png)

## 느낀점
- neural model은.. 작은 변화에 너무 민감하다?!

## Abstract
- GPTs 계열에서 기존의 fine-tuning 방법이 NLU task에서 좋은 결과를 내기 어려웠음
- 새로운 방법인 p-tuning이라는 방법을 제안해서  BERT와 비슷한 크기의 모델에서는 좋은 결과를 내게함
- knowledge probing (LAMA) benchmark에서 64%(P@1)를 기록했음, SuperGlue에선는 BERT의 지도학습보다 좋은 결과를 냄
- p-tuning이 BERT 성능도 좋게함을 발견함(few-sho & 지도학습 셋팅에서)
- p-tuning은 few-shot SuperGlue에서 SOTA임

## Introduction
- ptrLM은 3가지로 나뉨, unidirectional LM, bidirectional LM, hybrid LM
- GPT-style LM은 파인튜닝할때 NLU에서 좋은 결과를 못냈었음
- GPT-3가 나오면서 hand-crafted prompts기반의 few-shot, zero-shot learning이 부각됨 
- prompt에 따라 NLU 좋아질수있음을 보임
- 하지만 handcraft로 best prompt 찾기는 건초더미에서 바늘찾기임
- neuralnet은 태생적으로 continuous한 성격을 갖기에, discrete prompts는 sub-optimal 정도를 갖게된다는 한계가 있음
- 본 연구에서는 P-tuning이라는 새로운 방법론을 제안해서 GPTs와 NLU의 갭을 메울것임
- p-tuning은 continuous free parameters를 prompts로 ptrLM에 입력하는 방식임
- continuous prompts를 gradient descent 방식으로 업데이트 해서 최적화함
- 본 연구에서는 다음과 같은 점을 밝혀냄 (contribution)
  - P-tuning사용시 GPTs 계열이 BERTs 계열에 대비 NLU 태스크에서 경쟁력이 있음
  - P-tuning은 GPTs, BERTs 계열 모두 few-shot, fully supervised setting에서 사용 가능함
![image](https://user-images.githubusercontent.com/7252598/145814238-6ea1674c-8725-4bc0-8c41-22ca396af58e.png)


## Motivation
- GPT-3, DALLE는 giant model이 machine intelligence를 촉진시킬 것을 보여줬지만 단점이 있었음
- transfer ability가 떨어짐, trillion-scale model을 파인튜닝하는 건 어려움
- handcrafted prompts 찾는 일도 버거운 방법이고, 이를 위해 validation set에 의존하는것도 비현실적임 뿐만아니라, prompt에 따라 성능도 너무 확확바뀜
![image](https://user-images.githubusercontent.com/7252598/145815114-97d0a101-0483-4222-bfc4-818f4f68bce1.png)

## Method: P-tuning
- input 수정하거나 자체를 바꾸진 않음
- 대신에 PLM의 input embedding의 일부를 differential output embedding으로 교체함

### Architecture
- pseudo token을 template에 맵핑해서 넣음 
![image](https://user-images.githubusercontent.com/7252598/145815905-e3e713e7-4674-4777-8084-c055a7688f80.png)

### Optimization
- prompt encoder를 통과시켜서 적용했고, LSTM & two-layer MLP(ReLU) 사용함

## Experiments
### Knowlege Probing
- transform the triple (Dante, born-in, Florence) into a cloze sentence with the handcraft prompt “Dante was born in [MASK].”, and then we ask language models to inference the target
![image](https://user-images.githubusercontent.com/7252598/145819472-290429e3-3fcf-4576-a140-7ca1f141028d.png)

### SuperGLUE
- 8개의 NLU task
![image](https://user-images.githubusercontent.com/7252598/145820072-02d1e870-6315-4f96-8fe0-cfcffcb15398.png)
![image](https://user-images.githubusercontent.com/7252598/145820417-c310447c-efd1-4ca9-b123-a4c496e58489.png)


## Conclusion
- best prompt를 continuous space에서 자동으로 찾아주는 P-tuning 방법을 제안함
- 기존대비 large validdation set에 의존하지 않음 (덜 의존함), adversial prompts에 덜 취약함
- knowledge probing (LAMA) benchmark에서 64%(P@1)를 기록
- GPT-style 모델이 BERTs 대비 NLU 잘할 수 있게 함
- P-tuning은 bidirectional model에도 도움을 줌
- few-shot SuperGlue에서 SOTA 뛰어넘음
