---
layout: post
title:  "WARP: Word-level Adversarial ReProgramming"
categories: paper
comments: true
date: 2021-12-06 12:00:00
tags: nlp
toc: true
---

## Author
- 저자:
    - Karen Hambardzumyan1, Hrant Khachatrian1,2, Jonathan May3    (1YerevaNN, 2Yerevan State University,
3Information Sciences Institute, University of Southern California), 2021
![image](https://user-images.githubusercontent.com/7252598/144816134-bd319295-1bc4-4ef1-82c1-e76a67b18b6f.png)




## 느낀점
- PET + p-tuning

## Abstract
- 대부분의 transfer learning은 params sharing을 최대화해서, 하나 혹은 여러 task-specific layers를 LM 위에 쌓아서 학습하는 형태임
- 본 논문에서는 다른 형태로 automatic propmpt generation이라는 선행연구 기반의 adversarial reprogramming 방법을 사용함
- Adversarial reprogramming에서는 task-specific word embeddings 을 학습하는데, 이는 특정 input text가 합쳐져서 입력으로 들어올때 LM이 specified task를 해결하게 하는 것임 (이래서 propmpt연구의 확장이라 했나..)
- 25K trainable params로 25M trainable params 모델까지 outperform했음 (GLUE benchmark 기준)
- task-specific human-readable prompts로 few-shot setting(32 training samples)에서 2개의 SuperGLUE task에서 GPT-3보다 좋은 성능을 냄


## Introduction
- 요즘 pretrained model을 쓰는 대안은 adapters라고도 불리는데, 모든 레이어에 new weights를 더하는 방식으로 진행됨(ptrLM params은 frozen)
  - 이러한 방법은 smaller set of task-specific params로 fine-tuning과 비슷한 성능을 냄
- 또 다른 연구는 "task descriptions"를 제공하는 방법론임 (labeled examples 없이)
  - GPT-3의 경우가 이에 해당
  - 이러한 방법론은 대신 huge LM (1.5B~175B) 가 필요함
- reformulation-based approach (prompt)에서 성능을 좋게 만드는 extra tokens을 찾을 수 있으면 손으로 직접 디자인한 것보다 좋은 성능 낼 수 있을 것
- optimal prompts르 찾는 테크닉 제안(WARP: Word-level Adversarial ReProgramming)
- 이 방법론은 이미지쪽 adaversarial program을 보고 아이디어를 얻음 (이름부터가 이미..)
![image](https://user-images.githubusercontent.com/7252598/144822008-20fd00c2-2f1b-4386-903d-12896eb653db.png)
- 여러 결과에서 좋은 성적 얻음
  - GLUE leaderboard에서 81.6 test score를 얻었음(25K trainable params)
  - 32 examples few-shot -> SuperGLUDE에서 GPT-3를 이기기도함(2개 태스크)

## Related Work
- Towards Fewer Trainable Parameters
  - 레이어마다 파라미터 추가하거나.. knowledge distillation하거나 등등
- Task Reformulation 
  - GPT 계열처럼 prompt 넣기
  - MLM 처럼 빈칸채우기 (PET)
- Adversarial Reprogramming
  - input값을 바꿔줘서 (perturbations) 학습시키는 것
  - text classification 쪽에도 연구가 있긴 했었음
  - AutoPrompt와 다르게, 본 연구에서는 word embedding space에 대해 gradient-based optimization을 수행함

## WARP
![image](https://user-images.githubusercontent.com/7252598/144824403-58fc9712-84a5-4b41-80fe-27982332cb17.png)
- Goal: MLM이 원하는 verbalizer token을 answer로 뱉어낼 수 있는 최고의 prompt (continuous embeddng)를 찾는 것
- 다른말로하면, <img src="https://render.githubusercontent.com/render/math?math=\Theta=\left\{\Theta^{P}, \Theta^{V}\right\}"> prompt에 대한 파라미터와 verbalizer embeddings 에대한 parameter를 찾고 있음
  - <img src="https://render.githubusercontent.com/render/math?math=\Theta^{*}=\arg \max _{\Theta}\left(-\log P_{\Theta}(y \mid x)\right)">
  -  확률은 다음과 같이 나타냄 <img src="https://render.githubusercontent.com/render/math?math=P_{\Theta}(y \mid x)=\frac{\exp \Theta_{y}^{V} f\left(T_{\Theta^{P}}(x)\right)}{\sum_{i \in C} \exp \Theta_{i}^{V} f\left(T_{\Theta^{P}}(x)\right)}">
  - T는 프롬프트 임베딩이 들어가는 템플릿을 뜻함
  - C는 클래스 집합
  - f(x)는 MLM output이고
  - theta P, theta V는 워드 임베딩임과 같은 임베딩 스페이스에 있는 벡터임
  - P쪽이 prompt, V쪽이 class 라고 보면 될듯
 
### Method
![image](https://user-images.githubusercontent.com/7252598/144829136-0f9dd456-b232-418d-808a-65b4f1f36dcc.png)
- prompt tokens `[P_1], [P_2], ..., [P_K]` 와 Maksed Token [MASK]를 input sequence에 추가함
- prompt template에 따라 프롬프트 토큰은 문장 앞뒤중간에 존재함 (이게 좀 애매하다.....영~)
- Xentory로 MLM의 output head와 verbalizer tokens `[V_1], [V_2], ..., [V_C]` 간의 loss를 optimization함 (약간 PET + p tuning인데..)
- 나머지 LM params은 건드리지 않음
- adversarial attack과는 다르게 original input tokens을 바꾸거나 하진 않음


### Implementation Details
- GLUE task
  - roberta-large 
  - pytorch
- few-shot task
  - albert-xxlarge-v2 (iPET과 비교 위해)
- Optim
  - Adam
  - slanted triangular scheduler (6% warm-up steps & 10-20 epochs on each task)
  - batch
    - 1024 tokens & 8 examples
  - speed up
    - ptrLM의 dropout 제거
    - 2.5-3배정도 fine-tuning보다 빠르고, frozen features 보다는 2배정도 느림

## Experiments on GLUE
![image](https://user-images.githubusercontent.com/7252598/144830801-8755b24c-8f92-44c6-a3fc-919efca3c21a.png)
![image](https://user-images.githubusercontent.com/7252598/144830812-4af45f90-697d-460a-b6b0-53591712729b.png)

## Few-Shot Experiments
![image](https://user-images.githubusercontent.com/7252598/144831066-698cfdba-56c3-4c99-8ec7-7871b9d15eca.png)

## Discussion
- prompts보단 verbalizers에서 좀 더 해석가능한 결과가 나왔음
- 해당 임베딩과 cosine sim으로 가장 가까운 토큰이 무엇인지 보여줌 (토큰벡터는 레이어중 어디꺼를 빼다 쓴건지.., 그냥 워드임베딩 레이어인가)
![image](https://user-images.githubusercontent.com/7252598/144832050-2b20f517-a98e-4d3d-bc06-48fdecd8fcd4.png)



## Conclusion
- optimized embedding을 input text에 추가하는 방법론으로 transfer learning의 다른 대안을 제안해봄 
- GLUE나 SuperGLUE에서 좋은 성능을 보여줌
