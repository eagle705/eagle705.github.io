---
layout: post
title:  "CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding"
categories: paper
comments: true
date: 2021-12-20 12:00:00
tags: nlp
toc: true
---

## Author
- 저자:
    - Dong Wang1,2∗ , Ning Ding1,2∗, Piji Li3† , Hai-Tao Zheng1,2† 
    - 1Department of Computer Science and Technology, Tsinghua University 2Tsinghua ShenZhen International Graduate School, Tsinghua University 3Tencent AI Lab
   - google scholar에서 찾긴 어려웠음

## 느낀점
- 이 논문에서는 adversarial을 같은말이라고 쓰는거 같고, constrastive를 반대말이라고 쓰는듯..
- PLM을 학습할때 두번째 pair에 아무 문장이나 넣는게 아니라 의미적으로 다른 문장을 넣겠다가 핵심임
- [https://github.com/kandorm/CLINE](https://github.com/kandorm/CLINE)

## Abstract
- PLM이 양질의 semantic representation을 만들어주지만 simple perturbations에 취약함
- PLM을 강건하게 하기위해 adversarial training에 포커스를 맞추고 있음
- 이미지 프로세싱과는 다르게 텍스트는 discrete하기 때문에 몇개의 단어 교체는 매우 큰 차이를 만들어내기도함
- 이러한 결과를 연구하기 위해 perturbation 관련 여러 파일럿 실험을 진행했음
- adversarial training이 useless하거나 오히려 모델에 안좋다는 사실을 발견함
- 이러한 문제를 해결하기 위해 Contrastive Learning withg semantIc Negative Examples (CLINE)을 제안함
- unsupervised 방식의 의미적으로 네거티브한 샘플들을 구축했고, 이를 통해 semantically adversarial attacking에 robust하도록 개선하려함
- 실험적 결과로는 sentiment analysis, reasoning, MRC 등 태스크에서 개선효과가 있었음
- 문장레벨에서 CLINE이 서로 다른 의미에 대해서 분리되고 같은 의미에 대해서는 모이는 것도 확인할 수 있었음(임베딩얘긴듯..)

## Introduction
- BERT, RoBERTa등 PLM이 NLP를 개선하는데 효과적임을 보였음
- PLM은 adversarial examples에 대해서 poort robustness를 가짐 (어쩌면..이래서 p-tuning등이 잘되는건 아닐까?)
- 테이블1에서 보면 *ultimately* 라는 단어를 비슷한 단어인 lastly로 교체하면 결과가 바뀌는걸 볼 수 있음
![image](https://user-images.githubusercontent.com/7252598/146749045-cc2901c5-62ce-458e-970e-a57ca0cdcb09.png)
- PLM의 robustness를 키우기 위해 adversarial training을 word embeddings에 gradient-based pertubation을 적용시켜서 하거나, high-quality adversarial textual examples를 추가하는 식으로 진행한 연구들이 있었음
- 하지만 작은 변화가 의미변화를 만드는걸 피할 순 없었음
- `can we train a BERT that is both defensive against adversarial attacks and sensitive to semantic changes by using both adversarial and contrastive examples?`
- robust semantic-aware PLM을 학습하기 위해서 CLINE을 제안함
  - adversarial & contrastive examples를 만드는 방법론임
  - WordNet을 사용함 (안돼 ㅠㅠ 넘 귀찮단...)
  - replaced token detection & contrastive objectives 적용
  - NLP benchmark에서 RoBERTa 모델 기준 +1.6% 개선(4 contrastive test sets), +0.5% 개선함(4 adversarial test sets) 

## Pilot Experiment and Analysis
- TextFooler (Jin et al., 2020), as the word-level adversarial attack model 을 통해 adversarial examples 만듬
- model’s true linguistic capabilities (Kaushik et al., 2020; Gardner et al., 2020)을 기반으로 contrastive sets 만듬 (MLM 같은건가..?)
### Model and Datasets
  - IMDB
  - SNLI
- 학습방법 및 모델
  - 방법: adversarial training method FreeLB (Zhu et al., 2020) for our pilot experiment.
  - 모델: vanilla BERT, RoBERTa

### Result Analysis
- Table2를 보면 방법, 데이터셋 간의 비교 결과를 확인할 수있음
- Adv에서는 성능오르지만 Rev에서 떨어지는건 역시, adversarial training이 constrative set에 negative effect를 줄 수 있다는걸 보여줌
- 아마도 adv training이 labels를 유지하려하고, contrastive set은 작은 변화에도 label이 바뀔수 있으니 그런 것 같음 (이해가 안되네....)
![image](https://user-images.githubusercontent.com/7252598/146767338-04756117-21d1-4c7e-900b-47c499764618.png)

### Case Study
- adversarial training이 contrastive sets에서 실패하는 이유에 대해 더 알아보기 위해 IMDB 데이터셋을 연구함
- Table3는 vanilla BERT에서는 잘 예측했지만, FreeLB BERT에서는 잘못 예측한 케이스임
![image](https://user-images.githubusercontent.com/7252598/146767866-a150a87d-53e3-4a32-9f38-e2f0308655b4.png)
- 대부분의 파트가 positive sentiments로 구성된걸 볼 수 있고 특정부분이 negative로 된걸 볼 수 있음, 전체적으로는 negative한 내용이 주를 이루고 vanilla BERT는 이를 잘 잡아냄, FreeLB BERT는 negative sentiment를 noise로 보고 전체문장을 positive로 예측한걸로 보임
- `adversarial training이 semantic changed adversarial examples에는 적합하지 않은것을 알 수 있음`
- 이러한 이유로 semantic negative examples로부터 semantic이 바뀌었는지를 배우는 적합한 방법을 찾을 필요가 있음

## Method
### Generation of Examples
- contrastive learning의 아이디어를 사용함
- positive pairs끼리 뭉치게하고 negative pairs는 밀어내게함
- 어떤 연구들은 augmentation 사용(synonym replacement, back translation..)해서 positive instances를 만들었지만, `negative instances들에 초점을 맞춘 연구는 거의 없었음`
- 직관적으로 문장에서 atonym(반대어)를 교체하는건 의미적으로 적합하지 않기 쉬움
![image](https://user-images.githubusercontent.com/7252598/146768890-257578b5-aed3-49d1-a7dd-f5b5c72cae40.png)
- Notation 설명: x_ori (원본), x_syn(동의어), x_ant(반의어)
- spaCy로 segmentation & POS 했고 verbs, nouns, adjectives, adversb등 추출함
- `x_syn은 synonyms로 교체한 버전이고, x_ant는 antonyms와 random words로 교체함`
- x_syn은 40% 토큰이 대체됨, x_ant는 20% 토큰이 대체됨

### Training Objectives
- neural text encoder(Transformer)를 학습시킴
- Masked Language Modeling objective
- Replaced Token Detection objective
    -  x_syn, x_ant에 대해서 어떤 토큰이 replace되었는지 디텍팅함
![image](https://user-images.githubusercontent.com/7252598/146769800-d8075ed2-30a5-4dec-a1ac-83ae9e542bbd.png)
- Contrastive Objective
  - (x_ori, x_syn) == positive, (x_ori, x_ant) == negative
  - `[CLS] embeddings`을 contrastive objective로 사용함 
![image](https://user-images.githubusercontent.com/7252598/146770241-03300e14-7a55-4d56-8322-c183c4215b0b.png)
  - 다른 contrastive strategies는 랜덤하게 multiple negative example를 뽑았지만, 본 연구에서는 x_ant만을 negative example로 사용함 (위에서 랜덤워드도 교체하긴 한다고 하지않았었나..? 뭐지..아아 아무 문장이나 두번째 페어로 쓰지 않고, 반대 문장 페어를 넣는다는 뜻인듯!)
  - 그 이유는 semantically adversarial attacking에 강건하게 만들기 위함임
- 최종 loss 구성
![image](https://user-images.githubusercontent.com/7252598/146770850-22de16e5-1147-4358-946b-f3ac1f80f764.png)

##  Experiments
### Implementation 
- RoBERTa
- 30K steps with a batch size of 256 sequences of maximum length 512 tokens
- Adam with a learning rate of 1e-4, β1 = 0.9, β2 = 0.999, ε =1e-8, L2 weight decay of 0.01, learning rate warmup over the first 500 steps, and linear decay of the learning rate
- 0.1 for dropout on all layers and in attention
- 32 NVIDIA Tesla V100 32GB GPUs
- Our model is pre-trained on a combination of BookCorpus and English Wikipedia datasets

### Datasets
- IMDB
- SNLI
- PERSPECTRUM
- BoolQ
- AG
- MR


### Experiments on Contrastive Sets
- Contrast consistency (Con) is a metric defined by Gardner et al. (2020) to evaluate whether a model’s predictions are `all correct for the same examples in both the original test set and the contrastive test set`
![image](https://user-images.githubusercontent.com/7252598/146771610-b35ad4ab-e913-4bbc-9ec2-e2aceb118033.png)

![image](https://user-images.githubusercontent.com/7252598/146771563-6f2293fe-7fbf-45dd-90d8-cdc52cabac1d.png)

### Ablation Study
- 1) w/o RTD: we remove the replaced token detection objective (LRTD) in our model to verify whether our model mainly benefits from the contrastive objective. 
- 2) w/o Hard Negative: we replace the constructed negative examples with random sampling examples to verify whether the negative examples constructed by unsupervised word substitution are better.
![image](https://user-images.githubusercontent.com/7252598/146771922-11fd1554-2d41-4778-8579-e1bd0c98b935.png)

### Sentence Semantic Representation
- 9626 문장 triplets를 MR sentiment analysis dataset에서 생성함
- the model correctly identifies the semantic relationship (e.g., if BertScore(x_ori,x_syn)>BertScore(x_ori,x_ant)) as Hits.
- max Hits on all layers (from 1 to 12) of Transformers-based encoder in Table 7
![image](https://user-images.githubusercontent.com/7252598/146773727-b933487b-6f2f-4e64-bb4c-ec013749b4e9.png)

## Conclusion
- `how to train a pre-trained language model with robustness against adversarial attacks and sensitivity to small changed semantics.`
- CLINE, a simple and effective method to tackle the challenge. In the training phase of CLINE, it automatically generates the adversarial example and semantic negative example to the original sentence
- the model is trained by three objectives to make full utilization of both sides of examples
