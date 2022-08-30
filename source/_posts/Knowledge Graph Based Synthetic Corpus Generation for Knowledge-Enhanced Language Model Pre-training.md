---
layout: post
title:  "Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training"
categories: paper
comments: true
date: 2021-11-29 12:00:00
tags: nlp
toc: true
---

## Author
- 저자:
    - Oshin Agarwal∗1 Heming Ge2 Siamak Shakeri2 Rami Al-Rfou2
    (1 University of Pennsylvania 2Google Research), 2021
![image](https://user-images.githubusercontent.com/7252598/143773523-88e7209e-723a-4a86-9ce7-3f3e279d077f.png)



## 느낀점
- 기존에 KG triples을 자연어 문장으로 바꾸는 연구가 생각보다 적었었나? 싶었음 (혹은 잘 안되었던것 같음. explicit하게 표현이 안된다던지)
-

## Abstract
- KG triples를 자연어로 바꾸는 연구들(Data-To-Text Generation)은 주로 도메인특화된 벤치마크셋 중심으로 연구되었음
- wikidata와 같은 데이터셋도 structetured KGs와 natural language corpora를 결합하는데 쓸수있음을 본 연구에서 보였음 (existing LM과 결합가능)


## Introduction
- `(subject, relation, object)`를 자연어 문장으로 바꾸는 연구 == `Data-To-Text Generation`
- WebNLG와 같은 standard dataset이 존재했음
- KG 전체를 verbalize하는 연구는 기존에 없었음 (KG를 verbalize한다는게 뭘까)
- full KG에 대해 verbalizing하는건 entitiy, relation 커버리지, triples 부족등의 단점이 있음
- 본 논문에서는 Wikidata KG 를 corpus로 바꿨고 KeLM Corpus로 칭하기로함
  - 코퍼스 스펙: ∼18M sentences spanning ∼45M triples with ∼1500 distinct relations
![image](https://user-images.githubusercontent.com/7252598/143773736-ea49d69c-f2ae-4d73-a520-27e1631e0035.png)
- 기존 연구에서는 facts가 문장에서 제대로 안드러나게 변환되던지하는 문제가 있었음
- contribution
  - data-to-text seq2seq 모델 개발
  - Text-KG aligned corpora 생성
  - KELM: 라지스케일의 Wikidata KG 를 자연어문장으로 만든 합성 코퍼스 (위에꺼를 기반으로 만들었다고 생각하면 될듯)
  - Data-totext generation이 open domain QA와 LAMA probe를 개선한다는걸 보여줌

## TEKGEN 
![image](https://user-images.githubusercontent.com/7252598/143774881-a8020915-4e84-4039-a70f-68a8bfa9bd32.png)

- TEKGEN (**Text** from **KG Gen**erator) 은 결국 wikipedia(text)와 wikidata(KG)를 align해서 input, target 데이터셋을 만들고, T5로 1차 파인튜닝 그리고 WebNLG로 2차 파인튜닝해서 BERT score가 높은 문장들을 모아서 Corpus를 만들어준다고 생각하면 될듯! (Wikidata에서 distant supervision 차이로 wikipedia랑 맵핑이 안된 triples는 나중에 다시 컨버팅해서 KELM Corpus에 넣는듯)
- Wikidata 스펙:
  - 6M entities
  - 1500 relations
- WebNLG 스펙:
  - 600 entities
  - 20 relation
- 모델링 철학
  - 약간 노이지한 큰 데이터셋을 distant supervision으로 만들자 (약간 수도레이블링 같은거죠..)
  - 순차적으로 T5 파인튜닝하자, 처음엔 노이지한 큰 데이터로하고 나중엔 작지만 클린한 데이터(WebNLG로 하자)
  - BERT를 시멘틱 필터로 써서 KG triples와의 의미적 품질을 확인해서 필터링하자
### Alignment 관련 알고리즘
![image](https://user-images.githubusercontent.com/7252598/143775068-e4bf993c-2cff-4fe3-b7df-8f4c8706252f.png)
- Align되었을때 결과 통계표
![image](https://user-images.githubusercontent.com/7252598/143775137-d6eaecc1-26a7-46b8-9e43-931c45f7145e.png)

### Types of Triples
- 위키피디아 페이지가 있는 오브젝트
- 위키피디아 페이지가 없는 오브젝트
- quantity
- date
- subproperty
![image](https://user-images.githubusercontent.com/7252598/143775225-f9b799f4-97e7-473c-a10a-bbc56c3f853d.png)

### Model
- two-step 순차적 파인튜닝을 T5-large로 함
- triples는 다 concat함 (`subject relation_1 object_1, ....relation_n object_n`)
- 5000 step까지는 엔티티커버리지 때문에 학습했지만, 예상했던 input triple이 없는 경우엔 랜덤한 값을 생성(hallucination 현상)해내기도 하기 때문에 WebNLG 2017 데이터로 500 step 파인튜닝함


### Quality Filtering
- BERT uncased model 사용
- `[CLS] concatenated-triples [SEP] reference-or-generated-sentence` 에 포멧으로 평가하고, WebNLG 2017 에 대해 1000 step 파인튜닝했음
- 시멘틱스코어는 0~1로 스케일링했고 gold reference는 스코어를 1로줌
- 2706개의 예제에 대해서 90%는 학습, 10%는 평가를 진행했고 점수와 human 평가가 높은 correlation이 나옴
![image](https://user-images.githubusercontent.com/7252598/143775353-b8b6112c-34bc-4e08-96f3-3d8e3e592ab7.png)

## Knowledge Enhanced LMs
![image](https://user-images.githubusercontent.com/7252598/143775740-b389e1d3-fbdb-4edb-888f-a276a39f8c25.png)


## Conclusion
- 본 논문에서는 다양한 시도를 통해 KG -> natural text 로 바꾸는 연구를함
- retrieval-based langauge model을 합성코퍼스인 KELM 코퍼스를 retrieval corpus로 써서 확장시킴
- 이러한 확장모델을 open domain QA와 knowledge prob에 적용했더니 둘다 개선효과가 있었음
- 데이터는 이곳에서 받아볼 수 있음: https://github.com/google-research-datasets/KELM-corpus
