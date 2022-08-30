---
layout: post
title:  "COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining"
categories: paper
comments: true
date: 2022-06-20 12:00:00
tags: nlp
toc: true
---

## 논문파일
- [COCO-LM- Correcting and Contrasting Text Sequences for Language Model Pretraining.pdf](https://github.com/eagle705/presentation/files/9043435/COCO-LM-.Correcting.and.Contrasting.Text.Sequences.for.Language.Model.Pretraining.pdf)

## Ref
- github: https://github.com/microsoft/COCO-LM
- 발표슬라이드: [COCO_LM_220622.pdf](https://github.com/eagle705/presentation/files/9043434/COCO_LM_220622.pdf)

## Author
- 저자: Yu Meng1∗, Chenyan Xiong2, Payal Bajaj2, Saurabh Tiwary2, Paul Bennett2, Jiawei Han1, Xia Song2
1 University of Illinois at Urbana-Champaign 2 Microsoft
  - NeurIPS 2021 논문
![image](https://user-images.githubusercontent.com/7252598/174862541-c622119c-fbbc-451e-a28e-b6d3533614d3.png)

## 요약
- ELECTRA 개선 버전 논문
  - 논문 나온 순서는 대략 ELECTRA - COCO-LM - SimCSE 가 됨
  - RTD를 copy mechanism에 녹여서 multi-task learning으로 All-token MLM 사용
- sentence similarity를 PLM self-supervised learning 안에 추가함
  - present a self-supervised learning framework, COCO-LM, that pretrains Language Models by COrrecting and COntrasting corrupted text sequences
![image](https://user-images.githubusercontent.com/7252598/174528476-33742ff9-6eef-492e-b2b9-74ee65587b2f.png)

## Related work
- 논문 순서는 대략 ELECTRA - COCO-LM - SimCSE 가 됨
- 약간 [ELECTRA 논문](https://blog.pingpong.us/electra-review/)의 연장선, 혹은 변형
  - ELECTRA의 contribution -> 15% 계산량 -> 100%로 늘려서 효율 높임
    - input token copy mechanism 실험
  - [MASK] token 대신 generator가 생성한 토큰을 써서 [MASK] 토큰이 일으키는 pre-train/fine-tune discrepancy 제거
  - Replaced Token Detection (RTD) 태스크 제안


## Abstract
- present a self-supervised learning framework, COCO-LM, that pretrains Language Models by COrrecting and COntrasting corrupted text sequences
- `Following ELECTRA-style pretraining`, COCO-LM employs an auxiliary language model to corrupt text sequences, upon which it constructs two new tasks for pretraining the main model
  - The first token-level task, Corrective Language Modeling, is to detect and correct tokens replaced by the auxiliary model, in order to better capture token-level semantics.
  - The second sequence-level task, Sequence Contrastive Learning, is to align text sequences originated from the same source input while ensuring uniformity in the representation space
  - achieves the MNLI accuracy of ELECTRA with `50% of its pretraining GPU hours`. With the same pretraining steps of standard base/large-sized models, COCO-LM `outperforms the previous best models by 1+ GLUE average points`.


## Introduction
- (무슨말이지..) ELECTRA, that uses an auxiliary language model (“generator”) to replace tokens in input texts and pretrains the main Transformer (“discriminator”) to detect replaced tokens. This improves the pretraining efficiency and effectiveness, but pretraining via binary classification hinders the model’s usage on applications requiring language modeling capability (e.g., prompt-based learning [15, 28, 46]). It could further distort the representation space as the Transformers are pretrained to output the same “non-replacement” label for all actual tokens.
- present a new self-supervised learning approach, `COCO-LM`, that pretrains Lan guage Models by `COrrecting and COntrasting corrupted text sequences`
- Following ELECTRA-style pretraining, COCO-LM employs an auxiliary model to corrupt the input texts, upon which it introduces two new pretraining tasks for the main Transformer, one at token level and one at sequence level. 
 - The token-level task, corrective language modeling (CLM), pretrains the main Transformer to detect and correct the tokens in the corrupted sequences. It uses a multi-task setup to combine the benefits of replaced token detection and language modeling. 
 - The sequence-level task, sequence contrastive learning (SCL), pretrains the model to align text sequences originated from the same source sequence and enforce uniformity of the representation space
- GLUE [54] and SQuAD [41] benchmarks, COCO-LM not only outperforms state-of-the-art pretraining approaches in effectiveness, but also significantly improves the pretraining efficiency

## Related Work
- Empirically, MLM is still among the most effective tasks to pretrain encoders
- Instead of randomly altering texts, ELECTRA [7] uses a smaller auxiliary Transformer pretrained by MLM to replace some tokens in the text sequences using its language modeling probability, and pretrains the main Transformer to detect the replaced tokens. ELECTRA achieves state-of-the-art accuracy in many language tasks [7]. Later, Clark et el. [6] developed ELECTRIC, which pretrains encoders by contrasting original tokens against negatives sampled from a cloze model. ELECTRIC re-enables the language modeling capability but underperforms ELECTRA in downstream tasks.
- Our work is also related to contrastive learning which has shown great success in visual representation learning [4, 22, 34]. Its effectiveness of in language is more observed in the fine-tuning stage, for example, in sentence representation [16], dense retrieval [60], and GLUE fine-tuning [19].

## Method
- We present the preliminaries of PLMs, their challenges, and the new COCO-LM framework.
### Preliminary on Language Model Pretraining
- In this work we focus on pretraining BERT-style bidirectional Transformer encoders
- first recap the masked language modeling (MLM) task introduced by BERT [11] and then discuss the pretraining framework of ELECTRA 
#### BERT Pretraining
![image](https://user-images.githubusercontent.com/7252598/174524483-f3da6248-8a99-4bdd-85c6-4930dbb60997.png)

#### ELECTRA Pretraining
![image](https://user-images.githubusercontent.com/7252598/174525203-78f1e8e7-6ffa-4d5d-ba80-f8fc0c611b91.png)

### Challenges of ELECTRA-Style Pretraining
- Missing Language Modeling Benefits.
  - classification task in ELECTRA is simpler and more stable [61], but raises two challenges.
    - first is the **lack of language modeling capability** which is a necessity in some tasks [6]. For example, prompt-based learning requires a language model to generate labels
    - second is that the binary classification task **may not be sufficient to capture certain word-level semantics** that are critical for token-level tasks
- Squeezing Representation Space
  - the representations from Transformer-based language models often reside in a narrow cone, where two random sentences have high similarity scores (lack of uniformity)
  - closely related sentences may have more different representations (lack of alignment)
  - Figure 1 illustrates such behaviors with random sentence pairs (from pretraining corpus) and semantically similar pairs (those annotated with maximum similarity from STS-B [3]). With RoBERTa, the cosine similarities of most random sentence pairs are near 0.8, bigger than many semantically similar pairs. The representation space from ELECTRA is even more squeezed. Nearly all sentence pairs, both random and similar ones, have around 0.9 cosine similarity. This may not be surprising as ELECTRA is pretrained to predict the same output (“non-replacement”) for all tokens in these sequences. The irregular representation space raises the risk of degeneration [37, 55] and often necessitates sophisticated post-adjustment or fine-tuning to improve the sequence representations [16, 30, 32, 60].
![image](https://user-images.githubusercontent.com/7252598/174526168-e668b74e-3d6b-49b6-9970-ceece3139ccb.png)


### COCO-LM Pretraining
- The auxiliary Transformer is pretrained by masked language modeling (MLM) and generates corrupted sequences. 
- The main Transformer is pretrained to correct the corruption (CLM) and to contrast the corrupted sequences with the cropped sequences (SCL)
![image](https://user-images.githubusercontent.com/7252598/174528476-33742ff9-6eef-492e-b2b9-74ee65587b2f.png)
![image](https://user-images.githubusercontent.com/7252598/174529064-894885ca-524a-459a-a049-b93b15fccf2a.png)
![image](https://user-images.githubusercontent.com/7252598/174529359-12e490de-4b9a-46dd-9131-9d1bbf9dd9d3.png)
![image](https://user-images.githubusercontent.com/7252598/174543773-6995f2c6-0bfd-4b2c-9957-b0a1d804bee8.png)
![image](https://user-images.githubusercontent.com/7252598/174944631-ade96c02-96ff-49c9-ba28-747fabbb7e1f.png)

#### Network Configurations
- auxiliary model
  - Similar to ELECTRA, the auxiliary Transformer is smaller than the main model
  - We reduce the number of layers to 1/3 or 1/4 (under base or large model setup, respectively) but keep its hidden dimension the same with the main model, instead of shrinking its hidden dimensions
  - We disable dropout in it when sampling replacement tokens.
- main model
  - standard architecture of BERT/ELECTRA

## Experimental Setup
- Pretraining Settings
  - base, base++, and large++. Base is the BERTBase training configuration [11]: Pretraining on Wikipedia and BookCorpus [63] (16 GB of texts) for 256 million samples on 512 token sequences
  - 32, 768 uncased BPE vocabulary
- Model Architecture
  - base/base++ model uses the BERT Base architecture [11]: 12 layer Transformer, 768 hidden size, plus **T5 relative position encoding**. 
  - large++ model is the same with BERTLarge, 24 layer and 1024 hidden size, plus T5 relative position encoding
  - auxiliary network uses the same hidden size but a shallow **4-layer Transformer** in base/base++ and a **6-layer** one in large++. When generating XMLM we disable dropout in the auxiliary model
- Downstream Tasks
  - GLUE [54] and SQuAD 2.0
  - Standard hyperparameter search in fine-tuning is performed, and the search space can be found in Appendix B.
  - **reported results are the median of five random seeds** on GLUE and SQuAD

## Evaluation Results
- COCO-LM outperforms all recent state-of-the-art pretraining models on GLUE average and SQuAD
![image](https://user-images.githubusercontent.com/7252598/174546688-70025097-e106-4803-ab3c-36cd2f5a1776.png)
![image](https://user-images.githubusercontent.com/7252598/174546704-3e4a823f-4c06-4b4f-b822-0956357d9d75.png)

### Efficiency
- COCO-LM is more efficient in GPU hours. It outperforms RoBERTa & ELECTRA by 1+ points
![image](https://user-images.githubusercontent.com/7252598/174547162-477cf8aa-626b-461c-968f-f94393212643.png)

### Ablation Studies
- base setting on GLUE DEV
- 예상과는 좀 다른..
![image](https://user-images.githubusercontent.com/7252598/174549284-2af5c4d4-8171-426e-af66-9949de2f4505.png)

#### Architecture. 
- Removing relative position encoding (Rel-Pos) leads to better numbers on some tasks but significantly hurts MNLI.
#### Pretraining Signal Construction.
- Using randomly replaced tokens to corrupt text sequence hurts significantly. Using a converged auxiliary network to pretrain the main model also hurts. It is better to pretrain the two Transformers together

#### CLM Setup. 
- Disabling the multi-task learning and using All-Token MLM [7] reduces model accuracy.
- The copy mechanism is effective. The benefits of the stop gradient operation are more on stability (preventing training divergence).

### Analyses of Contrastive Learning with SCL
#### Ablation on Data Augmentation
![image](https://user-images.githubusercontent.com/7252598/174555297-094088d9-5de3-48db-b6b4-d04d8c052889.png)

#### Alignment and Uniformity
- The representation space from COCO-LM is drastically different from those in Figure 1
- With COCO-LM, similar pairs are more aligned and random pairs are distributed more uniformly
- Their average cosine similarity is 0.925 when pretrained with SCL, while is 0.863 without SCL. This better alignment and uniformity is achieved by COCO-LM with SCL via pretraining
#### Regularizing the Representation Learning for Better Few-Shot Ability.
- SCL is necessary to regularize the representation space and to reduce the risk of degeneration
![image](https://user-images.githubusercontent.com/7252598/174947724-8442f682-af96-4c76-84ac-001cfa92413c.png)

![image](https://user-images.githubusercontent.com/7252598/174556264-c75922d7-8196-4f05-97eb-3483e2b314e5.png)


### Analyses of Language Modeling with CLM
![image](https://user-images.githubusercontent.com/7252598/174615775-2632203a-4f23-4178-872d-d82e7a618274.png)
- CLM과 All-Token MLM 비교
- It is quite an unbalanced task
  - For the majority of the tokens (Original) the task is simply to copy its input at the same position.
  - For the replaced tokens (7 − 8% total), however, the model needs to detect the abnormality brought by the auxiliary model and recover the original token
  - Implicitly training the copy mechanism as part of the hard LM task is not effective: The copy accuracy of All-Token MLM is much lower, and thus the LM head may confuse original tokens with replaced ones
    - As shown in Table 3 and ELECTRA [7], pretraining with All-Token MLM performs worse than using the RTD task, though the latter is equivalent to only training the copy mechanism
    - The multi-task learning of CLM is necessary for the main Transformer to stably learn the language modeling task upon the corrupted text sequence.

### Prompt-Based Fine-Tuning with CLM
- the prompt-based fine-tuning experiments on MNLI for RoBERTa and COCO-LM under base++ and large++ sizes
- COCO-LM’s main Transformer does not even see any
[MASK] tokens during pretraining but still performs well on predicting masked tokens for prompt-based learning.
- Note that ELECTRA and COCO-LM variants without the CLM task are not applicable: Their main Transformers are not pretrained by language modeling tasks (thus no language modeling capability is learned to generate prompt label words).
![image](https://user-images.githubusercontent.com/7252598/174616974-fa24a5ee-7123-4b4a-a005-12943b75ffa4.png)

## Conclusion and Future Work
- we present COCO-LM, which pretrains language models using Corrective Language Modeling and Sequence Contrastive Learning upon corrupted text sequences
- With standard pre- training data and Transformer architectures, COCO-LM improves the accuracy on the GLUE and SQuAD benchmarks, while also being more efficient in utilizing pretraining computing resources and network parameters
- **One limitation of this work is that the contrastive pairs are constructed by simple cropping and MLM replacements**
- To better understand and tailor the training of the auxiliary model to the main model is another important future research direction


## 코드구현
- loss 관련 코드 스니펫: https://github.com/microsoft/COCO-LM/issues/2#issuecomment-1003639940
- scl쪽 (span으로 한번 임베딩뽑고, src로도 한번 뽑고)
![image](https://user-images.githubusercontent.com/7252598/174631039-71b4977c-02ef-498c-b577-cfd973ef0fc5.png)
- 위 코드 위치: https://github.com/microsoft/COCO-LM/blob/6bb6e5f62d65349657dd51f2f535454a1c50c2e9/fairseq/fairseq/models/cocolm/model.py#L190
- unofficial implementation: https://github.com/lucidrains/coco-lm-pytorch/blob/main/coco_lm_pytorch/coco_lm_pytorch.py
- 
