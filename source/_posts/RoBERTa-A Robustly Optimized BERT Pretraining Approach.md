---
layout: post
title:  "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
categories: paper
comments: true
date: 2022-05-19 12:00:00
tags: nlp
toc: true
---

## Author
- 저자: Yinhan Liu∗§ Myle Ott∗§ Naman Goyal∗§ Jingfei Du∗§ Mandar Joshi† Danqi Chen§ Omer Levy§ Mike Lewis§ Luke Zettlemoyer†§ Veselin Stoyanov§ 
  - † Paul G. Allen School of Computer Science & Engineering, University of Washington, Seattle, WA 
  - § Facebook AI


## 느낀점


## Abstract
- hyperparameter choices have significant impact on the final results
- carefully measures the impact of many key hyperparameters and training data size
- find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it

## Introduction
- We present a replication study of BERT pretraining (Devlin et al., 2019), which includes a careful evaluation of the effects of hyperparmeter tuning and training set size.
- modifications
  - (1) training the model **longer, with bigger batches, over more data**; 
  - (2) **removing the next sentence prediction** objective; 
  - (3) training on **longer sequences**; and 
  - (4) **dynamically changing the masking pattern** applied to the training data.
- contributions
  - (1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance; 
  - (2) We use a novel dataset, CC-NEWS, and confirm that using more data for pretraining further improves performance on downstream tasks; 
  - (3) Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods.

## Background
### Training Objectives
- MLM objective is a cross-entropy loss on predicting the masked tokens
- BERT uniformly selects 15% of the input tokens for possible replacement. Of the selected tokens, 80% are replaced with [MASK ], 10% are left unchanged and 10% are replaced by a randomly selected vocabulary token
- In the original implementation, random masking and replacement is performed once in the be- ginning and saved for the duration of training, although in practice, data is duplicated so the mask is not always the same for every training sentence (기존과 달리 dynamic masking 하겠습니다~)
- NSP (로버타는 쓰지 않지만)
### Optimization
- BERT is optimized with Adam (Kingma and Ba, 2015) using the following parameters: β1 = 0.9, β2 = 0.999, ǫ = 1e-6 and L2 weight decay of 0.01. 
- The learning rate is warmed up over the first 10,000 steps to a peak value of 1e-4, and then linearly decayed
- BERT trains with a dropout of 0.1 on all layers and attention weights, and a GELU activation function
- Models are pretrained for S = 1,000,000 updates, with mini-batches containing B = 256 sequences of maxi- mum length T = 512 tokens.
### Data
-  trained on a combination of BOOKCORPUS (Zhu et al., 2015) plus English WIKIPEDIA, which totals 16GB of uncompressed text

## Experimental Setup
### Implementation
- optimization hyperparameters, given in Section 2, except for the peak learning rate and number of warmup steps, which are tuned separately for each setting.
- additionally found training to be very sensitive to the Adam epsilon term, and in some cases we obtained better performance or improved stability after tuning it
- we found setting **β2 = 0.98 to improve stability** when training **with large batch sizes**.
- Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we **do not train with a reduced sequence length for the first 90% of updates**. We train only with full-length sequences.
- train with **mixed precision floating point** arithmetic on DGX-1 machines, each with 8 × 32GB Nvidia V100 GPUs


### Data
- five English-language corpora of varying sizes and domains, totaling over **160GB** of uncompressed text
  - **BOOKCORPUS** (Zhu et al., 2015) plus English WIKIPEDIA. This is the original data used to train BERT. (16GB)
  - **CC-NEWS**, which we collected from the English portion of the CommonCrawl News dataset (Nagel, 2016). The data contains 63 million English news articles crawled between September 2016 and February 2019. (76GB after filtering)
  - **OPENWEBTEXT** (Gokaslan and Cohen, 2019), an open-source recreation of the WebText corpus described in Radford et al. (2019). The text is **web content extracted from URLs shared on Reddit with at least three upvotes**. (38GB).
  - **STORIES**, a dataset introduced in Trinh and Le (2018) containing a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas. (31GB)

### Evaluation
- GLUE: is a collection of 9 datasets for evaluating natural language understanding systems
- SQuAD: is to answer the question by extracting the relevant span from the context.
  - evaluate on two versions of SQuAD: V1.1 and V2.0
  - In V1.1 the context always contains an answer, whereas **in V2.0 some questions are not answered in the provided context,** making the task more challenging.
  - **For SQuAD V2.0, we add an additional binary classifier to predict whether the question is answerable**, which we train jointly by summing the classification and span loss terms
- RACE: is a **large-scale reading comprehension** dataset with more than 28,000 passages and nearly 100,000 questions
  - **is collected from English examinations in China**, which are designed for middle and high school students.
  - each passage is associated with multiple questions. For every question, the task is to select one correct answer from four options
  - RACE has significantly **longer context** than other popular reading comprehension datasets and the proportion of questions that requires reasoning is very large

## Training Procedure Analysis
- Model Architecture: BERT_BASE (L = 12, H = 768, A = 12, 110M params)
### Static vs. Dynamic Masking
- To avoid using the same mask for each training instance in every epoch, **training data was duplicated 10 times so that each sequence** is masked in 10 different ways over the **40 epochs** of training
- Thus, **each training sequence was seen with the same mask four times during training**
  - 1문장을 10개씩 복사해서 다른 mask를 생성하게 만들게 하고 40에폭 돌림 -> 10*4 -> 완전히 똑같은 마스킹이된 문장은 4번만 보게됨!
  - 약간 에폭 계산 방법이 이상한데 이게 맞긴맞나봄
- becomes **crucial when pretraining for more steps** or with larger datasets
  - 같은 문장을 여러 방법으로 쓸 수 있다는 점에서 dynamic masking에 epoch이 많은게 중요할듯
![image](https://user-images.githubusercontent.com/7252598/169627448-c6b7d8a0-3860-44a3-b3ae-f2de1ce1143b.png)
- dynamic masking is comparable or slightly better than static masking
  - 살짝만 좋아지긴하지만 5번 반복에 대한 meidan 값이니 유의미한 셋팅으로 봐야

### Model Input Format and Next Sentence Prediction
![image](https://user-images.githubusercontent.com/7252598/169627719-c894030b-17d2-4988-ba76-ca1e40be9c0f.png)
- SEGMENT-PAIR+NSP: This follows the original input format used in BERT (Devlin et al., 2019), with the NSP loss. Each input has a pair of **segments, which can each contain multiple natural sentences**. 
  - 여러 문장들
- SENTENCE-PAIR+NSP: Each input contains a pair of natural sentences, either sampled from a contiguous portion of one document or from separate documents. Since these **inputs are significantly shorter** than 512 tokens.
  - 딱 한문장
- FULL-SENTENCES: Each input is packed with **full sentences sampled contiguously from one or more documents**. Inputs may cross document boundaries. When we reach the end of one document, we begin sampling sentences from the next document and add an extra separator token between documents
  - 여러 문장들 + 문서 끊기면 SEP 추가후 다음문서에서 여러문장들
- DOC-SENTENCES: Inputs are constructed similarly to FULL-SENTENCES, except that they may not cross document boundaries. Inputs sampled near the end of a document may be shorter than 512 tokens, so we dynamically increase the batch size in these cases to achieve a similar number of total tokens as FULL- SENTENCES
  - 문서 끝에서 샘플링하면 토큰수가 적을테니 이런 경우는 dynamic하게 배치사이즈 키워서 FULL-SENTENCES와 비슷한 토큰개수 만들려고함
  - `DOC-SENTENCES가 결과상으로는 제일 좋네.. SEP 없이 한 문서내에서 샘플링하는게 좋다는걸까?`

##### Results
- **using individual sentences hurts performance** on downstream tasks
  - hypothesize is because the model is not able to learn long-range dependencies
-  removing the NSP loss matches or slightly improves downstream task performance
- restricting sequences to **come from a single document (DOC-SENTENCES) performs slightly better than packing sequences from multiple documents (FULL-SENTENCES).**
- However, because the **DOC-SENTENCES format results in variable batch sizes, we use FULL-SENTENCES in the remainder of our experiments for easier comparison** with related work

#### Training with large batches
- Devlin et al. (2019) originally trained BERTBASE for 1M steps with a batch size of 256 sequences. This is equivalent in computational cost, via gradient accumulation, to training for 125K steps with a batch size of 2K sequences, or for 31K steps with a batch size of 8K.
  - 첫번째는 gradient accumulation 8, 두번째는 32
![image](https://user-images.githubusercontent.com/7252598/169628883-4b353398-f7e7-4573-b820-51336059fed1.png)
- observe that training with large batches improves perplexity for the masked language modeling objective, as well as end-task accuracy. 
- Large batches are also easier to parallelize via distributed data parallel training, and in later experiments we **train with batches of 8K sequences**.

#### Text Encoding
- Byte-Pair Encoding (BPE)
- BPE vocabulary sizes typically range from 10K-100K subword units. However, unicode characters can account for a sizeable portion of this vocabulary when modeling large
- Radford et al. (2019) introduce a clever implementation of **BPE that uses bytes instead of unicode characters as the base subword units.** Using bytes makes it possible to learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any “unknown” tokens.
  - The original BERT implementation (Devlin et al., 2019) uses a **character-level** BPE vocabulary of **size 30K**, which is learned after preprocessing the input with heuristic tokenization rules.
  - Following Radford et al. (2019), we instead consider training BERT with a larger **byte-level** BPE vocabulary containing **50K subword units**, **without any additional preprocessing or tokenization of the input.** This adds approximately 15M and 20M additional parameters for BERTBASE and BERTLARGE, respectively
    - RoBERTa에서 byte-level BPE를 사용했었군.. 약간 골치아프겠네
- Early experiments revealed only slight differences between these encodings, with the Radford et al. (2019) BPE achieving slightly worse end-task performance on some tasks. Nevertheless, we believe the advantages of a universal encoding scheme outweighs the minor degredation in performance and use this encoding in the remainder of our experiments. A more detailed comparison of these encodings is left to future work.
  - 의외로 성능상에선 애매한 결과가 있었나..?

### RoBERTa
- We call this configuration RoBERTa for **R**obustly **o**ptimized **BERT a**pporach
- trained with
  - dynamic masking (Section 4.1)
  -  FULL-SENTENCES without NSP loss (Section 4.2)
  - large mini-batches (Section 4.3)
  - large byte-level BPE (Section 4.4)
- investigate two other important factors
  - (1) the data used for pretraining, and 
  - (2) the number of training passes through the data.
- 학습했던 백본 셋팅: we begin by training RoBERTa following the BERTLARGE architecture (L = 24, H = 1024, A = 16, 355M parameters)

![image](https://user-images.githubusercontent.com/7252598/169629631-adc42f43-9068-42ec-b132-68d8c4c08d1e.png)

##### Results
- 학습셋 추가 + 길게 학습하면 성능이 오른다
- gradient accumulation 셋팅은 32정도인가 -> 8k 배치를 위해서

#### GLUE Results
- **In the first setting (single-task, dev)** 
  - consider a limited hyperparameter sweep for each task, **with batch sizes ∈ {16, 32} and learning rates ∈ {1e−5, 2e−5, 3e−5},** 
  - with a **linear warmup for the first 6%** of steps followed by a linear decay to 0. 
  - We **finetune for 10 epochs** and perform early stopping based on each task’s evaluation metric on the dev set
- **In the second setting (ensembles, test)**
  - While many submissions to the GLUE leaderboard depend on multi-task finetuning, our submission depends only on single-task finetuning. For RTE, STS and MRPC we found it helpful to finetune starting from the MNLI single-task model, rather than the baseline pretrained RoBERTa.
![image](https://user-images.githubusercontent.com/7252598/169629833-e1d001d9-bb7e-42ac-a7f1-0d58869e1f11.png)

#### SQuAD Results
![image](https://user-images.githubusercontent.com/7252598/169656755-982ecfd7-0989-46a4-8c3f-688a197c5ca2.png)
- we only finetune RoBERTa using the provided SQuAD training data

#### RACE Results
![image](https://user-images.githubusercontent.com/7252598/169656773-da929dcb-8fc9-4033-9c61-8b26066d5385.png) 

## Conclusion
- We find that performance can be substantially improved by training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data


## Params
![image](https://user-images.githubusercontent.com/7252598/169656859-cb41c844-846a-4a25-a8ed-ee15defde753.png)
![image](https://user-images.githubusercontent.com/7252598/169656899-8ce502b7-c3e7-4bd0-a52c-e913ccf0d04e.png)
