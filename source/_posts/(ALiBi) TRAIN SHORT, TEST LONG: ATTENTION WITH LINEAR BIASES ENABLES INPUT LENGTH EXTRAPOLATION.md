---
layout: post
title:  "(ALiBi) TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION"
categories: paper
comments: true
date: 2022-07-13 12:00:00
tags: nlp
toc: true
---

## Ref
- https://github.com/ofirpress/attention_with_linear_biases
- huggingface 구현
  - https://github.com/huggingface/transformers/blob/ad28ca291bf851b48d7f2d4becf96ca90c98f8f1/src/transformers/models/bloom/modeling_bloom.py#L96
## Author
- 저자: Ofir Press1,2 Noah A. Smith1,3 Mike Lewis2
1Paul G. Allen School of Computer Science & Engineering, University of Washington 2Facebook AI Research 3Allen Institute for AI

## 요약
- extrapolation 잘됨
- 11% 빠름 속도, 11% 메모리 적게씀
- 동일 길이로 학습한 모델 대비 짧은 길이로 학습해도 ppl 유지됨
- 구현도 간단하다
- position embedding 지우고 대신에 길이에 linear하게 비례해서 attention score 깎아버리자!

## Abstract
- 이 질문에 대한 답변이 요구되어져왔음
  - `how does a model achieve extrapolation at inference time for sequences that are longer than it saw during training?`
- position representation만 변경하면됨
  - `Attention with Linear Biases (ALiBi)`
    - not add positional embeddings to word embeddings; instead, it `biases query-key attention scores with a penalty that is proportional to their distance`.
- 1.3B모델에서 1024 -> 2048로 바꿔도 2048모델을 sinusoidal position embedding으로 학습한 모델과 pl 같았고, 학습속도는 11%빠르고 11% 메모리 적게 사용함
  - 여기서 말하고자 하는건 긴 길이로 사용할꺼면 학습도 거기에 맞게 해놔야된다 라는 것 같음
  - 맨 처음 질문 자체가 짧은 길이에 대해서 학습한 후 긴길이에 대해서 사용해도 되냐는거 였으니
- ALiBis' inductive bias는 token의 recency(거리)와 관련있는데, 이게 WikiText-103 benchmark에 있던 multiple strong position methods보다 성능이 좋았음 

## Introduction
- More context, achieved by larger L, improves predictions at inference time. But longer sequences are more expensive to train on
- We define **extrapolation** as a model’s ability to continue performing well as the number of input tokens during validation increases beyond the number of tokens on which the the model was trained
- transformer language models (LMs) that use sinusoidal position embeddings have very weak extrapolation abilities; see Figure 1.
![image](https://user-images.githubusercontent.com/7252598/178661915-adcc3a20-c0c3-4199-be75-9b68b005d9cb.png)
- 최근에 나온 포지셔널 인코딩은 성능 좋긴함.. 하지만!! 느리고 추가 메모리 필요함
  - However, the better of these, the T5 bias, is considerably slower than the sinusoidal approach and uses extra memory and parameters  (Figure 2).
  - ![image](https://user-images.githubusercontent.com/7252598/178676014-ac42f75b-245d-4c12-9ef8-2b262d1d0e29.png)
- introduce Attention with Linear Biases (ALiBi) to facilitate efficient extrapolation. 
  - 한마디로 길이에 따라서 QK로 매칭되는 attention score를 좀 깍아버리고 position embedding은 삭제시켜버리겠다는 것!
  - ALiBi **negatively biases attention scores** with a linearly decreasing penalty proportional to the distance between the relevant key and query. **Our simple approach eliminates position embeddings**
- ALiBi can be imple- mented by changing only a few lines of existing transformer code.

## Background and Experimental Setup
- During both training and perplexity evaluation (i.e., scoring a fixed sequence), many predictions can
be calculated at once; this is done using a “causal mask” that ensures each position’s prediction is
influenced only by tokens to its left
- Let L be the length of each input subsequence during training;
it includes L predictions, which on average have access to (L+1)/2 tokens of (left) context
### sinusoidal
- transformer에서 사용됨
- constant, non-learned vectors that are added to token embeddings on input to the first layer of the transformer.

### Rotary
- `Roformer`: Enhanced transformer with rotary position embedding, 2021.에서 첫 등장함
- has recently been popularized by the open source GPT-3 (Brown et al., 2020) implementation GPT- J (Wang & Komatsuzaki, 2021)에서 자주 사용됨
- Instead of adding sinusoidal embeddings at the bottom of the transformer, they multiply the keys and queries of every attention layer by sinusoidal embeddings.
  - `every attn layer의 key랑 query쪽에 sin embedding을 다 곱한건가`
  - Unlike the sinusoidal or learned positional embedding approach, the rotary method injects position information into the model at every layer, not just at the initial one
  - In addition, it adds no position information to the values of the self-attention sublayer. The output of a self-attention sublayer is a linearly transformed, weighted sum of the input value vectors; therefore, by not inserting position information into the values, the outputs of each transformer-layer contain no explicit position information. We suspect that this segregation of position information may be beneficial for extrapolation, and we draw inspiration from it in the design of our method
      - 포지션 임베딩을 직접 더해주지 않는 방식이 오히려 extrapolation에 더 좋을거라 판단했다는게 그 이유는 뭔지 안알려줌

## T5 bias
-  the T5 model of Raffel et al. (2020) uses a `relative position method` (Shaw et al., 2018; Huang et al., 2019) that `adds no position information to word embeddings` (as in the previous method)
- Instead, it modifies the way attention values are computed. We refer to this as the “T5 bias” method.
- In this method, we compute the attention values as before, but then we add a learned, shared bias to each query-key score that is dependent on just the distance between the query and key.
  - QK score에 +learnable bias를 더해주되 그 bias는 query key distance에 의존성이있도록한다라.. ALiBi처럼 어떻게 보면 attention Score에 직접 개입하는것 같은데..
  - 그래서 성능이 Rotary보단 좋았나보다
  - 그래서 추가적인 메모리가 필요하고 느리다고 했나보다 (모든 레이어에 대해서 bias가 필요해서)
- As in the rotary method, the T5 bias injects position information into the model at every layer and integrates no explicit position information into the self-attention value vectors.

## ATTENTION WITH LINEAR BIASES (ALIBI)
![image](https://user-images.githubusercontent.com/7252598/178683688-0c6fdf7d-ea04-4f42-a26e-94213625793a.png)
- 그림을 보면 learnable한게 없다 (m도 안배움..) 그래서 변수가 없으니 속도도 빠르고, 메모리도 절약했네
- 궁금한건 T5 relative position method는 learnable한걸 쓰는데 겨우 저걸로 어떻게 이긴거지 싶은거야
- we **do not add position embeddings at any point in the network. The only modification we apply is after the query-key dot product**, where we add a static, non-learned bias:
- ![image](https://user-images.githubusercontent.com/7252598/178702621-584132b5-03f9-4fa5-a4b9-ca35f09c6421.png)
  - slope 개념이 잘 이해가 안가네, m에 대한 값을 head따라 다르게 줘보겠다가 핵심인거 같긴한데, 좋은 head에 나쁜 m값이 할당 될 수 있는 리스크를 안고 가는 느낌 (애초에 좋은 head라는걸 우린 알수도없지만)
  - The ALiBi bias is not multiplied by the √dk scaling factor from Equation 1 of Vaswani et al. (2017).
- ALiBi has an inductive bias towards recency; it penalizes attention scores between distant query-key pairs, with the penalty increasing as the distance between a key and a query grows. The different heads increase their penalties at different rates, depending on the slope magnitude.
- We initially experimented with making the slopes trainable, but this did not yield strong extrapolation results
  - 처음에 slope(m)을 trainable하게 해봤지만 좋은 결과를 얻진 못했다고함
  - 왜일까..?
  - trainable하게 하면 속도도 3%정도 느려짐
- Our main insight from this exploration is that the slope sets that work best are those with slopes in the (0, 1) range, with the slopes’ density increasing as we get closer to 0
- We also found our method to be robust to slope choice. Even randomly sampling from the exponential distribution worked well in some cases (although that method had high variance).
  - 이렇게 지수적으로 하도록 (exp 분포에서 랜덤하게 뽑아봐도) m을 정하면 꽤 robust한 결과가 나온다고..
- Since ALiBi is a relative position method, we add position information at every layer to the keys and queries but not to the values, as is done in the T5 bias and rotary methods. We hypothesize that these properties might be beneficial for extrapolation.

### Implementation
- implement it by modifying the mask matrix by adding the linear biases to it (in practice, when training a transformer LM, query qi attends only to keys 1 to i; this is implemented by adding a mask matrix to the query-key dot product before the softmax operation is applied ). This means that there is no runtime penalty when using our method since we add no operations to the network
  - 왜 런타임에는 operation이 없을까.. 학습할때만쓴다? 뭐지..코드봐야되나

## Results
![image](https://user-images.githubusercontent.com/7252598/178723716-a70eeb0f-1dd2-49a6-970a-e9892241b800.png)
![image](https://user-images.githubusercontent.com/7252598/178723943-1dbae7de-0099-4e29-b54b-2783a1b6e564.png)
- 이 그래프는 왜 놨는지 잘 모르겠음, ALiBi의 길이가 더 길때 ppl이 낮다를 보여줘야될것같은데 뭐지..

![image](https://user-images.githubusercontent.com/7252598/178724180-84712785-16f1-4c42-a67a-ce833fc6ba56.png)

## Conclusion 
- showed that the sinusoidal position embedding approach does not enable transformers to extrapolate to inputs longer than the ones they were trained on
- established that extrapolation in transformers can be enabled by just changing the position method
- showed that our ALiBi method offers an extremely simple replacement for existing position approaches and allow models to extrapolate
- ALiBi is simple to implement and does not slow down runtime or require extra parameters
- sped up the training of a 1.3 billion parameter model evaluated on the same input sequence length as GPT-3 (2048)
