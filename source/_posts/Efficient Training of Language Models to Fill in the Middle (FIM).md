---
layout: post
title:  "Efficient Training of Language Models to Fill in the Middle (FIM)"
categories: paper
comments: true
date: 2022-08-01 12:00:00
tags: nlp
toc: true
---

# 발표자료
[(발표)Efficient Training of Language Models to Fill in the Middle.pdf](https://github.com/eagle705/presentation/files/9248738/Efficient.Training.of.Language.Models.to.Fill.in.the.Middle.pdf)


# 느낀점
- 첫인상은 data augmentation 기법에 관련된 내용을 extensive하게 검증했다정도..?
- free-form generation을 하고 싶다에 초점을 두고 논문 전개

# Note
- 50%란게 어떤걸까
  - 데이터셋에서 FIM으로 transformation하는 비율 (FIM 자체는 랜덤하게 짜르니까)
- SPM에서 캐싱이 무슨 의미 일까


# Author
- Mohammad Bavarian ∗ Heewoo Jun∗ Nikolas Tezak John Schulman Christine McLeavey Jerry Tworek Mark Chen
  - OpenAI
- Github
  - https://github.com/openai/human-eval-infilling

# Abstract
- autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply **moves a span of text from the middle of a document to its end**
- While this data augmentation has garnered much interest in recent years, we provide **extensive** evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability
- Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default

# Introduction
- Finally, causal decoder-based language models, like the GPT model series [Radford et al., 2018, 2019, Brown et al., 2020], are trained using the left-to-right next token prediction objective. The largest and most capable generative language models today, such as GPT-3, Codex, LaMDA, GLaM, PaLM, Gopher, Jurassic-1, and Chinchilla
  - due to their superiority in open-ended text generation, in-context learning (using few-shot priming), pretraining computational efficiency
  - also architecturally simpler and generally more effective without task specific finetuning, making them more attractive for inference and deployment.
- 모델 구조에 따라서 infilling하는 능력이 제한되어있다
  - All model classes are limited when it comes to infilling, where the model is tasked with generating text at a specific location within a prompt, while conditioning on both a prefix and a suffix. Left-to-right models can only condition on the prefix. While encoder-only and encoder-decoder models are capable of conditioning on suffixes, the lengths of infill regions seen at training time are typically much shorter than what is useful in practice
    - 여러 application 상황에선 이런점들이 unfortunate로 볼 수 있음
    - This is unfortunate because infilling naturally arises in applications where there is context both before and after the point of generation. For example, in creating a coding assistant, infilling can be used for docstring generation, import statement generation, or for completing a partially written function.
- Our goal in this work is **to address this limitation by adding fill-in-the-middle (FIM) capability to causal decoder-based language models** which are currently the most dominant paradigm for large scale language modelling
  - 왜 limitation을 address하는거에 초점을 맞췄을까.. 해결하는게 보통의 접근방법인데 이것만으로도 의미가 있다는건가
- a simple modification to training data and without changing the model architecture, causal decoder-based autoregressive (AR) language models can learn infilling without compromising their normal left-to-right generative capability.
- we split documents into three pieces at random and move the middle piece to the end:
  - `document → (prefix, middle, suffix) → (prefix, suffix, middle)`
- **concatenate the three pieces** **using sentinel tokens** (`<PRE> prefix <SUF> suffix <MID> middle`)
- Compared to prior work, our work emphasizes the computational efficiency of training FIM models. This emphasis is important given the increased interest in training very large language models, which are very expensive to train and have a substantial energy footprint
- show that models trained jointly on a mixture of FIM transformed data and ordinary left-to-right data achieve the same left-to-right capability while learning how to fill-in-the-middle
  - call this the FIM-for-free property
- use the term 
  - FIM model to refer to any model trained on a mixture of FIM transformed and normal left-to-right data. 
  - We refer to models trained without any FIM data (i.e. 0% FIM rate) as AR models.

### Our contributions
- cetral contributions
  - FIM-for-free property (기존 loss에 영향안줌)
    - extensive scaling study by training a suite of 8 models, with and without FIM, and show that FIM can be learned without compromising the left-to-right capability in pretraining. We examine this claim in both code and language, using both perplexity and sampling-based benchmarks
  - Best practices for FIM in pretraining (FIM rate 찾아냄)
    - clarify the effects of many hyperparameters related to training FIM models using comprehensive ablations. In particular, we study the FIM rate (the probability at which FIM transformation is applied to the data), different variants of FIM transformation, and t**he choice of middle span.**
  - Finetuning inefficiency (finetuning으로 하기엔 pretraining만큼 필요함, 흥미로운점임)
    - An alternative to training FIM models from scratch is to learn this capability by finetuning existing language models. We show that finetuning with FIM is computationally inefficient. While FIM can be learned for free during pretraining, learning FIM during finetuning requires a significant amount of additional compute to reach similar levels of performance as pretraining.
  - New infilling benchmarks
    - need to evaluate the correctness of free-form generated samples. For this, we focus on code where we can use unit tests to evaluate the correctness of long FIM samples
      - use the single-line and multi-line infilling benchmarks introduced by [Fried et al., 2022] by removing non-empty lines of canonical solutions of HumanEval
    - create two new benchmarks called **random span infilling** and **random span infilling light**.
  - Need for sampling evaluations
    - find that changing various hyperparameters in FIM training often leads to negligible differences in FIM test losses but large differences in **sampling based benchmarks**. (어떤 벤치마크지..?, 생성쪽인듯)
- ablation study on both code and language across a range of scales
  - FIM-for-free property를 확인하는 실험셋팅
    - train 8 models from 50M to 6.9B parameters, both with and without FIM, and compare the performance across a variety of autoregressive benchmarks.
  - code와 LM에서 left-to-right LM test loss 비교하는 실험셋팅
    - train 16 models on code for 100B tokens and another 16 models on natural language for 100B tokens. The comparison of these models in terms of normal autoregressive left-to-right language modeling test loss is presented in Figure 1. In both domains, FIM models achieve similar AR test loss as the non-FIM models
  - ![image](https://user-images.githubusercontent.com/7252598/182298137-c2a6674c-d345-4ff5-8861-92857fad8cca.png)
  - org 데이터 반정도만 봐도 left-to-right loss 차이가 거의 없게 나온다
- it is also important to show that the models are in fact learning to infill from FIM training. Figure 2 provides evidence for this in the context of FIM test losses
  - ![image](https://user-images.githubusercontent.com/7252598/182299318-32cea50d-7d56-4d1b-8a86-f44d9e41ecbb.png)
  - 그냥 left-to-right loss는 거의 같은데, FIM loss는 차이가 나니까 FIM이 AR모델보다 낫다고 얘기하는것 같은데

# Evaluation
- AR loss refers to the cross entropy loss on normal left-to-right data and 
- FIM loss as the loss on 100% FIM transformed data
- All test losses are in nats per token unit. 
- In all sampling-based benchmarks, we use nucleus sampling [Holtzman et al., 2020] with a nucleus parameter of 0.95

## Autoregressive evaluation
- benchmarks with the exception of DROP and QuAC are evaluated with few-shot prompting. For code, we measure the pass rates on HumanEval

## Infilling evaluation
- To create FIM tests, we apply the FIM transformation to the examples from the AR test sets with a FIM rate of 100%.
- Using the same underlying examples in FIM and AR test sets allows us to compare FIM and AR test losses
- create a masked version of these test sets where we only measure the loss on the middle span tokens. The latter test sets are used to measure `P (middle∣prefix, suffix)` for FIM models and `P(middle∣prefix)` for AR models allowing us to investigate the amount of information FIM models gain by being able to condition on the suffix.
  - 이건 FIM이 정보 더 보는데 약간 애매하지않나..
- For generative infilling capabilities, we focus on code since we are interested in free-form generation in contrast to single or few token generations common in cloze-style natural language benchmarks
  - 코드는 open ended generation이여도 정답 체크가 가능하니까 장점이 있다 (The advantage of working with code is that we can use test suites to evaluate the correctness of samples in our tasks even when evaluating long samples from open-ended generations.)
- All the sampling based infilling benchmarks
  - `single-line`, `multi-line`, and `random span infilling`
  - partial function completions tasks created by removing middle spans from the canonical solutions of HumanEval
    - use the single-line and multi-line infilling benchmarks proposed by [Fried et al., 2022] where different spans of non-empty lines in the canonical solutions of HumanEval are turned into a FIM task
- create a new benchmark called random span infilling2, where for each HumanEval problem, we create infilling tasks by selecting the middle span from the canonical solution uniformly at random. We show an example of such a task below where the model must predict the highlighted section (or an alternative completion accomplishing the same goal)
- 
![image](https://user-images.githubusercontent.com/7252598/182306509-f7a73b1f-d8be-4b0c-a755-2cb040ffcb69.png)
- also use random span infilling light, a smaller version of random span infilling, with only one random FIM task per HumanEval problem and just 164 tasks, to track the infilling capability trends **during training**.
- FIM can be prepared in two different ways denoted as PSM and SPM. We report just the SPM infilling results for brevity, except in cases when the use of PSM changes the conclusions.

# FIM training and inference
- implement FIM using a random transformation applied to our dataset
- experiment with two different implementations: **document** level and **context** level
  - difference between the two is at which stage of the data loading pipeline the FIM transformation occurs.
  - This choice naturally arises because a long document can be broken into many contexts, or a context can contain multiple documents when the documents are small.
- document-level case
  - In document-level FIM, with a certain probability p called the FIM rate (we use p = 0.5 for our main suite of models), we cut each document into three parts: prefix, middle, and suffix. **We perform this split prior to tokenization, when the document is still a sequence of characters.** (토큰화하기 전에 3등분한다)
  - split uniformly at random, which means the lengths of prefix, middle, and suffix are each 1/3 of the full document in expectation. (랜덤하게 3등분씩 잘라~)
- encode each of the three sections separately and prepend sentinel tokens to the beginning of each section. We denote these sentinel tokens by `<PRE>`, `<MID>`, and `<SUF>`
- PSM, SPM 등등으로 구성가능, 문서사이는 `<EOT>`로 분리
- **keep the loss on all three sections prefix, middle, and suffix**, so FIM training does not cause a decrease in the autoregressive learning signal.
  - section에 대한 loss를 backward할때 고려안한다는건가..? 아니면 해서 도움이 된다는건가 -> 다 계산한다는 의미일듯 그래야 AR효과가 나서..?!
  - this choice is crucial for the FIM-for-free property to hold. This property does not change whether the sentinels are masked or not; however, it is important to always train on the `<EOT>` tokens as it signals a successful join to the suffix.
![image](https://user-images.githubusercontent.com/7252598/182312000-6e668059-ef3a-49c5-a29f-eafb0e35eff0.png)
  - 무슨 의미일까

## SPM mode
- a variant of the above procedure where we swap the order of prefix and suffix, called SPM, to emphasize the changing of the order to suffix, prefix, and middle
- Our main motivation for introducing SPM is improved key-value caching during inference.
- The reason for this advantage is that with SPM, appending tokens to the prefix no longer invalidates the keys and values computed in the suffix section
  - 무슨의미일까
-  Note that superiority of SPM caching is not universal and may depend on the applications. In particular, in the SPM mode, minor changes to the suffix will invalidate the cache for prefix, but we expect changes to the suffix to be rarer than changes in prefix in real workloads. Interestingly, we find in Section 4.3 beside the caching advantages, **SPM in fact has also a slight edge over PSM in the infilling benchmarks**.
- apply the FIM transformation with 50% probability in PSM mode and with 50% probability in SPM mode, so the model is able to handle both types of formatting in inference. In other words, each mode inherits half of the total FIM rate p
![image](https://user-images.githubusercontent.com/7252598/182314481-b629b487-e0d9-479c-a6c5-04406f86b7ff.png)
- 여기서는 joint training의 효과를 극대화하기 위해 SPM의 variant를 활용함 (괴랄하다 괴랄해..)
![image](https://user-images.githubusercontent.com/7252598/182315670-4b0b8729-985a-421c-a915-59a141a28dbc.png)

## Context-level FIM
- In language model training, documents are often joined with a boundary token, referred to as `<EOT>`, and are then chunked to the model context length. When applying FIM to long documents, this operation can result in fragmented FIM data where the entire prefix or suffix could get cut out of the context during chunking
- we can **apply FIM after the chunking step**. A context slice may have multiple documents in them joined with the `<EOT>` boundary token
- we split based on `<EOT>`, turn some of the documents into FIM examples with probability given by the FIM rate, and join the examples again with `<EOT>`
  - 문서마다 `<EOT>`로 짤라서 FIM하고 다시 `<EOT>`로 붙였다는 말
```python
def token_level_psm_fim(document: str, vocab: Vocab) -> List[int]:
    tokens = vocab.encode(document)
    prefix, middle, suffix = randomly_split(tokens)
    return [
        vocab.sentinel("prefix"), *prefix,
        vocab.sentinel("suffix"), *suffix,
        vocab.sentinel("middle"), *middle,
]
def character_level_psm_fim(document: str, vocab: Vocab) -> List[int]:
    prefix, middle, suffix = randomly_split(document)
    return [
        vocab.sentinel("prefix"), *vocab.encode(prefix),
        vocab.sentinel("suffix"), *vocab.encode(suffix),
        vocab.sentinel("middle"), *vocab.encode(middle),
]
```
- show this technique can boost performance relative to document-level FIM, and **adopt context-level FIM in all our main FIM runs in this work**

# Pretraining results
## Evaluation of left-to-right capabilities in downstream benchmarks
- train a series of models from 50M to 6.9B parameters from scratch with and without 50% FIM augmentation on natural language and code domains
- evaluate our models on a suite of standard downstream benchmarks, the result of which is presented in Figure 3. We again find that joint FIM pretraining does not result in any degradation in standard AR benchmarks as the performance matches within error for both natural language and code.
- 성능 깍아먹지 않고 비슷하게 잘 나온다가 포인트 (10^9 == B 단위에서도 다 비슷비슷)
![image](https://user-images.githubusercontent.com/7252598/182321849-3d477644-c9e3-4c6c-b0b7-dcc0d4c9c60c.png)

## FIM rate
- Questions
  - Does FIM-for-free still hold even at higher FIM rates? If so, how high can we increase the FIM rate without degrading the left-to-right capabilities?
  - Does using a higher FIM rate lead to stronger FIM capabilities? Or does the benefit saturate after a threshold?
- Settings
  - 6 large models with FIM rates (0, 0.25, 0.5, 0.75, 0.9, 1.0) for 50B tokens
- results
  - FIM rate even up to 90% does not cause any degradation in left-to-right capabilities. However, there is a clear sign of degradation in ordinary AR test loss with 100% FIM rate
  - ![image](https://user-images.githubusercontent.com/7252598/182322976-92a7f52b-ca8c-48d6-a245-e1f5a8295596.png)
  - On the other hand, we find that the FIM rate does significantly affect infilling capabilities. Even though the gain in FIM perplexity in Figure 4 due to a higher FIM rate is negligible, increasing this rate yields a consistent improvement in the infilling pass rate as shown in the right plot in Figure 5.
  - ![image](https://user-images.githubusercontent.com/7252598/182323414-23a0ad81-1b37-49a7-9dd3-2df7568cc62e.png)
- Appendix B Scaling trends for FIM rate ablations
  - ![image](https://user-images.githubusercontent.com/7252598/182324160-b50b1bc6-5ccd-4e69-865e-968038e7fc1d.png)

## SPM vs PSM vs joint SPM+PSM training
- The main finding is that SPM is slightly stronger than PSM in our benchmarks in general as evidenced by Figure 6.
  - ![image](https://user-images.githubusercontent.com/7252598/182325009-efe8e6d5-f509-48bb-a4ea-f1b71145ccc7.png)
  - This is likely due to the fact that in SPM, there is no distinction between the prefix and the middle sections as they are one contiguous sequence of text. This makes it more natural for the model to continue from the prefix in contrast to PSM where attention has to first identify where the span token is.
  - However, this does not imply that we should train solely on SPM. In Table 1, we train large models on pure PSM, pure SPM, and our default 50-50 SPM+PSM mix, and evaluate them in all modes. 
  - Not only is joint pretraining the most efficient, but it also yields the most flexible model with two inference modes.
  - ![image](https://user-images.githubusercontent.com/7252598/182314481-b629b487-e0d9-479c-a6c5-04406f86b7ff.png)

## Context-level vs document-level FIM
- **context-level** and document-level FIM, where augmentation is applied either **before** or after packing and chunking
- **perplexity evaluation does not always capture the gains in the sampling performance**. (4.2 참고)
![image](https://user-images.githubusercontent.com/7252598/182326776-cb78e2e8-b133-4f40-881f-b3a545f693fa.png)
- document-level FIM can result in fragmented FIM data with a missing prefix and/or suffix from the chunking step of data loading pipeline. Figure 8 (left) shows that training on these invalid examples in document-level FIM does not affect the left-to-right evaluation. Hence, practitioners might still sometimes prefer document-level FIM due to its simpler implementation.

## Middle span selection
- An important consideration in FIM training is the choice of middle span.
- select spans in three different ways, splitting randomly by lines, tokens, and characters. The section boundaries are selected uniformly at random from the allowed splitting positions based on the span type. Here, a token refers to a word in the byte-pair encoding (BPE) vocabulary. In practice, this is implemented by applying the FIM augmentation after the documents are encoded with BPE (see Appendix C). For simplicity, we run all our experiments in PSM mode in this ablation.
-  line-based middle spans 
  - gives the models a slight advantage in the single-line and multi-line infilling benchmarks
  - On the other hand, the line based training fails almost completely in the random span infilling benchmark
- token-level random spans
  - does slightly better on random span infilling, but is still not competitive compared to character-level runs on this benchmark
- character level
  - subtokens are introduced naturally at the beginning and the end boundaries of the middle section. There is no train-test mismatch and the model is able to understand and solve more random span infilling tasks while still performing well in single-line and multi-line infilling.

![image](https://user-images.githubusercontent.com/7252598/182329347-9fcd8fbe-d7f1-48ae-a56b-7e93a387dc4a.png)

# Finetuning results
- 결론은..finetuning보단 pretraining으로 하는게 나을 수 있다. finetuning도 50B tokens& 90% FIM로 해야 그나마 비슷해진다
  - investigate whether we can finetune existing AR models to learn the FIM capability.
  - 16 models with that of the XL model trained for 100B tokens with a FIM rate of 50% without any finetuning. It is evident from this figure that even with significant additional finetuning compute, AR models finetuned with FIM do not reach the same performance as the models pretrained with FIM (and without any FIM finetuning)
- More generally, we find that higher learning rate, FIM rate, and longer finetuning all seem helpful for improving FIM performance in finetuning.
![image](https://user-images.githubusercontent.com/7252598/182331302-3e7317b6-5bc8-432d-b696-f6ae8d60dc46.png)

# Discussion
- Pretraining vs finetuning
  - The main intuition for why FIM can be learned for free in pretraining is that breaking a document into three pieces and shifting the middle one to the end effectively creates three smaller documents.
    - In particular, each piece still requires predicting next tokens from left to right, keeping the total number of tokens processed autoregressively the same
  - On the other hand, even though FIM data is locally identical to autoregressive data, FIM does impose a different global attention pattern over the whole document.
    -  show the causal attention mask of a FIM document in Figure 10. These new attention pattern could be the reason why it takes a relatively long token horizon and a high learning rate to learn FIM in finetuning
  - ![image](https://user-images.githubusercontent.com/7252598/182337942-d33d5c30-6489-4647-83c4-08fd8e37fa9f.png)
- FIM loss, AR loss, and **the difficulty of FIM task**
  - There is substantial evidence that FIM can often be much harder than normal left-to-right generation.
  - Intuitively, it is often easier to continue a text in a plausible manner **than to continue the text conditioned on ending in a specific suffix.** The latter requires planning a plausible narrative connecting the two pieces, starting the generation in a way that matches the prefix, and stopping the generation at the right time so it connects to the suffix
  - in FIM the model is trained to generate <EOT> when the middle ends and connects to the suffix. On the other hand, when the model fails to produce <EOT> in the allotted budget, we often end up with truncated samples which do not connect well to the suffix. For example, consider the following:
    - prefix뿐만 아니라 suffix하고도 문맥상 잘 맞게 생성해야되서 더 어렵다.. post processing으로 커버도 어렵고
      - this type of failure is more troublesome in FIM since a failure to connect to the suffix cannot easily be fixed by post-processing.
    - ![image](https://user-images.githubusercontent.com/7252598/182401957-5f854ff7-0053-4055-b762-887e268444b0.png)
    - Both completions above connect well to the prefix, but only the first manages to connect well to the suffix. The second completion in contrast fails to produce <EOT> in the allotted budget resulting in
a bad sample.5 This turns out to be a common failure in FIM sampling
  - Appendix H
    - 성공케이스 
      - ![image](https://user-images.githubusercontent.com/7252598/182402748-dee89b18-a860-4ce3-9708-b9b7e9ec41e1.png)
    - 실패케이스
      - ![image](https://user-images.githubusercontent.com/7252598/182403191-5c64ba3b-80b6-480a-bd12-eb0afb8ccbd7.png)
    - 완충케이스 (numbered items로 힌트주기)
      - ![image](https://user-images.githubusercontent.com/7252598/182404194-1a415840-f332-4f77-99d5-83baec7506d3.png)
  - PPL도 FIM이 AR보다 더 높은걸 보면 FIM이 좀 더 어려운 task로 볼 수 있음
  - 아래 그래프는 모두 FIM 모델이고 test loss의 type만 바꾼거임
    - ![image](https://user-images.githubusercontent.com/7252598/182406636-a2808ae5-314c-440f-867d-dc696b85ac1c.png)
  - Context-level vs document-level FIM and FIM rate
    - The basic observation is that document-level FIM effectively leads to a lower FIM rate compared to context-level FIM, even with the same nominal value of FIM rate.
    - 이하 생략

# Related work
- similar to this work, (utilize left-to-right autoregressive modeling by moving the infill regions to the end of context, with regions separated by sentinels)
  - [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26.pdf)
  - [CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET (Facebook AI Research)](https://arxiv.org/abs/2201.07520)
  - [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999)

# Conclusion
- show that causal decoder-based language models can learn to fill in the middle of a document after being jointly trained on a mixture of traditional left-to-right and FIM transformed data
- A single FIM model can import modules, write docstrings, and complete functions, subsuming specialized models finetuned for individual tasks [Chen et al., 2021], providing substantial extra capability over traditional left-to-right language models.
- One important finding here is the FIM-for-free property.
  -  FIM models achieve the same test loss as AR models on left-to-right test loss while achieving lower FIM loss
-  investigate FIM finetuning since a lot of the existing language models do not have FIM capabilities. Our results demonstrate that a canonically pretrained left-to-right model does not acquire the new skill to the fullest extent of the given model size even with careful hyperparameter tuning and a significant amount of finetuning compute relative to pretraining. This suggests that for the best FIM performance, pretraining jointly from scratch with our recommended hyperparameters is more effective than finetuning.
- use the infilling code benchmarks from InCoder [Fried et al., 2022] and introduce the new random span infilling benchmarks based on HumanEval [Chen et al., 2021]. From these, we learn a few important lessons
  - First, **perplexity does not reflect the true infilling performance**, and one should design the infilling benchmarks carefully to measure progress
  - Second, FIM capabilities depend considerably on the FIM rate and implementation like context-level FIM but left-to-right capabilities are unaffected by these choices as long as the FIM rate is kept below 100%
  - Third, applying FIM at the character level imbues the model with natural robustness to subtokens and makes it possible to deploy the model in the wild, for example, as a coding assistant.

## Recommended FIM hyperparameters
- FIM transformation at the character level and always including some character- level random spans as it allows the model to generate sensible completion even when the prefix and suffix end in the middle of a token
-  pretraining with joint PSM and SPM yields the best performance due to a positive transfer between the two formats
- context-level FIM is superior but document-level FIM is also an option if a simpler implementation is desired
- Finally, we observe improved performance even up to a FIM rate of 90% without any cost in AR capabilities
  -  In practice, any value in the range between 50% and 90% is a reasonable choice.
  - Note that this is in contrast with some related prior work such as InCoder [Fried et al., 2022] which typically uses lower values of FIM rate such as 15%, which our results indicate to be suboptimal.

## Future directions
- Smarter span selection
- Steerable generation
- Further examination of the FIM-for-free property
- Multiple infilling slots
- Improving natural language FIM performance
- Role of bidirectionality and attention

Finally, our experience with the FIM-for-free property brings up the intriguing question of *what other useful skills can be learned jointly with no or little cost to the original capabilities of language models.*
We propose the following methodology to help advance research toward answering this question:
  1. Establish a budget in the amount of original capabilities that one is willing to sacrifice to learn a new capability.
  2. Maximize the new capability within this budget.
