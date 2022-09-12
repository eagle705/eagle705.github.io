---
layout: post
title:  "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
categories: paper
comments: true
date: 2022-05-23 12:00:00
tags: nlp
toc: true
---

## 용어
- reduce -> 각 프로세스가 가진 값들을 특정한 하나의 process에 연산해서 모으는 연산 (하나의 값으로 모음)
- all+* -> 이름 앞에 all이 붙으면 연산결과를 참여한 모든 프로세스가 동일하게 반환받음
- all reduce -> 하나의 디바이스가 reduce한 값을 참여한 모든 프로세스가 동일하게 받을 수 있게 전달
## 논문파일
[Megatron-LM- Training Multi-Billion Parameter Language Models Using Model Parallelism.pdf](https://github.com/eagle705/presentation/files/8760288/Megatron-LM-.Training.Multi-Billion.Parameter.Language.Models.Using.Model.Parallelism.pdf)


## Ref
- https://www.youtube.com/watch?v=w4a-ARCEiqU

## Author
- 저자: Mohammad Shoeybi 1 2 Mostofa Patwary 1 2 Raul Puri 1 2 Patrick LeGresley 2 Jared Casper 2 Bryan Catanzaro 2


## 느낀점
- gelu이슈로 col parallel 후 아풋값을 유지하기 위해 row parallel 하는게 포인트


## Abstract
- efficient intra-layer model parallel approach that enables training transformer models with billions of parameters.
- can be fully implemented with the insertion of a few communication operations in native PyTorch
- this approach by converging transformer based models up to 8.3 billion parameters using 512 GPU
- achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).


## Introduction
- 메모리를 잘 쓰기 위해서 여러 방법들이 있었음
  - activation checkpointing
- ADAM은 momentum이랑 optimizer state때문에 파라미터마다 해당 정보를 저장해야했음
  - model parallelism overcome this limit by partitioning the model such that the weights and their associated optimizer state do not need to reside concurrently on the processor.
  - Mesh-Tensorflow같은게 대표적인 프레임워크임
- In this work, we implement a simple and efficient model parallel approach using **intra-layer model-parallelism**.
<img width="341" alt="image" src="https://user-images.githubusercontent.com/7252598/169754157-50b7bf9a-7119-47f1-b9e0-fcbc1dc6e9ac.png">

- We show that the **existing BERT architecture** results in model **degradation as the size increases**
  - We overcome this challenge **by rearranging the layer normalization and residual connection** in the transformer layers and show that with this change, results for the downstream tasks on development sets **improve monotonically as the model size increases**
- our contributions are as follows:
  - • We implement a simple and efficient model parallel approach by making only a few targeted modifications to an existing PyTorch transformer implementation.
  - • We perform an in-depth empirical analysis of our model and data parallel technique and demonstrate up to 76% scaling efficiency using 512 GPUs.
  - • We show that **careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased accuracies as the model grows**.
  - • We demonstrate that **scaling the model size results in improved accuracies for both GPT-2 (studied up to 8.3 billion parameters) and BERT (studied up to 3.9B parameters) models**.
  - • We showcase that our models achieve state of the art results on test sets: perplexity on WikiText103 (10.8 ppl), accuracy on LAMBADA (66.5%), and accuracy on RACE (90.9%).

## Background and Challenges
### Transformer Langauge Models and Multi-Head Attention
<img width="489" alt="image" src="https://user-images.githubusercontent.com/7252598/169754953-c984b3e6-8afd-4b20-8b72-34957cb9e8d2.png">

### Data and Model Parallelism in Deep Learning
- One solution to this problem is to employ parameter sharing to reduce the memory footprint of the model (Lan et al., 2019), but this limits the overall capacity of the model.
- Our approach is to utilize model parallelism to split the model across multiple accelerators. This not only alleviates the memory pressure, but also increases the amount of parallelism independently of the microbatch size.
- Within model parallelism, there are two further paradigms:
  - layer-wise pipeline parallelism
    - groups of operations are performed on one device before the outputs are passed to the next device in the pipeline where a different group of operations are performed.
    - However these suffer from inconsistency issues. The GPipe framework for TensorFlow (Huang et al., 2018) overcomes this inconsistency issue by using synchronous gradient decent. This approach requires additional logic to handle the efficient pipelining of these communication and computation operations, and suffers from pipeline bubbles that reduce efficiency, or changes to the optimizer itself which impact accuracy.
  - more general distributed tensor computation.
    - Distributed tensor computation is an orthogonal and more general approach that partitions a tensor operation across multiple devices to accelerate computation or increase model size
    - FlexFlow (Jia et al., 2018), a deep learning framework orchestrating such parallel computation, provides a method to pick the best parallelization strategy. Recently, Mesh-TensorFlow (Shazeer et al., 2018) introduced a language for specifying a general class of distributed tensor computations in TensorFlow (Abadi et al., 2015)

## Model Parallel Transformers
- We take advantage of the structure of transformer networks to create a simple model parallel implementation **by adding a few synchronization primitives**
- We introduce model parallelism in **both of these blocks separately**.
![image](https://user-images.githubusercontent.com/7252598/169761610-46dacd5d-e364-4081-bf6f-95cad8ebdbc3.png)

- Hence, we partition the first GEMM(X) in this column parallel fashion and split the second GEMM(A = [A_1, A_2]) along its rows so it takes the output(Y = [Y_1, Y_2]) of the GeLU layer directly without requiring any communication as shown in Figure 3a. **The output of the second GEMM is then reduced across the GPUs** before passing the output to the dropout layer.
- This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass (g operator) and a single all-reduce in the backward pass (f operator).
```python
 class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, gradient):
        all_reduce(gradient)
        return gradient

# Code 1. Implementation of f operator.
# g is similar to f with identity in the backward and all-reduce in the forward functions.
```
![image](https://user-images.githubusercontent.com/7252598/169782401-33951ac6-8d78-4e07-87bb-5897da92c5b5.png)

- for the **self attention block** we exploit inherent parallelism in the multihead attention operation, partitioning the GEMMs associated with key (K), query (Q), and value (V ) in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU. This allows us to split per attention head parameters and workload across the GPUs, and doesn't require any immediate communication to complete the self-attention
- The subsequent GEMM from the output linear layer (after self attention) is parallelized along its rows and takes the output of the parallel attention layer directly, without requiring communication between the GPUs.
- This approach for both the MLP and self attention layer fuses groups of two GEMMs, eliminates a synchronization point in between, and results in better scaling
<img width="532" alt="image" src="https://user-images.githubusercontent.com/7252598/169794932-272b6e2e-75ed-4f16-adf1-af1453764783.png">

- This enables us to perform all GEMMs in a simple transformer layer using **only two all-reduces in the forward path(self-attn g func에서 한번 MLP g func에서 한번) ** and **two in the backward path (self-attn f func에서 한번 MLP f func에서 한번)**

<img width="1507" alt="image" src="https://user-images.githubusercontent.com/7252598/169791604-8a04422a-41db-4405-9760-768699ec2e11.png">

![image](https://user-images.githubusercontent.com/7252598/169792298-be713f6a-d65f-4474-b497-261765a8da87.png)

![image](https://user-images.githubusercontent.com/7252598/169792336-d486bfeb-a448-490b-998a-b6ac0f918ffc.png)

![image](https://user-images.githubusercontent.com/7252598/169792367-ba5d9965-aa4b-451d-abf7-87d9ed880397.png)

![image](https://user-images.githubusercontent.com/7252598/169794065-ed50cf0d-b2fd-4ba5-9928-f86654544212.png)

![image](https://user-images.githubusercontent.com/7252598/169794100-19cc6d84-535c-4403-85e0-621b0226781f.png)

![image](https://user-images.githubusercontent.com/7252598/169794131-ef94665b-b373-492b-b2c0-94dab4f70c39.png)

- We parallelize the input embedding weight matrix E_{H×v} along the vocabulary dimension E = [E1, E2] (column-wise)
- To reduce the communication size, we fuse the output of the parallel GEMM [Y1 , Y2 ] with the cross entropy loss which reduces the dimension to b × s. **Communicating scalar losses instead of logits** is a huge reduction in communication that improves the efficiency of our model parallel approach.
- Rather than having one GPU compute part of the dropout, layer normalization, or residual connections and broadcast the results to other GPUs, we choose to duplicate the computation across GPUs. Specifi- cally, we maintain duplicate copies of layer normalization parameters on each GPU, and take the output of the model parallel region and run dropout and residual connection on these tensors before feeding them as input to the next model parallel regions
  - 카피 했다는게, 각자 그냥 갖고 있는다는건지.. 아니면 하나가 계산한 값을 같이 쓴다는건지.. 후자는 좀 이상한거같고

## Setup
- In this work we focus on GPT-2 (Radford et al., 2019), a left- to-right generative transformer based language model, and BERT (Devlin et al., 2018), a bi-directional transformer model based on language model masking

### Training Dataset
- filtered out all the documents with content length less than 128 tokens
- used locality-sensitive hashing (LSH) to deduplicate content with a jaccard similarity greater than 0.7
- resulting aggregate corpus contains 174 GB of deduplicated text

### Training Optimization and Hyperparameters
- utilize **mixed precision training with dynamic loss scaling** to take advantage of the V100’s Tensor Cores
````
ref: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
Q: How does dynamic scaling work?
A: Dynamic loss scaling basically attempts to ride the edge of the highest loss scale it can use without causing gradient overflow, to make full use of the FP16 dynamic range.

It does so by beginning with a high loss scale value (say, 2^24), then in each iteration, checking the gradients for overflows (infs/NaNs). If none of the gradients overflowed, gradients are unscaled (in FP32) and optimizer.step() is applied as usual. If an overflow was detected, optimizer.step is patched to skip the actual weight update (so that the inf/NaN gradients do not pollute the weights) and the loss scale is reduced by some factor F (F=2 by default). This takes care of reducing the loss scale to a range where overflows are not produced. However, it's only half the story.

What if, at some later point, training has stabilized and a higher loss scale is permissible? For example, later in training, gradient magnitudes tend to be smaller, and may require a higher loss scale to prevent underflow. Therefore, dynamic loss scaling also attempts to increase the loss scale by a factor of F every N iterations (N=2000 by default). If increasing the loss scale causes an overflow once more, the step is skipped and the loss scale is reduced back to the pre-increase value as usual. In this way, by: reducing the loss scale whenever a gradient overflow is encountered, and Intermittently attempting to increase the loss scale, the goal of riding the edge of the highest loss scale that can be used without causing overflow is (roughly) accomplished.
````
- initializing our weights W with a simple normal distribution W ∼ N (0, 0.02)
- then scale weights immediately before residual layers by 1/root(2N) where N is the number of transformer layers comprised of self attention and MLP blocks
- our optimizer we utilize Adam (Kingma & Ba, 2014) with weight decay (Loshchilov & Hutter, 2019) λ = 0.01.
- use global gradient norm clipping of 1.0 to improve the stability of training large models.
- dropout of 0.1 is used
- to better manage our memory footprint we utilize activation checkpointing (Chen et al., 2016) after every transformer layer.
- For GPT-2 models
  - all training is performed with sequences of 1024 subword units at a batch size of 512 for 300k iterations. 
  - Our learning rate of 1.5e-4 utilizes a warmup period of 3k iterations before following a single cycle cosine decay over the remaining 297k iterations
  - We stop the decay at a minimum learning rate of 1e-5.
- For BERT models
  - use the original BERT dictionary with vocab size of 30,522
  - replace the next sentence prediction head with sentence order prediction
  - use whole word n-gram masking 
  - set the batch size to 1024 and use a learning rate of 1.0e-4 warmed up over 10,000 iterations and decayed linearly over 2 million iterations. Other training parameters are kept the same as (Devlin et al., 2018).

## Experiments
- 실험용 머신
  - 32 DGX-2H servers (a total of 512 Tesla V100 SXM3 32GB GPUs)
  - multi-node deep learning applications, with 300 GB/sec bandwidth between GPUs inside a server via NVSwitch and 100 GB/sec of interconnect bandwidth between servers using 8 InfiniBand adapters per server

### Scaling Analysis
- consider GPT-2 models with four sets of parameters detailed in Table 1
- To have consistent GEMM sizes in the self attention layer, the hidden size per attention head is kept constant at 96 while the number of heads and layers are varied to obtain configurations ranging from 1 billion to 8 billion parameters
- 1.2 billion parameters fits on a single GPU whereas the 8 billion parameter model requires 8-way model parallelism (8 GPUs).
- original vocabulary size was 50,257, 
  - however, to have efficient GEMMs for the logit layer, it is beneficial for the per-GPU vocabulary size to be a multiple of 128.
  - Since we study up to 8-way model parallelism, we pad the vocabulary such that it is divisible by 128 × 8 = 1024, resulting in a padded vocabulary size of 51,200.
    - 그럼 나머지 1024 * 5에 해당하는 5개는 뭘까.. 8 way씩 5개를 쓴건가
- For the **model parallel scaling, a fixed batch size of 8** is used across all configurations.
  - Data parallel scaling is necessary for training many state of the art models which typically use a much larger global batch size.
  - To this end, for the **model+data parallel** cases we fix the global **batch size to 512** for all experiments which corresponds to **64-way data parallelism**.

<img width="407" alt="image" src="https://user-images.githubusercontent.com/7252598/169964881-f9facd09-498c-4171-8f0c-61987e0a3981.png">

-  observe excellent scaling numbers in both settings. For example, the 8.3 billion parameters case with 8-way (8 GPU) model parallelism achieves 77% of linear scaling. Model+data parallelism requires further communication of gradients and as a result the scaling numbers drop slightly. However, even for the largest configuration (8.3 billion parameters) running on 512 GPUs, we achieve 74% scaling relative to linear scaling of the strong single GPU baseline configuration (1.2 billion parameters).
  - weak scaling -> linear 대비 나쁘지않다

### Language Modeling Results Using GPT-2
<img width="472" alt="image" src="https://user-images.githubusercontent.com/7252598/169968501-54d315af-d484-4612-97e4-5c8afb268128.png">

<img width="468" alt="image" src="https://user-images.githubusercontent.com/7252598/169968717-ab24a4bb-89b2-4252-bed1-c3b115e36259.png">

### Bi-directional Transformer Results Using BERT
- Prior work (Lan et al., 2019) found that increasing model size beyond BERT-large with 336M parameters results in unexpected model degradation. To address this degradation, the authors of that work (Lan et al., 2019) introduced parameter sharing and showed that that their models scale much better compared to the original BERT model.
  - BERT에서 스케일업하려면 param sharing이 필요하다는건가..
- We further investigated this behaviour and empirically demonstrated that **rearranging the order of the layer normalization and the residual connections** as shown in Figure 7 is critical to enable the scaling of the BERT-style models beyond BERT-Large.
- 
<img width="462" alt="image" src="https://user-images.githubusercontent.com/7252598/169969872-d9686b5d-7545-48fb-9a20-d68bbd9791e9.png">

- In all cases, the hidden size per attention head is kept constant at 64. 336M and 1.3B models are trained for 2 million iterations while the 3.9B model is trained for 1.5 million iterations and is still training.
- **For finetuning**, we follow the same procedure as (Liu et al., 2019b -> 로버타 논문인듯..?). We **first perform hyperparameter tuning on batch size and learning rate**. Once we obtain the best values, we report the median development set results over 5 different random seeds for initialization
- 
<img width="404" alt="image" src="https://user-images.githubusercontent.com/7252598/169971514-362a89eb-9523-4c9d-972f-0e31ffc5f324.png">

<img width="816" alt="image" src="https://user-images.githubusercontent.com/7252598/169970590-8b90ab49-453d-45a5-a88c-44460f8088a4.png">


## Conclusion and Future Work
- we successfully surpassed the limitations posed by traditional single-GPU-per-model training by implementing model parallelism with only a few modifications to the existing PyTorch transformer implementations
- showed that for BERT models, careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased accuracies as the model size increases
- model size on down-stream task accuracy and achieve far superior results on downstream tasks and establish new SOTA for WikiText103, LAMBADA, and RACE datasets
