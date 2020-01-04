---
layout: post
title:  "2018 패캠퓨처컨퍼런스"
excerpt:   "FastCampus Future Conference 2018 - AI Tracks"
categories: cslog
tags: linux
comments: true
---

FastCampus Future Conference 2018 - AI Tracks

Keywords

- 개인화
- 시각 인지력
- 컨텐츠 생성(Generative+RL)
- 예측

Domains

- 광고,마케칭

- 패션

- 농림, 태블릿

- 인사조직관리

- 교육
- 제조 효율
- 자동주행을 통한 물류
- 소매업을 위한 감시카메라, 추천시스템
- 스마트대출 
- eCommerce
- 공항: 모니터링 (YoLo)

- TTS, 책읽기, 가족감성..
- TTS+챗봇+곰인형+AI Speaker (캐릭터사업)

### Computing Power의 변화와 트렌드

##### NVIDIA 차정훈 상무(9:50-10:30)

30-year era of Moore's law is ending

One Architecture (One-Chip, One-Algorithm)

RTX == real time ray tracing (Scientist는 안된다고했었지만..AI 도움을 통해 해결)

(http://squid.kr/221341532067)

(https://www.top500.org/)

implement of AlexNet is impossible without GPU 

인프라비지니스..

단일매트릭스사이즈 커지면 -> 메모리 사이즈 커져야 -> GPU 사이즈커져야 -> DGX-2(16개 GPU 결합, 최초의 2페타플롭스 시스템)

Hello World < Large Scale





#### Decentralizing AI Computing Power

##### Common Computer 김민현 대표(10:45-11:30)

Ref: https://www.ainetwork.ai/

Leela Zero: https://github.com/gcp/leela-zero

Leela Zero client on Colab's NVIDIA Tesla K80

P2P Network with blockchain (rewards가 확실히 존재하는 P2P라..)

IPFS, 공유하고 있으면 내려가지 않는 웹이 될 수도(Block chain은 IPFS를 연동하는걸 좋아함)

SONM (SONM 토큰을 통해 컴퓨팅파워를 공유하는 마켓플레이스 형성)

Network 유지 - mining - get money - mining software upgrade - more money(immediately)

Reconstruction Attack (Homomorphic Encryption으로 해보자, but performance iussues, Homomorphic operation이 되는 문제에만 해당); 실용적으론 bad (p2p client < AWS 완승)

Dark side of decentralized AI Network(DDos with AI)



#### AI Implicaitons in eCommerce

##### eBay KOREA 현은석 CTO(11:45-12:30)

eCommerce: 오픈마켓(22조; 33%), 소셜커머스, 종합몰, 전문몰..

Robots: 물류로봇 

Chatbots: 작년에 도입하고자했으나 미루는걸로 결론냄, eCommerce에 충분한 효율을 가져다주는거라고 하기엔 부족함. 대부분 도입했으나 거의 접었음. Alaxa 이용객중 쇼핑경험은 2%뿐

DATA: 데이터 정제에 매우 신경씀. BAIKAL 이라는 DATA LAKE를 만들어서 운영중 + Helix라는 DNA 컨셉으로 Platform 운영

ML: workflow Automation for ML services; AskFlow, FxFlow (feature engineering), AiFlow (for production)

AI-BASED FDS SYSTEM: LSTM쓰니 rule-base 했던건 다 잡아냄; Training data를 rule-base로 뽑았을것같은데, 당연히 그럼 rule-base로 하던건 잘 잡아낼거고,, 검증 어떻게하는지..?

Amazon Dynamic Pricing: 쿠키같은걸로 사람가려가며 가격 책정하는...?!

User Clustering: Matrix Factorization(회원 database) > Clustering : Matrix Factorization 할때 크기 엄청엄청 클텐데 흠.. 라이브러리같은거 뭐쓰는지..궁금하기도 ; 

회원정보, 구매기록으로 타겟? X // 구매 특성으로 파악해야



ML/DL 기반 상품 속성 추출: 상품정보 구조화(Structural data)

A quantum-inspired classical algorithm for recommendation systems

AD: audience에 따라 개인화된 노출

 결국엔 비지니스(힘을 받으려면)

시계열분석은 돈이다 라는 얘기도 있다

복잡한 시계열은 ARIMA < LSTM

Hyper parameter tuning : AutoML은 feature에 대한 bias도 없앨 수 있음



전망: Lisa Su; 레거시 장점 활용; 선택과 집중; 미래 성장 동력에 집중





#### AI Implicaitons in Robotics

##### Polariant 장혁 대표(13:20-14:00)

Sense Think Act

인간의 삶의 87%를 실내 공간에서

Naver Labs M1 (실내지도를 로봇이만든다?!) - HD map

'실내 완전자율주행 로보틱스'가 당면한 문제

- Fully Control 

- 24/7

- Robustness



돌파하기위한 기술적과제

- Precise Localization (5m ->>> 30, 10cm 이하 가능해야) : Polariant (6cm정도)
- Periodical indoor mapping for Lastest updated HD-map



PLS 개발키트 플랫폼

- 대학병원
- 마트 & 쇼핑몰
- 드론 플랫폼
- 연구기관





#### AI Implicaitons in Speech Synthesis

##### 네오사피엔스 이영근 Research Scientist(14:10-14:50)

TTS(Text to Speech) 스타트업

성우 - ICEPICK - 컨텐츠

Google, 2017; Tacotron

Baidu, 2018; Clarinet (End to End)

End-to-End TTS에서 어려운점 

- Seq2seq 문제

  - input과 output을 정렬하는 정보가 주어지지 않음

  - Attention이용해서 해결

- One-to-many mapping

  - input text에 대해 생성 가능한 음성의 경우의 수가 무한히 많음

  - 감정, 운율등에 대한 정보가 주어지지 않음
  - 추가적인 input을 주도록 모델을 설계하고 학습함(How? prosody encoder로 억양추가해서 decoder에 encoding vector를 매 스텝마다 concat으로 넣어줌; 그러나 전반적인 감정을 잡아내는건 가능해도 특정 부분을 강조하는등의 시간에 대한 정보는 날라간다는 한계가 있음)
  - Prosody p_i에 대한걸 넣어줌~ (Tacotron with sequential prosody input, 하지만 길이가 다르면 쫌 애매한데.. 이것도 뭐..어텐션이라도 쓰나-> 기존방법으로 초안생성 후, 순차적으로 넣어서 느낌만 바꾼다.. 사실 이게 좀 문제였다는 답변을 함; embedding말고 음.. label 넣은 GAN으로 하면 재밌을듯)
  - 

#### AI Implicaitons in Entertainment

##### SM Ent CT-AI Labs 주상식 (15:00-15:40)

Music AI





#### AI Implicaitons in Manufacturing

##### Pick-it N.V 구성용 (16:40-17:20)

CV기반 Pick it 하는 로봇 (그전까진 쉽지 않았나봄)







