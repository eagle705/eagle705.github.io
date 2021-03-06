---
layout: post
title:  "Unity Mahine Learning Day"
excerpt:   "ML-Agents"
categories: cslog
tags: deeplearning
comments: true
---

회사에서 업무차 아래 행사에 다녀오게 되었다. 한줄로 정리하자면 강화학습과 모방학습, 로보틱스 등을 하려한다면 유니티야말로 최고의 솔루션이 될 수 있다는 것이다. ML-Agent가 거기에 날개를 달았다는 것이고.

### Unity Mahine Learning Day

장소: 구글캠퍼스 서울
시간: 18.8.21(화) 14:00 ~ 18:00
관련링크: https://onoffmix.com/event/146742
내용:

#### 오용주 팀장 세션
Unity 발전 방향
- Game Engine -> Real Time Rendering Engine
- 기존에 유니티를 게임 엔진으로 보는 사람들이 많았지만 이젠 시네마틱부터 인공지능까지 다양한 분야에 활용될 수 있는 툴로써 발전되고 있음

Unity의 3가지 Principles
- 개발의 민주와 (툴에 제한 받으면 안된다; Personal License)
- 난제 해결
- 성공 돕기

Unity의 3가지 지표
- 2.75B : 27.5억개의 디바이스에서 유니티가 사용
- 60% : AR/VR중의 60%가 유니티 사용
- 50% : 다운되는 앱중의 50%가 유니티 사용

Unity 기술의 3 가지 키워드
- Creation : 생산하는 도구로써 유니티
- Monetization (Ads & Asset Store) : 수익을 창출할 수 있게하는 도구로써 유니티 (광고, 에셋스토어)
- Network (Unity Connect) : 유니티 사용할줄 아는 사람들에 대한 니즈가 계속 증가하고 있음

Unity와 Machine Learning
- Danny Lange (Unity ML Lead; 현 유니티 부사장이자 과거 우버에서 ML Head였음)을 필두로 시작
- Danny Lange 설명: https://en.wikipedia.org/wiki/Danny_Lange
- Unity의 ML은 주로 강화학습(RL), 모방학습(IL)에 집중하고 있음
- RL: 리워드, Trial and Error를 통한 학습, 최적화 러닝
- IL: 리워드 불필요, 데모를 통한 학습, 실시간 인터렉션, 사람 행동 모방
- RL과 IL 두가지를 모두를 위한 것이 Unity ML Agent


#### 제프리 쉬 세션 (Jeffrey Shih; Senior Product Manager, Machine Learning)

Unity가 왜 Machine Learning에서 중요한가
- "Deep learning with Synthetic data will democratize the tech industry"
- 관련링크: https://techcrunch.com/2018/05/11/deep-learning-with-synthetic-data-will-democratize-the-tech-industry/
- 머신러닝, 딥러닝을 위해서는 학습데이터가 필요한데, 이 데이터를 실제 환경에서 수집하는 것은 곧 테스트하는 것을 의미함. 이는 비용 & 위험성 측면에서 문제가 됨
- 작은 시장에 배포해서 테스트할 경우 위험성 및 배포 계획등을 고려해야함
- 사람들을 사서 테스트하기엔 비용이 비쌈
- 그러므로 위의 문제를 해결하기 위해 시뮬레이션 같은 것을 통해 인공적인 데이터를 만들어낼 필요가 있음
- 유니티가 만들어내는 환경은 이를 위한 최적의 조건을 갖춤

OpenAI 연구 사례
- Learning Dexterity
- 관련링크: http://blog.openai.com/learning-dexterity
- 로봇의 복잡한 손동작을 시뮬레이션을 통해 학습
- Sim2Real (Reality gap from the simulated world to the real world)
- 논문 링크: https://arxiv.org/pdf/1712.07642.pdf

#### 민규식 연구원 세션 (한양대학교 미래자동차공학과 석박통합과정)
- 머신러닝, 딥러닝, 강화학습 공부 후 자율주행차 관련 프로젝트 진행
- 유니티는 강화학습 하면서 접하게 되었고 시작한지 1년 됨
- 강화학습 관련 Github: https://github.com/Kyushik/DRL
- 유니티 환경에서 비전과 레이더 두가지를 입력을 통해 '현재상태유지', '가속', '차선변경', '감속'등의 액션 선택에 대해서 학습함
- Unity ML Agent가 없었을 당시, Unity와 Python간에 Socket 통신으로 학습을 했으나 불안정한 부분과 버그가 존재했음
- 예를 들면, 통신이 일정 이상진행되면 끊김, 통신간 동기화 문제, 유니티와 파이썬 코드간의 진행속도 차이, 많은 코딩 등등
- ML-Agent Challenge라는 대회에서 수상하면서 ML-Agent 방식으로 전향함
- 파이썬에서 Action을 보내주면 유니티에서 시뮬레이션이 한스텝 진행되고 그 정보를 다시 파이썬으로 보내 학습하고 다시 Action을 보내는 방식
- 강화학습을 하는 사람들은 대부분 환경에 구속받는 상황인데 유니티는 그런 구속을 벗어날수 있게 도와주므로 배우길 꼭 추천함 
- 유니티가 게임 개발의 민주화라면, ML Agent는 강화학습 환경 개발의 민주화다
- 자율주행 관련 Github: https://github.com/MLJejuCamp2017/DRL_based_SelfDrivingCarControl
- ML-Agent 관련 Github: https://github.com/Kyushik/Unity_ML_Agent

#### 느낀점
- Game뿐 아니라 다양한 분야에서도 유니티가 유용하게 쓰일 수 있음을 확인함
- Unity를 배워두는게 앞으로 미래를 생각할때 좋은 선택이 될 수 있음
- 강화학습이나 로보틱스 같은 분야에서 유니티는 거의 필수로 자리잡을 것으로 보임
- Game Engine -> Real Time Rendering Engine 으로 범위를 확대한 것이 매우 인상적이고 좋은 시도로 보임
- 전체적인 내용은 Unite Seoul 2018 내용과 비슷하나 구체적인 연구 사례를 추가로 알 수 있어서 좋았음


