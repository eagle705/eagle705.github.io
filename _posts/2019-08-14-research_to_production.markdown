---
layout: post
title:  "Research to Production"
subtitle:   "Logs"
categories: cslog
tags: algorithm
comments: true
use_math: true
---

회사에서 제품을 개발하면서, Research To Production을 하기 위한 프로세스 및 노트들에 대해서 한번 정리를 해야함을 느꼈다. 이런 프로세스에 좀 익숙해져야 언젠가 프로젝트로 잘 리딩 할 수 있을 것 같아서 지금부터 한개씩 정리해보고자한다.


### Research
- 개발 기간이 필요함
- 대략적인 시간을 산정해서 보고해야함
- 논문을 많이 읽으면 읽을수록 좋긴함
- 갖다 쓸수있는건 빠르게 갖다 써야함
- 개인적으로는 처음부터 좋은 모델보단 baseline부터 서서히 올려가는게 결과를 확인하고 model capacity를 조정하면서 추후 모델 선택할 때 좋음
- Speed한 프로토타이핑이 생명
- Hyper Params 에 대한 실험 관리 + feature에 대한 실험 관리 도구가 좀 필요함
- 안해본 것에 대한 두려움이 없어야함


### Production
- 환경셋팅 문서화 (한번에 뜨고 설치할 수 있게 ~~도커가 그립다~~)
- 클래스, 플로우 다이어그램: PlantUML (https://meetup.toast.com/posts/117)
- L4 (Load Balancer)
- healthcheck
- 부하테스트
  - 실서버와 동일한 환경인 Sandbox가 필요함
  - nGrinder
- 안해본 것에 대한 두려움이 없어야함
