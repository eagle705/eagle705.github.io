
---
layout: post
title:  "온톨로지와 RDF에 대한 개념설명 및 생성방법 "
categories: cslog
comments: true
date: 2018-06-11 12:00:00
tags: nlp, kb
toc: true
---

본 Knowlede base를 구축하기 위해 온톨로지 및 RDF에 관한 설명, 문법 그리고 생성방법에 대해 다룬 문서입니다.

### A. 소개
- Ontology란?
- 왜 Ontology를 만드는가?: 온톨로지는 한 도메인 내에서 정보를 공유하고자 하는 연구자들을 위한 공통된 어휘집(common vocabulary)을 제공한다. 온톨로지를 만들어야 하는 이유중 몇 가지는 사람들 또는 소프트웨어 에이전트 사이에 정보의 구조에 관한 [1] 공통된 이해(understanding)를 공유 [2] 도메인 지식(domain knowledge)을 재사용 [3] 도메인 가설(domain assumptions)을 분명히 [4] 운용 지식(operational knowledge)으로부터 도메인지식(domain knowledge)을 분리 [5]도메인 지식을 분석하기 위함
- 
- RDF란?
- 기본적인 OWL, RDF, RDFS, LINED DATA에 대한 설명:   
http://www.ezmeta.co.kr/page/?p=248   
http://operatingsystems.tistory.com/entry/Basic-of-Semantic-Web?category=578406   
http://operatingsystems.tistory.com/entry/Linked-Data-and-RDF?category=578406   
http://operatingsystems.tistory.com/entry/RDFS?category=578406   
http://operatingsystems.tistory.com/entry/OWL-Web-Ontology-Language   
- 공식 OWL 관련 문서:   
http://www.w3c.or.kr/Translation/REC-owl-features-20040210/   



### B. Ontology 구축 툴
- 조사 결과 Stanford에서 나온 **Protege**라는 툴을 많이 쓰는 것으로 확인됨
- Protégé(프로테제)는 온톨로지 에디터임
- Visualization plugin도 존재함 (https://www.youtube.com/watch?v=yOeSqu30PPQ)
- 설치방법: 접속 후 OS에 맞는 버전 다운로드: http://protege.stanford.edu/
- 튜토리얼: http://wiblee.tistory.com/entry/Protege-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-01-%ED%94%84%EB%A1%9C%ED%85%8C%EC%A0%9C-%EA%B0%9C%EC%9A%94-%EC%84%A4%EC%B9%98
- 그밖: ```protege 사용법``` 검색 in Google
- refs   
- How To build a RDF dataset: https://www.youtube.com/watch?v=leO7__ZonbQ
- How To Create Classes And Properties: https://www.youtube.com/watch?v=MbauHV2-XYw
- visualization: https://www.youtube.com/watch?v=bpjMYBc98bk
- DataType Prop vs ObjectType Prop : https://stackoverflow.com/questions/17724983/how-can-i-recognize-object-properties-vs-datatype-properties
- Running Simple SPARQL Queries: https://www.youtube.com/watch?v=0zUos1zWB5k
- importing data plugin: https://protege.stanford.edu/conference/2009/slides/ImportingDataProtegeConference2009.pdf


### C. RDF 구축 툴
- 파이썬 라이브러리가 있었음 (https://github.com/RDFLib)
