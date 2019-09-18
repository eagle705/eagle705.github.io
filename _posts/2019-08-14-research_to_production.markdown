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
- 어떤 데이터를 쓸지 어떤 데이터를 모을 수 있을지 어디까지 라벨링 할 수 있을지, 어떤 데이터로 원하는 데이터를 비슷하게 대체할 수 있을지 등을 생각해야함
- 대략적인 시간을 산정해서 보고해야함
- 논문을 많이 읽으면 읽을수록 좋긴함
- 갖다 쓸수있는건 빠르게 갖다 써야함
  - 케글이나 이런데서 빠르게 참조할 필요가 있음
- 프로토 타이핑은 매우 빠르게~!
- 개인적으로는 처음부터 좋은 모델보단 baseline부터 서서히 올려가는게 결과를 확인하고 model capacity를 조정하면서 추후 모델 선택할 때 좋음
- Speed한 프로토타이핑이 생명
- Hyper Params 에 대한 실험 관리 + feature에 대한 실험 관리 도구가 좀 필요함
- git 관리를 잘해야함
  - [gitignore](https://www.gitignore.io/)
- 안해본 것에 대한 두려움이 없어야함
- DL Framework
  - Prototyping: PyTorch
  - Service: TensorFlow
    - eager mode로 logit + loss 까지 tensor format & shape 확인
    - graph mode로 학습시켜서 pb 추출
  - AutoML: 어떤 오픈소스 쓸지 TBD


### Production
- 환경셋팅 문서화 (한번에 뜨고 설치할 수 있게 ~~도커가 그립다~~)
- 클래스, 플로우 다이어그램: PlantUML (https://meetup.toast.com/posts/117)
- L4 (Load Balancer)
  - L4에서는 1초간 계속 health check를 해서 서버 하나가 꺼지면 떼버리고 다시 살아있으면 붙임
  - 반응못하면 아마 500에러 낼듯
- 네트워크 프로토콜
  - 패킷 찍어보기
  - HTTP, HTTP2, gRPC(HTTP2+Protocol buf), status code 등등 체크
  - Timeout 관리 (conn timeout, read timeout)
  - 서비스로 사용하는 프로토콜의 doc 숙지
  - [HTTP, HTTP2 관련 문서](https://www.popit.kr/%EB%82%98%EB%A7%8C-%EB%AA%A8%EB%A5%B4%EA%B3%A0-%EC%9E%88%EB%8D%98-http2/)
  - [HTTP2 관련문서2_구글](https://developers.google.com/web/fundamentals/performance/http2/?hl=ko)
- healthcheck
  - 프로세스 관리
  - JandiAlert
- 부하테스트
  - 실서버와 동일한 환경인 Sandbox가 필요함
  - nGrinder
- 로그 관리
  - 파이썬 실행 전체 로그 파일로도 남기기~!
  - python gRPC_server.py > /home/디렉토리/logs/python_logs.log 2>&1 &
  - [HTTP protocol에 따른 에러 처리](https://hyeonstorage.tistory.com/97)
- 안해본 것에 대한 두려움이 없어야함
- Jandi Alert Code

```python
import json
import requests
import linecache
import sys
import os

def jandi_alert(body, server, webhook_url, api="name of API"):
    """
    ref: https://drive.google.com/file/d/0B2qOhquiLKk0TVBqc2JkQmRCMGM/view

    ERROR_COLOR = "#FF0000";
    INFO_COLOR = "#0000FF";
    WARNING_COLOR = "#FFFF00";
    DEFAULT_COLOR = "#FAC11B;
    """

    # ref: https://stackoverflow.com/questions/14519177/python-exception-handling-line-number
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    file_name = f.f_code.co_filename
    linecache.checkcache(file_name)
    line = linecache.getline(file_name, lineno, f.f_globals)
    # print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
    file_name = os.path.basename(file_name)

    payload = {
        "body": body,
        "connectColor": "#FF0000",
        "connectInfo": [{
            "title": "___ 서버 이상",
            "description": "server: {}\napi: {}\nfile_name: {}\nLINE {} '{}': {}".format(server, api, file_name, lineno, line.strip(), exc_obj)
        }]
    }

    requests.post(
        webhook_url, data=json.dumps(payload),
        headers={'Accept': 'application/vnd.tosslab.jandi-v2+json',
                 'Content-Type': 'application/json'}
    )


```
