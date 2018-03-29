---
layout: post
title:  "Unix부터"
subtitle:   "from Stanford cs1U"
categories: cslog
tags: linux
comments: true
---

### Practical Unix
프로젝트를 하다보니 얼추 쓸수는 있게 되었는데 뭔가 디테일이 부족하다고 느끼던 차, 스탠포드에서 강의가 있는걸 알게되서 한번 빠르게 정주행해야곘다 생각했다. 아래에 간단하게 정리해둔다.

command set: https://www.tjhsst.edu/~dhyatt/superap/unixcmd.html

video:
https://practicalunix.org/video-schedule

#### week 2: Intro
- 사용할 shell (bash, zsh 등등)정하기
- 사용할 eiditor (vim, emacs) 배우기
- shell과 editor 커스터마이징하기 (dot file! ex .zshrc)
- github에 올려놓고 자유롭게 저장, 바꿔쓰기

참고로 난 macOS / iTerm / zsh 환경에서 작업 중
https://practicalunix.org/content/week-2-intro

#### week 3: Pipelines - Input/Output Redirection
- 입출력 대상을 표준 입력(stdin), 표준 출력(stdout), 표준 오류(sterr)를 쓰지 않고 다른 경로인 파일로 재지정 하는 것

##### 표준 입력 재지정(Input Redirection)
- 키보드 입력(표준 입력)을 파일에서 받도록 대체하는 것
- "<" 연산자를 사용해서 키보드로 연결된 표준 입력 방향으로 파일로 변경(명시적)
- cat 명령어를 사용하는 것과 동일한 결과


##### 표준 출력 재지정(Output Redirection)
- 명령의 실행 결과나 에러 상황을 화면에 출력하지 않고 바로 파일로 저장
- ">" 연산자를 파일명 앞에 지정하여 사용함
- ">" 연산자로 출력방향을 지정할 때 목적 파일은 항상 처음부터 다시 작성됨(파일 덮어씀)
- ">>" 연산자를 사용하면, 존재하지 않는 파일이면 ">"과 마찬가지로 파일이 생성되고, 파일이 있는 경우에는 이어서 작성 됨


##### 표준 오류 재지정(Error Redirection)
- 리다이렉션 연산자가 필요없음 (이부분은 아직 잘 모르겠음)

##### 파이프(Pipe), 파이프라인(Pipeline)
- 둘 이상의 명령을 함께 묶어 출력의 결곽 다른 프로그램의 입력으로 전환하는 기능임
- 즉, 명령어의 표준출력을 또 다른 명령어의 표준 입력과 연결 시킬 수 있음
- 명령어와 명령어의 연결은 "|" 기호를 사용함
- "|" 기호 앞의 명령 결과가 "|" 기호 뒤의 명령에 입력 데이터로 사용됨

```bash
ls /bin  /usr/bin | sort | uniq | grep zip
```
```bash
python read-input.py < nums-0-999 >> result.txt
```

##### 기타 명령어
- 'head', 'tail' : 파일의 시작, 끝을 보여줌
- 'tr'
- 'sort'
- 'uniq'
- 'cut'
- 'join'
- 'sed'
- 'awk'
- 'tee'


http://eunguru.tistory.com/89
https://practicalunix.org/content/week-3-pipelines

#### week 4: Grep and Regular Expressions
