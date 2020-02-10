---
layout: post
title:  "Linux, Unix 정리"
excerpt:   "리눅스 명령어 정리 및 stanford CS1U 강의 요약"
categories: cslog
tags: linux
comments: true
---

## Shorcut
자주 쓰는 명령어 모음
- lshw: 하드웨어 스펙보기
- 예약변수: ```HOME```, ```PATH```, ```PWD```, ```LANG```, 등등


### 위치 매개 변수(Positional Parameters)

| 문자 |	설명 |
| --- | --- |
| $0 |	실행된 스크립트 이름 |
| $1 |	$1 $2 $3...${10}인자 순서대로 번호가 부여된다. 10번째부터는 "{}"감싸줘야 함 |
| $* |	전체 인자 값 |
| $@ |	전체 인자 값($* 동일하지만 쌍따옴표로 변수를 감싸면 다른 결과 나옴) |
| $# |	매개 변수의 총 개수 |

### 특수 매개 변수(Special Parameters)

| 문자 |	설명 |
| --- | --- |
| $$	| 현재 스크립트의 PID |
| $? |	최근에 실행된 명령어, 함수, 스크립트 자식의 종료 상태 |
| $!	| 최근에 실행한 백그라운드(비동기) 명령의 PID |
| $-	| 현재 옵션 플래그 |
| $_	| 지난 명령의 마지막 인자로 설정된 특수 변수 |

----------

### 디버깅(Debugging)
- 간단하게는 echo, exit 명령나 tee 명령어로 디버깅한다.
- 다른 방법으로 실행 시 옵션을 주거나 코드에 한줄만 추가하면 해볼수 있다.


| Bash 옵션(스크립트 실행 시) |	set 옵션(스크립트 코드 삽입) | 설명 |
| --- | --- | --- |
| bash -n	| set -n, set -o noexec	| 스크립트 실행없이 단순 문법 오류만 검사(찾지 못하는 문법 오류가 있을수 있음) |
| bash -v	| set -v, set -o verbose	| 명령어 실행전 해당 명령어 출력(echo) |
| bash -x	| set -x, set -o xtrace	| 명령어 실행후 해당 명령어 출력(echo) |
| | set -u, set -o nounset |	미선언된 변수 발견시 "unbound variable" 메시지 출력 |


### 배열(Array Variable)
- 배열 변수 사용은 반드시 괄호를 사용해야 한다.(예: ${array[1]})
- 참고: 1차원 배열만 지원함

```bash
# 배열의 크기 지정없이 배열 변수로 선언
# 참고: 'declare -a' 명령으로 선언하지 않아도 배열 변수 사용 가능함
declare -a array

# 4개의 배열 값 지정
array=("hello" "test" "array" "world")

# 기존 배열에 1개의 배열 값 추가(순차적으로 입력할 필요 없음)
array[4]="variable"

# 기존 배열 전체에 1개의 배열 값을 추가하여 배열 저장(배열 복사 시 사용)
array=(${array[@]} "string")

# 위에서 지정한 배열 출력
echo "hello world 출력: ${array[0]} ${array[3]}"
echo "배열 전체 출력: ${array[@]}"
echo "배열 전체 개수 출력: ${#array[@]}"

printf "배열 출력: %s\n" ${array[@]}

# 배열 특정 요소만 지우기
unset array[4]
echo "배열 전체 출력: ${array[@]}"

# 배열 전체 지우기
unset array
echo "배열 전체 출력: ${array[@]}"
```

### 반복문(for, while, until)
- 반목문 작성 시 아래 명령어(흐름제어)을 알아두면 좋다.
- 반복문을 빠져 나갈때: break
- 현재 반복문이나 조건을 건너 뛸때: continue

```bash
# 지정된 범위 안에서 반복문 필요 시 좋음
for string in "hello" "world" "..."; do;
    echo ${string};
done

# 수행 조건이 true 일때 실행됨 (실행 횟수 지정이 필요하지 않은 반복문 필요 시 좋음)
count=0
while [ ${count} -le 5 ]; do
    echo ${count}
    count=$(( ${count}+1 ))
done

# 수행 조건이 false 일때 실행됨 (실행 횟수 지정이 필요하지 않은 반복문 필요 시 좋음)
count2=10
until [ ${count2} -le 5 ]; do
    echo ${count2}
    count2=$(( ${count2}-1 ))
done
```

### 조건문(if...elif...else...fi)
- 조건문 작성 시 주의해야될 부분은 실행 문장이 없으면 오류 발생함

```bash
string1="hello"
string2="world"
if [ ${string1} == ${string2} ]; then
    # 실행 문장이 없으면 오류 발생함
    # 아래 echo 문장을 주석처리하면 확인 가능함
    echo "hello world"
elif [ ${string1} == ${string3} ]; then
    echo "hello world 2"
else
    echo "hello world 3"
fi

# AND
if [ ${string1} == ${string2} ] && [ ${string3} == ${string4} ]
..생략

# OR
if [ ${string1} == ${string2} ] || [ ${string3} == ${string4} ]
..생략

# 다중 조건
if [[ ${string1} == ${string2} || ${string3} == ${string4} ]] && [ ${string5} == ${string6} ]
..생략
```

### Practical Unix
프로젝트를 하다보니 얼추 쓸수는 있게 되었는데 뭔가 디테일이 부족하다고 느끼던 차, 스탠포드에서 강의가 있는걸 알게되서 한번 빠르게 정주행해야곘다 생각했습니다. 아래에 간단하게 정리했습니다.

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
- 명령어와 명령어의 연결은 ```"|"``` 기호를 사용함
- ```"|"``` 기호 앞의 명령 결과가 ```"|"``` 기호 뒤의 명령에 입력 데이터로 사용됨

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

## Reference
- https://blog.gaerae.com/2015/01/bash-hello-world.html