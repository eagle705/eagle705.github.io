---
layout: post
title:  "A Neural Network Solves and Generates Mathematics Problems by Program Synthesis: Calculus, Differential Equations, Linear Algebra, and More"
categories: paper
comments: true
date: 2022-01-10 12:00:00
tags: nlp
toc: true
---

## Author
- 저자: 
    - Iddo Drori1,a,b, Sunny Trana, Roman Wangb, Newman Chengb, Kevin Liua, Leonard Tangc, Elizabeth Kea, Nikhil Singha, Taylor L. Pattic, Jayson Lynchd, Avi Shporera, Nakul Vermab, Eugene Wub, and Gilbert Strang(아니 그 유명한 길버트 스트랭..)a
    - aMIT; bColumbia University; cHarvard University; dUniversity of Waterloo


## 느낀점
- large scale 모델이 생각보다 할줄아는게 많다는걸 알게됨.. 코드로 파인튜닝하면 수학문제 푸는 코드도 만드는구나 (그런 코드가 깃헙에 있었겠지만..!)

## Abstract
- program synthesis을 통해 PLM & code에 finetune된 모델(`Codex Transformer model`)이 수학문제를 풀수있음을 논함
- university-level Mathematics course questions을 생성하는 연구(?)

## Introduction
![image](https://user-images.githubusercontent.com/7252598/148768047-d0363fb4-49f7-45fd-805a-a69b2362738f.png)

- PLM: text, Finetuning with code (from OpenAI)
- novel techniques to automatically `rephrase problems` so neural networks can `synthesize correct executable programs`
- Main Contribution:
  - 특별한 파인튜닝없이도 6개의 MIT 수학코스와 1개의 컬럼비아 대학 코스를 푸는 뉴럴넷을 보임 (table 1)
  ![image](https://user-images.githubusercontent.com/7252598/148768857-2768a859-c67f-4d5b-8285-0b294f0b8426.png)
  - 아웃풋은 an executable program임
  - 채점도 가능하고, 새로운 문제를 만들기도함 (깃헙의 날리지를 다 이런식으로 흡수하면 이런 놀라운 일도 하는 뉴럴넷이 되는건가)
- Adding Context
  - Topic
  - Library (like `sympy`, `streamplot`, ....)
  - Definition Context
![image](https://user-images.githubusercontent.com/7252598/148769645-f8df0531-6dfa-4610-bae2-ead7052ef0e6.png)

## Conclusion
- Codex가 대학수준의 문제를 program synthesis통해 풀고, 채점하고, 생성할 수 있음을 보임
- 단순 PLM은 안된다
