---
layout: post
title:  "Line Message API Setting with Heroku"
excerpt:   "개발환경셋팅"
categories: cslog
tags: deeplearning
comments: true
use_math: true
---

본 문서에서는 Line Message API 활용을 위한 Heroku 셋팅 방법에 대해서 알아보고자한다. (Mac 기준)

#### Reference
Line: https://developers.line.biz/en/docs/messaging-api/building-sample-bot-with-heroku/
Line Developer: https://developers.line.biz/en/
Line Message API blog: https://engineering.linecorp.com/ko/blog/line-messaging-api-samplebot/
Line Chatbot blog: https://m.blog.naver.com/PostView.nhn?blogId=n_cloudplatform&logNo=221245743135&proxyReferer=https%3A%2F%2Fwww.google.com%2F
Heroku: https://dashboard.heroku.com/apps/chatbot-v/deploy/heroku-git

```bash
$ brew tap heroku/brew && brew install heroku
$ heroku login

# Create a new Git repository
$ cd my-project/
$ git init
$ heroku git:remote -a chatbot-v

# Deploy your application
$ git add .
$ git commit -am "make it better"
$ git push heroku master

# if Existing Git repository
# For existing repositories, simply add the heroku remote
$ heroku git:remote -a chatbot-v
```