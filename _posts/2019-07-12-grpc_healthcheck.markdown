---
layout: post
title:  "gPRC + Healthcheck 뽀개기"
subtitle:   "프로토콜"
categories: cslog
tags: deeplearning
comments: true
use_math: true
---

본 문서에서는 gRPC 뽀개기(?)에 대해서 다루고자한다. 마침 healthcheck 관련 내용도 다룰 일이 생겨서 이참에 정리겸 남겨둔다.

reference:
- https://grpc.io/docs/quickstart/python/
- https://john-millikin.com/sre-school/health-checking
- https://github.com/grpc/grpc/blob/master/src/python/grpcio_health_checking/grpc_health/v1/health.py
- https://github.com/grpc/grpc/blob/master/doc/health-checking.md
- https://github.com/grpc/grpc/blob/master/src/proto/grpc/health/v1/health.proto


