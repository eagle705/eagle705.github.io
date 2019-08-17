---
layout: post
title:  "gPRC + Healthcheck 뽀개기"
subtitle:   "프로토콜"
categories: cslog
tags: deeplearning
comments: true
use_math: true
---
본 문서에서는 gRPC에 대해서 가볍게 다루고자한다. 마침 healthcheck 관련 내용도 다룰 일이 생겨서 이참에 정리겸 남겨둔다.

#### gRPC 개념 설명
- [Microservices with gRPC](https://medium.com/@goinhacker/microservices-with-grpc-d504133d191d)

#### gRPC.proto 살펴보기
- filename: projectname.proto

```
// Copyright 2015 The gRPC Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
syntax = "proto3";

option java_multiple_files = true;
option java_package = "com.eagle705.demogrpc.proto.projectname";
option objc_class_prefix = "projectname_prefix";

package projectname;

// Interface exported by the server.
service Chatbot {
  // A simple RPC.
  //
  // Obtains the feature at a given position.
  //
  // A feature with an empty name is returned if there's no feature at the given
  // position.
  
  rpc 호출할함수명(input자료구조명) returns (stream output자료구조명) {}

  // A server-to-client streaming RPC.
  //
  // Obtains the Features available within the given Rectangle.  Results are
  // streamed rather than returned at once (e.g. in a response message with a
  // repeated field), as the rectangle may cover a large area and contain a
  // huge number of features.
  // rpc ListFeatures(Rectangle) returns (stream Feature) {}

  // A client-to-server streaming RPC.
  //
  // Accepts a stream of Points on a route being traversed, returning a
  // RouteSummary when traversal is completed.
  // rpc RecordRoute(stream Point) returns (RouteSummary) {}

  // A Bidirectional streaming RPC.
  //
  // Accepts a stream of RouteNotes sent while a route is being traversed,
  // while receiving other RouteNotes (e.g. from other users).
  // rpc RouteChat(stream RouteNote) returns (stream RouteNote) {}
}

// 프로퍼티에 입력되는 값을 순서를 의미하는 듯?!(TBD: 확인필요)
message input자료구조명 {
  string 프로퍼티1 = 1;
  int32 프로퍼티2 = 2;
  repeated int32 프로퍼티3 = 3; // repeadted는 리스트를 뜻하는 듯
  string 프로퍼티4 = 4;
}

message output자료구조명 {
  int32 프로퍼티1 = 1;
  double 프로퍼티2 = 2;
}
```

### gRPC python file 생성
- grpc module이 설치되어 있어야함
- ```pip install grpcio```

```bash
python -m grpc_tools.protoc -I./ --python_out=. --grpc_python_out=. ./grpc_modules/projectname.proto
```

- 결과: ```projectname_pb2.py```, ```projectname_pb2_grpc.py``` 두가지 파일이 생성됨


### 서버 실행

```python
import grpc
from grpc_modules import projectname_pb2_grpc

# gRPC 서버 실행
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
projectname_pb2_grpc.add_ProjectnameServicer_to_server(projectname_engine, server)
server.add_insecure_port('[::]:8980')
server.start()
try:
    while True:
        time.sleep(_ONE_DAY_IN_SECONDS)
except KeyboardInterrupt:
    server.stop(0)
```

### gRPC python examples
- [Official Repo Examples](https://github.com/grpc/grpc/tree/master/examples/python)

### Health Checking
- gRPC는 Health Checking도 기존 RPC와 동일하게 핸들링함
- [Official gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md)
- [Official gRPC Python Health Checking](https://github.com/grpc/grpc/tree/master/src/python/grpcio_health_checking)


reference:
- http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221313389714&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
- https://grpc.io/docs/quickstart/python/
- https://john-millikin.com/sre-school/health-checking
- https://github.com/grpc/grpc/blob/master/src/python/grpcio_health_checking/grpc_health/v1/health.py
- https://github.com/grpc/grpc/blob/master/doc/health-checking.md
- https://github.com/grpc/grpc/blob/master/src/proto/grpc/health/v1/health.proto


