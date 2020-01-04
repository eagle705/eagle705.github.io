---
layout: post
title:  "gPRC + Healthcheck 뽀개기"
excerpt:   "본 문서에서는 gRPC에 대해서 가볍게 다루고자한다. 마침 healthcheck 관련 내용도 다룰 일이 생겨서 이참에 정리겸 남겨둔다."
categories: cslog
tags: deeplearning
comments: true
use_math: true
---


#### gRPC 개념 설명
- [Microservices with gRPC](https://medium.com/@goinhacker/microservices-with-grpc-d504133d191d)

#### gRPC 사용할 때 주의할점
- retry 이슈
  - gRPC는 HTTP2 기반인데, 양방향 통신이라 커넥션을 계속 붙잡고 있는데, 이게 가끔 30분에 한번씩 끊길때가 있다 (뭐가 헤더 크기를 넘어가면서..어쩌구저쩌구 들었던거 같은데 다시 찾아봐야함)
  - 그럴땐 클라이언트 쪽에서 보낸 요청이 fail되고 서버가 못듣게 되는데 단순히 클라가 한번 더 retry해주면 된다. 
  - 보통 http2를 쓰는 프로토콜은 retry 로직이 필수라한다
- 헬스체크 이슈 (ulimit, channel close, Too many open files)
  - grpc는 status 를 제공하고 health check 프로토콜도 제공한다. ~~어찌다가 try except으로 에러날때 status 코드를 꺼내는 방식으로 꼼수로 구성한적이 있었다..(이럼 안되지.. 이것 때문에 연차썼다가 출근해서 반차처리한 적이..흑흑)~~
  - 이때 grpc connect을 따로 close해주지 않으면 소켓연결이 쌓이게 되고 리눅스 운영체제에서 file open개수에 대한 ulimit을 초과하면 Too many open files 에러가 뜬다
  - 보통 이런경우 ulimit을 올려주면 되지만, 근본적인 에러원인인 소켓 증가 이유를 찾아야했고 찾다보니 health check때 retry 이슈로 except뜬게 쌓이고 있었다는 결론을 내렸다
  - 결과적으로 connect close를 잘해주자 안그러면 too many file opens 에러뜨니까

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
- https://github.com/grpc/grpc/blob/master/doc/statuscodes.md


