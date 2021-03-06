---
layout: post
title:  "DBpedia 셋팅"
excerpt:   "Knowledge base 셋팅을 위하여.."
categories: cslog
tags: deeplearning
comments: true
---

본 문서는 챗봇과 Knowlede base를 연동하기 위해 DBpedia를 설치하고 사용하는 내용을 다룬 문서입니다.

### A. 소개

- 오픈소스 명: DBpedia
- 기본적인 OWL, RDF, RDFS, LINED DATA에 대한 설명:   
http://operatingsystems.tistory.com/entry/Basic-of-Semantic-Web?category=578406   
http://operatingsystems.tistory.com/entry/Linked-Data-and-RDF?category=578406   
http://operatingsystems.tistory.com/entry/RDFS?category=578406   
http://operatingsystems.tistory.com/entry/OWL-Web-Ontology-Language   
- 공식 OWL 관련 문서:   
http://www.w3c.or.kr/Translation/REC-owl-features-20040210/   
- Reference:    https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/
- Reference2: http://pacifico.cc/programming/virtuoso-dbpedia-setup
- Virtuoso vs Neo4j: https://db-engines.com/en/system/Neo4j%3BVirtuoso
- Docker 기반 DBpedia_virtuoso 설치: https://github.com/harsh9t/Dockerised-DBpedia-Virtuoso-Endpoint-Setup-Guide


### B. 설치
#### virtuoso 설치
```
brew install virtuoso
vi ~/.zshrc
// 수정 아래처럼!
export VIRTUOSO_HOME="/usr/local/Cellar/virtuoso/7.2.4.2"
export PATH=$PATH:$VIRTUOSO_HOME/bin
```


#### virtuoso 실행
/bin/virtuoso-t 루트로 실행하면 되는데
에러가 발생함
```
dyld: Library not loaded: /usr/local/opt/xz/lib/liblzma.5.dylib
  Referenced from: /usr/local/bin/virtuoso-t
  Reason: Incompatible library version: virtuoso-t requires version 8.0.0 or later, but liblzma.5.dylib provides version 6.0.0
```
-> 해결하기 위해서 
```
brew install xz
```
해주면 간단히 해결됨

그후 ini파일이 있는 곳에서 virtuosos-t +f 로 실행   
http://localhost:8890/conductor  에 접속해서 테스트!   

종료할 땐, 
```
ìsql
shutdown();
```

기본적인 계정 정보 (id:dba / pw:dba)
http://docs.openlinksw.com/virtuoso/defpasschange/


#### DBpeida 다운로드

```
# see comment above, you could also get another DBpedia version...
sudo mkdir -p /usr/local/data/datasets/dbpedia/2016-10
cd /usr/local/data/datasets/dbpedia/2016-10
wget -r -nc -nH --cut-dirs=1 -np -l1 -A '*.*' -A '*.owl' -R '*unredirected*' http://downloads.dbpedia.org/2016-10/{core/,core-i18n/en,core-i18n/ko,dbpedia_2016-10.owl}

brew install pigz pbzip2
for i in core/*.*.bz2 core-i18n/*/*.*.bz2 ; do echo $i ; pbzip2 -dc "$i" | pigz - > "${i%bz2}gz" && rm "$i" ; done
// bz2 파일을 virtuoso에서 읽을 수 없음. 압축 안된거거나 gz파일이여야함
```

아래 폴더 셋팅 잘 해주면 좋음
```
cd /usr/local/data/datasets/dbpedia/2016-10/
mkdir importedGraphs
cd importedGraphs

mkdir dbpedia.org
cd dbpedia.org
# ln -s ../../dbpedia*.owl ./  # see below!
ln -s ../../core/* ./
cd ..

mkdir ext.dbpedia.org
cd ext.dbpedia.org
ln -s ../../core-i18n/ko/* ./

cd ..

mkdir pagelinks.dbpedia.org
cd pagelinks.dbpedia.org
ln -s ../../core-i18n/ko/page-links_ko.* ./
cd ..

mkdir topicalconcepts.dbpedia.org
cd topicalconcepts.dbpedia.org
ln -s ../../core-i18n/en/topical-concepts_ko.* ./
cd ..

mkdir ko.dbpedia.org
cd ko.dbpedia.org
ln -s ../../core-i18n/ko/article-categories_ko.nt.gz ./

cd ..

mkdir pagelinks.ko.dbpedia.org
cd pagelinks.ko.dbpedia.org
ln -s ../../core-i18n/ko/page-links_ko.nt.gz ./
cd ..
```

#### virtuoso 로 import

```
isql
ld_add('/Users/eagle/datasets/dbpedia/2016-10/dbpedia_2016-10.owl', 'http://dbpedia.org/resource/classes#');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/dbpedia.org', '*.*', 'http://dbpedia.org');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/ko.dbpedia.org', '*.*', 'http://ko.dbpedia.org');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/ext.dbpedia.org', '*.*', 'http://ext.dbpedia.org');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/pagelinks.dbpedia.org', '*.*', 'http://pagelinks.dbpedia.org');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/pagelinks.ko.dbpedia.org', '*.*', 'http://pagelinks.ko.dbpedia.org');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/importedGraphs/topicalconcepts.dbpedia.org', '*.*', 'http://topicalconcepts.dbpedia.org');

또는 아래 명령어로 얻어진 쿼리 이용(이때 현재 디렉토리 위치는 /Users/eagle/datasets/dbpedia/2016-10)   

for g in * ; do echo "ld_dir_all('$(pwd)/$g', '*.*', 'http://$g');" ; done   

명령어 결과   
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/core', '*.*', 'http://core');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/core-i18n', '*.*', 'http://core-i18n');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/dbpedia_2016-10.owl', '*.*', 'http://dbpedia_2016-10.owl');
ld_dir_all('/Users/eagle/datasets/dbpedia/2016-10/robots.txt', '*.*', 'http://robots.txt');
```

저장하기 (위의 쿼리가 실질적으로 실행되는 부분, 시간 꽤 걸림!)
```
rdf_loader_run();
```

테스트
```
SELECT * WHERE {?s ?p ?q} LIMIT 10

또는

select * where {
 ?s <http://ko.dbpedia.org/property/장소> ?o
 } LIMIT 100
 
 
 default Grahp IRI: http://core-i18n  (첨 만들때 넣어줬던게 Graph IRI인듯, 나중에 Jena와 연동시 ```FROM``` 부분에 들어갈 부분! )
 SELECT * WHERE {?s ?p ?q} LIMIT 1000
 
 SELECT * WHERE {?s a <http://dbpedia.org/ontology/SoccerClub> } LIMIT 1000 (a가 예약어임)
 
 SELECT * WHERE {?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/SoccerClub> 
 } LIMIT 1000
 
PREFIX  dbpedia-owl:  <http://dbpedia.org/ontology/>
SELECT * WHERE {?s a <http://dbpedia.org/ontology/SoccerClub>.
?s dbpedia-owl:abstract ?abstract.
 } LIMIT 1000
 
 PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX  dbpedia-owl:  <http://dbpedia.org/ontology/>
PREFIX res: <http://ko.dbpedia.org/resource/>
SELECT * WHERE {?fc rdfs:label  "FC 즈브로요프카 브르노" @ko.
?fc dbpedia-owl:abstract ?abstract
 } LIMIT 1000
 
 PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX  dbpedia-owl:  <http://dbpedia.org/ontology/>
PREFIX res: <http://ko.dbpedia.org/resource/>
SELECT * WHERE {?name rdfs:label  "컴투스" @ko.
?name dbpedia-owl:abstract ?abstract
 } LIMIT 1000
 
 
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX  dbpedia-owl:  <http://dbpedia.org/ontology/>
PREFIX res: <http://ko.dbpedia.org/resource/>
SELECT * WHERE {?name rdfs:label  "컴투스" @ko.
?name dbpedia-owl:abstract ?abstract.
?name dbpedia-owl:wikiPageWikiLink ?Link.
 } LIMIT 1000
 
 PREFIX foaf: <http://xmlns.com/foaf/0.1/>
 PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
 PREFIX dbpedia-owl:  <http://dbpedia.org/ontology/>
 PREFIX res: <http://ko.dbpedia.org/resource/>
 SELECT * WHERE {?name rdfs:label  "컴투스" @ko.
 ?name dbpedia-owl:wikiPageWikiLink ?Link.
 ?name dbpedia-owl:abstract ?abstract.
  } LIMIT 1000
  
  
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbpedia-owl:  <http://dbpedia.org/ontology/>
SELECT * 
FROM <http://core-i18n>
WHERE {?name rdfs:label  '컴투스' @ko.
?name dbpedia-owl:wikiPageWikiLink ?Link.
?name dbpedia-owl:abstract ?abstract.
 } LIMIT 3
```
```FROM <http://core-i18n>```이 부분이 디비(그래프)를 선택하는 쪽이기 때문에 꼭 필요함!   
(참고: https://www.programcreek.com/java-api-examples/?api=virtuoso.jena.driver.VirtGraph)
라이브러리 설치

https://github.com/srdc/virt-jena/tree/master/lib/virtuoso/virtjdbc4/4.0

```
1.2.1 Namespaces
In this document, examples assume the following namespace prefix bindings unless otherwise stated:

Prefix	IRI
rdf:	http://www.w3.org/1999/02/22-rdf-syntax-ns#
rdfs:	http://www.w3.org/2000/01/rdf-schema#
xsd:	http://www.w3.org/2001/XMLSchema#
fn:	http://www.w3.org/2005/xpath-functions#
```



