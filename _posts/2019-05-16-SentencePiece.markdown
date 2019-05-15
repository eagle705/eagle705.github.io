---
layout: post
title:  "SentencePiece: A simple and language independent subword tokenizer
and detokenizer for Neural Text Processing"
subtitle:   "embedding"
categories: paper
tags: deeplearning
comments: true
use_math: true
---

BERT에서는 input을 WordPiece로 짤라서 주는데, 이걸 더 일반화 시킨 라이브러리가 SentencePiece다. 웹상에 사용법이 좀 부족하게 나와있는 것 같아서, 공부한거 정리할겸 남겨본다.

### SentencePiece
- 저자:Taku Kudo, John Richardson (Google, Inc)
- EMNLP 2018
- Official Repo: https://github.com/google/sentencepiece

### Who is an Author?

![](/assets/img/markdown-img-paste-20190516010154436.png)

#### 장점
- 언어에 상관없이 적용 가능
- OOV 대처 가능
- 적은 vocab size로 높은 성능기록

#### Install
- python module 설치
- tf에서 사용가능한 모듈이 따로 있음 (computational graph안에 tokenizer 포함됨)
  - 참고: https://github.com/google/sentencepiece/blob/master/tensorflow/README.md
```bash
pip install sentencepiece
pip install tf_sentencepiece
```

#### Usage
##### Training
- 전체적인 arg는 아래 그림 참조
![](/assets/img/markdown-img-paste-20190516012243358.png)
- input은 String이 아니라 문서 파일을 사용함
- vocab_size 때문에 에러가 날때가 있음, 실행할 때 에러메세지에서 적합한 vocab_size 알려주니 거기에 맞추면됨
- 아래와 같이 코드를 실행해주면 sentencepiece tokenizer가 학습이 됨
```python
import sentencepiece as spm
templates = '--input={} --model_prefix={} --vocab_size={} --control_symbols=[CLS],[MASK],[SEP] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'
vocab_size = 778
prefix = 'm'
input_file = './data_in/sentencepiece_train.txt'

cmd = templates.format(input_file, prefix, vocab_size)

spm.SentencePieceTrainer.Train(cmd)
```
- SentencePiece에서는 Custom token을 2가지로 나누는데, Control symbol과 User defined symbols임
  - Control symbol은 ```<s>, </s>```와 같은 텍스트를 인코딩하고 디코딩할때 사용하는 특수 토큰임
  - User defined symbol은 그냥 넣고 싶은거 넣는것임. 얘는 input text에 들어가면 나중에 extract할때 다른 것과 같이 하나의 piece로 인식됨
  - [문서 참고](https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md)
- 보통 Control symbol을 많이 쓰기 떄문에 추가해줘야함
- control symbol인 ```[CLS], [MASK], [SEP]``` 토큰을 추가해주기 위 ```--control_symbols``` 옵션을 사용함
- default control token으로 pad, bos, eos, unk 토큰등이 있음
  - pad 토큰의 경우 default 값은 비활성화라서 사전의 0번째 인덱스는 보통 ```<s>``` 토큰임
  - 우리는 pad 토큰도 쓸거기 때문에 활성화 시켜줘야하는데, 옵션값으로 id를 부여하면 활성화됨 ```--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3```


- 결과 화면
```
sentencepiece_trainer.cc(116) LOG(INFO) Running command: --input=./data_in/sentencepiece_train.txt --model_prefix=m --vocab_size=778 --control_symbols=[CLS],[MASK],[SEP] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3
sentencepiece_trainer.cc(49) LOG(INFO) Starts training with : 
TrainerSpec {
  input: ./data_in/sentencepiece_train.txt
  input_format: 
  model_prefix: m
  model_type: UNIGRAM
  vocab_size: 778
  self_test_sample_size: 0
  character_coverage: 0.9995
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 16
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  treat_whitespace_as_suffix: 0
  control_symbols: [CLS]
  control_symbols: [MASK]
  control_symbols: [SEP]
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 3
  bos_id: 1
  eos_id: 2
  pad_id: 0
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  ⁇ 
}
NormalizerSpec {
  name: nmt_nfkc
  add_dummy_prefix: 1
  remove_extra_whitespaces: 1
  escape_whitespaces: 1
  normalization_rule_tsv: 
}

trainer_interface.cc(267) LOG(INFO) Loading corpus: ./data_in/sentencepiece_train.txt
trainer_interface.cc(315) LOG(INFO) Loaded all 251 sentences
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: <pad>
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: <s>
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: </s>
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: <unk>
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: [CLS]
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: [MASK]
trainer_interface.cc(330) LOG(INFO) Adding meta_piece: [SEP]
trainer_interface.cc(335) LOG(INFO) Normalizing sentences...
trainer_interface.cc(384) LOG(INFO) all chars count=39749
trainer_interface.cc(392) LOG(INFO) Done: 99.9522% characters are covered.
trainer_interface.cc(402) LOG(INFO) Alphabet size=771
trainer_interface.cc(403) LOG(INFO) Final character coverage=0.999522
trainer_interface.cc(435) LOG(INFO) Done! preprocessed 251 sentences.
unigram_model_trainer.cc(129) LOG(INFO) Making suffix array...
unigram_model_trainer.cc(133) LOG(INFO) Extracting frequent sub strings...
unigram_model_trainer.cc(184) LOG(INFO) Initialized 5206 seed sentencepieces
trainer_interface.cc(441) LOG(INFO) Tokenizing input sentences with whitespace: 251
trainer_interface.cc(451) LOG(INFO) Done! 4681
unigram_model_trainer.cc(470) LOG(INFO) Using 4681 sentences for EM training
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=3116 obj=17.083 num_tokens=12289 num_tokens/piece=3.94384
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=2902 obj=15.6584 num_tokens=12336 num_tokens/piece=4.25086
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=2176 obj=16.7119 num_tokens=13290 num_tokens/piece=6.10754
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=2176 obj=16.4999 num_tokens=13299 num_tokens/piece=6.11167
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=1632 obj=18.2899 num_tokens=14896 num_tokens/piece=9.12745
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=1632 obj=17.9621 num_tokens=14925 num_tokens/piece=9.14522
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=1224 obj=19.8661 num_tokens=16922 num_tokens/piece=13.8252
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=1224 obj=19.5698 num_tokens=16937 num_tokens/piece=13.8374
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=918 obj=22.0437 num_tokens=19383 num_tokens/piece=21.1144
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=918 obj=21.6499 num_tokens=19435 num_tokens/piece=21.171
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=0 size=855 obj=22.3546 num_tokens=20096 num_tokens/piece=23.5041
unigram_model_trainer.cc(486) LOG(INFO) EM sub_iter=1 size=855 obj=22.2358 num_tokens=20096 num_tokens/piece=23.5041
trainer_interface.cc(507) LOG(INFO) Saving model: m.model
trainer_interface.cc(531) LOG(INFO) Saving vocabs: m.vocab
```

#### Load model & Encoding, Decoding
- 학습 후 위키피디아 본문의 일부를 SentencePiece로 tokenization 해봄
- default control symbol은 학습할때 넣어주었던 값대로 나옴
- SentencePiece에서는 default control symbol을 인코딩시에 text 앞뒤에 추가할 수 있는 옵션이 있음 
  - ```bos:eos``` 옵션은 문장에 ```<s> , </s>``` 토큰을 추가함
  - ```reverse```옵션은 순서를 거꾸로 만들어서 인코딩함
  - ```:``` 표시로 중첩해서 사용할 수 있음
  - BERT에서는 굳이 쓸 필요 없고, 따로 추가하는 작업을 하는게 맞을 듯  
  ```python
  extra_options = 'bos:eos' #'reverse:bos:eos'
  sp.SetEncodeExtraOptions(extra_options)
  ```
- SentencePiece tokenizer APIs (나머지는 문서 참조):
  - raw_text-to-enc_text: ```sp.EncodeAsPieces``` 
  - raw_text-to-enc_id: ```sp.EncodeAsIds``` 
  - enc_text-to-raw_text: ```sp.decode_pieces```
  - enc_id-to-enc_text: ```sp.IdToPiece```

- 코드
```python
# Load model
sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format(prefix))

print(sp.pad_id()) # 결과: 0
print(sp.bos_id()) # 결과: 1
print(sp.eos_id()) # 결과: 2
print(sp.unk_id()) # 결과: 3

training_corpus = """
초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다. 
그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다. 
인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다. 
"""

training_corpus = training_corpus.replace("\n", '').split('.')[:-1] # 개행문자제거, 문장 분리
training_corpus = [_.strip() for _ in training_corpus] # 문장 앞 뒤의 불필요한 공백 제거

# 사실상 extra_options은 쓰지 않아도됨, 각자 추가해야할 듯
extra_options = 'bos:eos' #'reverse:bos:eos'
sp.SetEncodeExtraOptions(extra_options)

training_ids = []
for sent in training_corpus:
    encode_piece = sp.EncodeAsPieces(sent)
    training_ids.append(sp.EncodeAsIds(sent))
    print("raw text: ", sent)
    print("enc text: ", encode_piece)
    print("dec text: ", sp.decode_pieces(encode_piece))
    print("enc ids: ", sp.EncodeAsIds(sent))
    print("")

# 사전 구성을 확인하자
for i in range(10):
    print(str(i)+": "+sp.IdToPiece(i))
```

- 결과
```
raw text:  초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다
enc text:  ['<s>', '▁', '초', '기', '▁', '인', '공', '지', '능', '▁', '연', '구', '에', '▁', '대', '한', '▁', '대', '표', '적', '인', '▁', '정', '의', '는', '▁', '다', '트', '머', '스', '▁', '회', '의', '에', '서', '▁', '존', '▁', '매', '카', '시', '가', '▁', '제', '안', '한', '▁', '것', '으', '로', '▁', '"', '기', '계', '를', '▁', '인', '간', '▁', '행', '동', '의', '▁', '지', '식', '에', '서', '와', '▁', '같', '이', '▁', '행', '동', '하', '게', '▁', '만', '드', '는', '▁', '것', '"', '이', '다', '</s>']
dec text:  초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다
enc ids:  [1, 7, 656, 22, 7, 43, 669, 30, 776, 7, 668, 81, 14, 7, 88, 16, 7, 88, 218, 54, 43, 7, 53, 9, 10, 7, 20, 82, 294, 41, 7, 290, 9, 14, 60, 7, 199, 7, 160, 637, 51, 19, 7, 123, 181, 16, 7, 667, 202, 25, 7, 77, 22, 101, 17, 7, 43, 247, 7, 139, 119, 9, 7, 30, 176, 14, 60, 50, 7, 337, 11, 7, 139, 119, 38, 91, 7, 65, 125, 10, 7, 667, 77, 11, 20, 2]

raw text:  그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다
enc text:  ['<s>', '▁', '그', '러', '나', '▁', '이', '▁', '정', '의', '는', '▁', '범', '용', '인', '공', '지', '능', '(', 'A', 'G', 'I', ',', '▁', '강', '한', '▁', '인', '공', '지', '능', ')', '에', '▁', '대', '한', '▁', '고', '려', '를', '▁', '하', '지', '▁', '못', '한', '▁', '것', '▁', '같', '다', '</s>']
dec text:  그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다
enc ids:  [1, 7, 252, 120, 46, 7, 11, 7, 53, 9, 10, 7, 377, 99, 43, 669, 30, 776, 24, 74, 183, 168, 15, 7, 357, 16, 7, 43, 669, 30, 776, 23, 14, 7, 88, 16, 7, 28, 145, 17, 7, 38, 30, 7, 634, 16, 7, 667, 7, 337, 20, 2]

raw text:  인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다
enc text:  ['<s>', '▁', '인', '공', '지', '능', '의', '▁', '또', '다', '른', '▁', '정', '의', '는', '▁', '인', '공', '적', '인', '▁', '장', '치', '들', '이', '▁', '가', '지', '는', '▁', '지', '능', '이', '다', '</s>']
dec text:  인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다
enc ids:  [1, 7, 43, 669, 30, 776, 9, 7, 116, 20, 439, 7, 53, 9, 10, 7, 43, 669, 54, 43, 7, 89, 208, 36, 11, 7, 19, 30, 10, 7, 30, 776, 11, 20, 2]

0: <pad>
1: <s>
2: </s>
3: <unk>
4: [CLS]
5: [MASK]
6: [SEP]
7: ▁
8: .
9: 의
```

#### 전체 코드
```python
import sentencepiece as spm
templates = '--input={} --model_prefix={} --vocab_size={} --control_symbols=[CLS],[MASK],[SEP] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'
vocab_size = 778
prefix = 'm'
input_file = './data_in/sentencepiece_train.txt'

cmd = templates.format(input_file, prefix, vocab_size)

spm.SentencePieceTrainer.Train(cmd)
# Load model
sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format(prefix))

print(sp.pad_id()) # 결과: 0
print(sp.bos_id()) # 결과: 1
print(sp.eos_id()) # 결과: 2
print(sp.unk_id()) # 결과: 3

training_corpus = """
초기 인공지능 연구에 대한 대표적인 정의는 다트머스 회의에서 존 매카시가 제안한 것으로 "기계를 인간 행동의 지식에서와 같이 행동하게 만드는 것"이다. 
그러나 이 정의는 범용인공지능(AGI, 강한 인공지능)에 대한 고려를 하지 못한 것 같다. 
인공지능의 또다른 정의는 인공적인 장치들이 가지는 지능이다. 
"""

training_corpus = training_corpus.replace("\n", '').split('.')[:-1] # 개행문자제거, 문장 분리
training_corpus = [_.strip() for _ in training_corpus] # 문장 앞 뒤의 불필요한 공백 제거

# 사실상 extra_options은 쓰지 않아도됨, 각자 추가해야할 듯
extra_options = 'bos:eos' #'reverse:bos:eos'
sp.SetEncodeExtraOptions(extra_options)

training_ids = []
for sent in training_corpus:
    encode_piece = sp.EncodeAsPieces(sent)
    training_ids.append(sp.EncodeAsIds(sent))
    print("raw text: ", sent)
    print("enc text: ", encode_piece)
    print("dec text: ", sp.decode_pieces(encode_piece))
    print("enc ids: ", sp.EncodeAsIds(sent))
    print("")

# 사전 구성을 확인하자
for i in range(10):
    print(str(i)+": "+sp.IdToPiece(i))

```

#### Reference
- https://lovit.github.io/nlp/2018/04/02/wpm/
- https://github.com/google/sentencepiece#redefine-special-meta-tokens
- https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
