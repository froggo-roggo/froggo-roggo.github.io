---
layout: post
title: 연구하면서,, 몰랏던 것들,,
date: 2025-11-26
category: Amumal
use_math: true
---

내가연?구?연구자? 그냥 말하는감자갓은디

1. PyTorch에서 LBFGS 옵티마이저는 아직 stable하지 않아 쓰는 것이 권장되지 않는다.
  * https://arxiv.org/pdf/2305.11930
  * 관련해서 참조한 discussion: https://github.com/pytorch/pytorch/issues/5953
  * BFGS, 또는 LBFGS 방식을 쓰고싶다면 기본적으로 scipy를 쓰자 (빠르다)
2. 근데 numpy랑 scipy는 또 multiprocessing 지원이 미흡하다
  * epoch나 event를 쪼개서,,, 병렬화하고 싶으면,,, 내가 multiprocessing 모듈 써서 알아서 쪼개고 프로세스별로 나눠줘야 함
  * torch는 이게 알아서 된다더라?? 어케한거임
3. scipy에서 사전 정의된 각종 분포 함수들은 부르는게 느리다 (overhead가 크다)
  * Normal, TruncNormal 같은 간단한 것들은, 만약 fitting 과정에서 자주 호출해야 한다면, 걍 custom 함수를 만들어 쓰자
  * 또 웬만하면 근사하자 (approximate)
  * 값의 하한선 (epsilon) 설정하는거 잊지 말기
    * 이 하한선이 또 너무 극단적이면 overflow/underflow가 생기니까... 적당히 잡기
  * 확률적 모델링 하다보니 Pyro 만든 사람이 왜 pyro 만들었는지 대충 감은 잡힘,,,
    * 근데 꼭 Pyro를 써야만 Bayesian Inference가 가능한건 아님!! 나같경 Pyro document 읽는시간보다 걍 내가 짜는게 빨랐는데 (사유: 2년내로 졸업해야 함) 프로젝트가 커질수록 이미 만들어진 바퀴 쓰는게 나을거같음,,
  * 베이즈가죽었으면좋겠음 (이미죽음, 260년전에)
4. 기껏 multiprocessing, async 쓰고 나서 직렬/순서대로 처리해야 하는 or 병목을 유발하는 작업을 코드 안에 넣지 말자
  * 디버깅할때 print 쓰지 말고 로깅을하자고제발
  * 근데 print어케참음...
5. $$P(C(t) \mid C(t-1))$$ 과 $$P(C(t))$$를 구분하자
  * $$C(t)$$는 시점 t에서 어떤 조건 C가 만족되거나 만족되지 않는 사건
  * 아진짜마르코프가너무싫음죽었으면좋겠음(이미죽음, 100년전에)
6. 교수님 말씀을 잘 듣자
7. 확률밀도함수의 누적값으로부터 특정 시점 i에서 어떤 사건이 일어날 확률을 구했다면 이건 밀도가 아니라 확률질량이다
  * 그리고 아무리 연속적인것처럼 보이고 아무리 길어보여도 니가 데이터를 저장한 거기는 결국 array다
  * 이산적으로 대하고 이산적으로 더해야 함
  * 전체 시간 구간에서 다 더하면 1에 수렴해야 함
  * 이걸 어떻게 헷갈리는지 나도 지금 생각해보면 어이가 없은,, 근데 졸렸나봄
8. 잠을 잘 자고 밥을 거르지 말자, 배가 안 고픈 것 같아도 꼭 정시에 먹자
