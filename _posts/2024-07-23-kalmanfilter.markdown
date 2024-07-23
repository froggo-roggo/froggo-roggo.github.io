---
layout: post
title: Kalman Filter
date: 2024-05-03
category: Probabilistic-Modelling
use_math: true
---

# Kalman Filter (KF)란 무엇인가?

[많이 참조한 글,,,](https://alida.tistory.com/54)

## 1. 필터링이란 무엇인가?
- 으아아아아아아아아아악

### 1.1. 칼만 필터는 "현재 상태의 믿을만한 정도(belief)가 가우시안 분포로 나타나는 베이지안 필터"다. 이게 대체 무슨 소리임?
- 우선 '베이지안'과 '믿을만한 정도'라는 단어부터 짚고 넘어가자.
- 베이즈 정리는 [여기](/probabilistic-modelling/2024/05/03/svi.html)에서도 언급했듯 사전 확률과 가능도를 결합하면 사후 확률이 나온다는 정리다.
  - 여기서 사전 확률은 가설 내지는 이미 알고 있던 정보(배경 지식, 상식, 기타등등...)를 나타내는 분포이다. (기상 관측 인공위성이 지금 위치한 곳을 열대지방이라고 가정하자.)
  - 가능도는 특정한 가설에 기반했을 때 어떤 사건이 발생할 가능성을 나타낸다. (위성이 위치한 곳이 열대지방이라면, 여기 눈이 올 확률은 얼마인가?)
  - 사후 확률은 특정한 데이터에 대해서 어떤 가설이 맞을 가능성을 나타낸다. (인공위성의 탐지기에 눈이 오는 것이 관측되었다면, 과연 위성이 있는 곳을 열대지방이라고 말할 수 있는가?)
    - 데이터를 잘 설명하지 못하는 가설일수록 이 확률은 매우 낮을 것이며, 우리가 위성의 위치에 대해 가진 가설을 재고해야 함을 나타낼 것이다.
- 이러한 베이즈 정리가 유추에 적극적으로 활용되는 모델 중 하나로는 **은닉 마르코프 모델(hidden Markov model, HMM)**이 있다.
  - 참조: [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model)
    - 어떤 시스템을 크게 "관측이 불가능한 부분 (state)"과 "관측이 가능한 부분 (observation)"으로 나눈다고 생각하면 편하다.
	- 기본적인 가정은 **관측할 수 없는 state에 의해 observable data가 방출(emission)된다**는 것이다.
	- 이러한 state가 시간에 따라 변하는(transition) 것까지 고려한다면 마르코프 연쇄(Markov chain, MC)가 된다.
  - 실제 상황에서는 대부분 observable data로부터 state를 유추해야 한다. 따라서 자연히 베이즈 정리가 쓰이게 된다!
  - 이 때 **믿을만한 정도(belief)**는 관측 가능한 것들로부터 추론한 상태 x가 진짜 상태 x일 가능성을 의미한다. (세상에 믿을 거 하나 없다는 사실을 다시금 되새기게 된다.)
  - 더 구체적으로는, 시간 0부터 t까지 측정한 데이터 z가 있을 때 t에서의 상태 x에 대한 belief를 $$ p(x_t \mid z_{0:t}) $$로 정의한다.

### 1.2. 이 베이즈 정리와 필터링은 대체 무슨 상관이 있을까?
- 이 맥락에서 필터(filter)가 정확히 뭘 뜻하는 것일까? 
  - 신호나 시스템 관련 수업에서 주파수 대역 필터링 같은 건 많이 들어봤는데... 이것과는 좀 다른 것 같다. 물론 HEPA 필터와 같은 물리적 필터도 아니다.
  - 사진이나 음성을 보정할 때 쓰는 노이즈 필터는 관련이 있을 수도 있겠다. 여기에서 필터링은 내가 "관측값을 믿지 못하겠을 때" 거치는 과정을 의미한다.
    - 이를테면 다음 그림과 같이 네비게이션이 GPS 신호를 통해 차량의 실시간 위치를 관측해서 화면 상에 표시하고 있다고 생각해보자.
	
	<center><img src="/assets/img/kalmanfilter_1.png" width="auto" height="400px"/></center>
	
  	- 연속된 시간 t=1에서 4까지는 GPS로 관측된 신호가 그럭저럭 멀쩡한 것으로 보인다. 그러나 t=5에서는 차가 갑자기 아파트 한가운데에 있는 것으로 나타난다.
	- 차가 하늘을 날아서 실제로 저 위치까지 갔을까? 아니면 단순히 GPS 신호의 일시적 오류(또는 잡음, noise)일 뿐일까? 당연히 후자다.
	- 그런데 당연히 후자가 맞을 것이라는 확신은 어디에서 나올까? 바로 이전까지 관측된 정보(z)에서 나온다. (물론 우리의 상식(h)도 작용했을 것이다)
  - 이처럼 현 시점의 관측값을 곧이곧대로 "믿지 못하는" 경우는 생각보다 자주 발생하기 때문에, 이미 **관측된 데이터임에도 다시금 추정**을 해야 하는 상황이 발생할 수 있다.
    - 그리고 위에서도 언급했듯 <u>어떤 데이터에 노이즈가 끼어있다는 확신(거꾸로 말하자면 새로 얻은 데이터에 대한 낮은 belief)은 이전까지의 정보를 통해서 얻어지기 때문에 베이즈 추론과 관련이 있는 것</u>이다.
- 참고로, filtering 이외에도 관측 데이터에 대해서 [다양한 추정 문제](https://alida.tistory.com/54)가 존재한다. (Smoothing, prediction, interpolation 등)
  - 어차피 다 나중에 알아야 할 것들이라서 메모해둠,,,

### 1.3. 그럼 베이지안 필터란 정확히 무엇인가?
- 

## 2. Kalman Filter

## 3. Particle Filter

## 4. 논문 읽음