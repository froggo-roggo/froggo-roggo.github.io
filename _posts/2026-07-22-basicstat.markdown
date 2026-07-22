---
layout: post
title: 각종 분포와 검정 (t검정, F검정 등)
date: 2026-07-22
category: Probabilistic-Modelling
use_math: true
---

# 아주 기초적인 통계 지식들...
- 확률분포(Probabilistic distribution)는 확률변수(Random variable)와 함께 존재한다.

## 1. Random Variable의 종류

### 1-1. 이산확률변수 (Discrete Random Variables)
- Bernoulli RV
- Binomial RV (중요)
- Poisson RV (중요)
- Hypergeometric RV
- (TODO)

### 1-2. 연속확률변수 (Continuous Random Variables)
- Uniform RV
- Normal RV (매우 중요)
- Exponential RV (중요)
- Pareto RV
- Gamma RV (중요)
- (TODO)

## 2. Sampling distributions
- 카이제곱분포 (Chi-square distribution)
- t분포
- F분포
- (TODO)

## 3. 데이터가 있고, 어떤 분포일 것이라고 가정했을 때, 그 분포의 모수(Parameter)를 알고 싶다면? Parameter estimation
- 모수(Parameter)는 모집단(Population) 또는 모분포(Population distribution)와는 전혀 다른 개념임
    - 모집단이나 모분포의 특성을 구체적으로 설명해 주는 고정된 수치(값)
- 근데 가끔 모집단이 들어가야 할 자리에 모수를 쓰는 경우가 있음... '모집단의 원소(element) 수'로 착각하는듯. '모집단의 특성 수치'를 줄여서 모수임

### 3-1. Maximum likelihood estimation (MLE)
- 전체적으로는 [예전 글](/probabilistic-modelling/2024/05/03/svi.html)을 참조
- 주의할 점: Estimator와 Estimate은 다른 개념
    - Estimator: 그냥 random variable. 추정자.
	- Estimate: Random variable을 sample하면 나오는 data. 추정치.
- 어떤 확률변수 $$X_i$$에 대한 데이터 $$(x_1, x_2, ... , x_n)$$이 주어졌을 때, 확률변수 $$X_i$$의 분포 종류 $$f$$를 알고 있다면(또는 가정했다면), $$f$$의 매개변수 $$\theta$$를 알아내는 과정이 바로 MLE
    - 이때 $$f(x_1, x_2, ..., x_n \mid \theta) = f_{X_1}(x_1) * f_{X_2}(x_2) * ... * f_{X_n}(x_n)$$
		- 모든 양의 정수 i에 대해 X_i가 서로 독립이라 가정하기 때문에 우변처럼 표현 가능
		- independent and identically distributed (i.i.d.)
	- 여기서 MLE: 이 독립사건들의 확률곱을 최대화하는 $$\theta$$의 값 또는 식
		- 값을 구하면 점추정(Point estimation), 범위를 구하면 구간추정(Interval estimation)

### 3-2. 다양한 분포들의 estimator
- Bernoulli
- Poisson
- Uniform
- Normal
- Lognormal
- (TODO)

### 3-3. 모르는 정보가 있을 때의 estimation: Interval estimation
- 데이터가 정규분포를 따를 것이라고 가정하는 경우가 많으므로, 아래 interval estimation의 예시들도 대부분 normal distribution의 variant
- 모평균은 모르고, 모분산은 알 때 ($$\mu$$ unknown, $$\sigma^2$$ known)
    - 표본평균의 분포는 표준정규분포
- 모분산을 모를 때
    - 표본평균의 분포는 t분포
	- 표본분산의 분포는 카이제곱분포
- 평균을 모르는 두 집단을 비교할 때
    - 모분산을 알 때 표본평균의 비교: 표준정규분포
	- 모분산을 모르지만 같다는 걸 알 때 표본평균의 비교: t분포 (이때 각 표본분산의 분포가 카이제곱분포)
	- 모분산을 모르는데 다를 때 표본평균의 비교: Welch's t-test
	- 모분산을 모르는 상태에서 두 표본분산의 비교: F분포
- 정규분포 이외의 interval estimation은 다음과 같다.
	- Bernoulli
	- Exponential
	- (TODO)

### 3-4. Point estimatior를 평가하는 법
- Estimation에는 당연히 오차가 따른다.
- n차원 확률변수 벡터 $$\mathbf{X} = (X_1, X_2, ... , X_n)$$와 parameter $$\theta$$에 대해, 
- point estimator $$d := d(\mathbf{X})$$가 $$\theta$$의 estimator로 충분한지 평가하려면 어떻게 하면 좋을까?
    - **평균제곱오차(Mean square error, MSE)** $$r(d, \theta) := E \left[ (d(\mathbf{X}) - \theta)^2 \right]$$
	    - 당연하지만 작을수록 좋음
	- 편향(Bias) $$b_{\theta}(d) := E \left[ d(\mathbf{X}) - \theta \right]$$
	    - 만약 모든 parameter $$\theta$$에 대해 이 값이 0이면, d를 unbiased estimator라고 한다.
    - 분산에 대한 식을 응용해서 일반화하면, $$r(d, \theta) = Var(d) + (b_{\theta}(d))^2$$ 임을 알 수 있다.

### 3-5. Bayes Estimator
- parameter $$\theta$$는 모르지만, prior distribution $$p(\theta)$$는 알고 있을 때
- (TODO)

## 4. 가설 검정 (Hypothesis Testing)
- 영가설, 귀무가설 (Null hypothesis) $$H_0$$: 아무런 차이가 없다는 가설. 실험이나 데이터를 통해 기각하고 싶은 가설.
- 대립가설 (Alternative hypothesis) $$H_1$$: 차이가 있다는 가설. 실험이나 데이터를 통해 증명하고 싶은 가설.
- 기각역 (Critical region): 확률변수 $$X_i$$의 값이 $$C$$에 포함될 경우, 귀무가설을 기각하는 범위 $$C$$. 달리 말해, 데이터가 $$C$$ 구간에서 나타나면 대립가설이 채택되고 연구 성공!

### 4-1. 평균의 검정 (z 검정, t 검정)
- 위 파트 3-3을 참고해보자.
- **모분산을 알고 있다면, 정규분포를 사용하면 된다**. 이를 **z 검정(z-test)**라고 한다.
    - 모분산을 알고 있으면, 표본분산을 추정 안 해도 되니까 자유도가 줄어들지 않는다.
    - 추정할 평균 $$\mu$$, 추정치 $$\overline{X}$$, 기준 평균 $$\mu_0$$라 하자.
	- 검정해야 할 식은 $$ \frac{\overline{X}-\mu_0}{\frac{\sigma}{\sqrt{n}}} \sim Z_0 $$
    - $$H_0$$: $$\mu = \mu_0$$ (별 차이가 없을 것이다)
	- $$H_1$$: $$\mu \ne \mu_0$$ (유의한 차이가 있을 것이다)
	- $$ \lVert Z_0 \rVert = \lVert \frac{\overline{X}-\mu_0}{\frac{\sigma}{\sqrt{n}}} \rVert > Z_{\frac{\alpha}{2}}$$일 경우 $$H_0$$를 기각한다.
	    - 왜 $$Z_{\frac{\alpha}{2}}$$인가? 양쪽 꼬리(유의하게 큰 경우와 유의하게 작은 경우)를 모두 합쳐서 $$\alpha$$가 되어야 하기 때문
- 단측 검정을 하려면, 반쪽 분포를 사용한다.
    - $$H_0$$: $$\mu = \mu_0$$ (별 차이가 없을 것이다)
	- $$H_1$$: $$\mu > \mu_0$$ (유의하게 클 것이다)
	    - 클 경우 오른쪽 분포(right tail), 작을 경우 왼쪽 분포(left tail)를 쓴다.
	    - $$ Z_0 = \frac{\overline{X}-\mu_0}{\frac{\sigma}{\sqrt{n}}} > Z_{\alpha}$$일 경우 $$H_0$$를 기각한다.
		- 따라서 단측 검정의 유의 영역은 양측 검정의 유의 영역의 한쪽 꼬리보다 약간 더 넓어지게 된다.
- **모분산을 모를 때 표본평균의 값을 검정하려 한다면, t분포를 사용해야 한다**. 이를 **t 검정(t-test)**이라고 한다.
    - 모평균도 모분산도 모르면, 표본분산은 표본평균을 추정한 후 그 표본평균을 이용해서 추정해야 한다. 이 과정에서 자유도가 하나 줄어든다.
	- 식은 $$\frac{\overline{X} - \mu_0}{\frac{s}{\sqrt{n}}} \sim t_{n-1}$$
	- 마찬가지로, $$\lVert \frac{\overline{X} - \mu_0}{\frac{s}{\sqrt{n}}} \rVert > t_{\frac{\alpha}{2}, n-1}$$ 면 $$H_0$$를 기각한다.
- **모분산을 모를 때 두 집단의 표본평균이 유의하게 다른지 검정하려 할 때도, t분포를 사용해야 한다**.
    - 두 집단의 모분산이 같다고 가정하면, **Student's t-test**
	- 두 집단의 모분산이 다르다고 가정하면, **Welch's t-test** (좀 더 일반적)
    - (TODO)
- 위 검정들도 마찬가지로, 단측 검정을 하고 싶다면 방향에 따라서 반쪽 꼬리만 쓰면 된다.

### 4-2. 분산의 검정 (카이제곱 검정, F 검정)
- 표본평균에 대해서는 정규분포와 t분포를 썼다면, 표본분산에 대해서는 카이제곱분포와 F분포를 쓴다고 기억하면 편하다. 분산을 검정하므로 당연히 모분산을 모르는 상태이다.
- 하나의 표본분산을 검정할 때, 카이제곱분포를 쓴다.
- 두 표본분산을 비교할 때, F분포를 쓴다.
- (TODO)

### 4-3. 정규분포 이외의 통계량의 검정
- 하나의 Bernoulli parameter
- 두 집단의 Bernoulli parameter: **Fisher-Irwin test**로, Hypergeometric distribution을 쓴다.
- 하나의 Poisson parameter
- 두 집단의 Poisson parameter: Bernoulli 분포를 쓴다.
- (TODO)