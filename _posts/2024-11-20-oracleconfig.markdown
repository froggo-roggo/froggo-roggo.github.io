---
layout: post
title: Oracle Free Tier 설정, 파이썬 업데이트
date: 2024-11-20
category: Hobby
---

# 세줄요약
1. Free tier 내에서 보장하는 범위 내면 요금이 안 나가니 예상 금액 SGD 2.76 청구된다고 떠도 쫄지 않기
2. yum으로 뭘 까는 것은 메모리 1GB로는 어림도 없으니 swap 8GB 이상 설정하기
3. 인스턴스 설정하자마자 시간 설정하기... 새벽에 알림 보내고 싶지 않으면

## 1. Oracle Clound 가입 후 free tier 사용하며 생긴 일
- Free tier 가입 방법은 현재(24년 11월) 시점에서 [이 글](https://velog.io/@kisuk623/%EC%98%A4%EB%9D%BC%ED%81%B4-%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C-%ED%94%84%EB%A6%AC%ED%8B%B0%EC%96%B4-%EA%B0%80%EC%9E%85%ED%95%98%EA%B8%B0)이 참고할 만한듯
	- 가입 지역 춘천으로 하지 말자... 자리가 없어 ARM 머신 생성이 안 된다. 단 다른 지역도 된다는 보장은 없다.
- 무료 계정이라도 카드를 등록해야 한다. SGD 1달러 (한국 돈 약 천원) 정도가 카드로 빠져나갔다가 돌아오는데 쫄지 말자. 그냥 카드 확인 절차이다.
- 현재 시점 오라클은 VM 2개, 블록 볼륨 스토리지 2개 합산 총 200GB 등등을 지원한다. *이 범위 내에서라면 금액이 청구되지 않는다.*
	- 따라서 새 인스턴스를 만들 때, 우측 하단에 2.76 달러가 청구된다고 떠도 두려워하진 말자. 제한 내에서만 쓰면 청구되지 않는다.

## 2. 메모리 1GB는 너무 적다!!
- 심지어 oracle linux의 기본 설치 명령어는 ```yum```인데, 메모리가 부족해서 ```yum```을 못 쓴다. Search도 안되고 install도 안 된다. 허허...
- 혹시 자신이 oracle linux에서 ```sudo yum install something``` 을 시도했다가 멈추고 process killed 되었다면 다음과 같은 절차를 따르자.
	- 우선 ```sudo yum clean all``` 해서 캐시를 전부 지운다.
	- ```free -h``` 해서 메모리를 확인한다.
		- Mem과 Swap이 뜨는데, Swap이 0B로 나타나면 스왑으로 사용중인 공간이 없다는 뜻이다. 새로 설정하면 된다.
		- 만약 Swap이 1.98GB 등 0이 아닌 값으로 나타나면 기존 스왑 용량이 있다는 뜻이다. (기본값은 0 또는 2GB 둘 중 하나인 것 같다.)
	- ```df -h``` 해서 가용 디스크 공간을 확인한다. 8GB를 스왑할 것이기 때문에 8GB보단 훨씬 많은 디스크 공간이 남아있어야 한다.
	- 만약 기존 스왑이 이미 존재한다면
		- ```sudo swapon --show``` 로 스왑 경로를 확인 (기본값은 ```/.swapfile``` 이다.)
		- ```sudo swapoff /.swapfile``` 로 기존 스왑 비활성화
		- 다음 명령어 차례대로 실행하여 스왑 공간 변경
			- ```sudo fallocate -l 8G /.swapfile```
				- 8G가 아닌 값을 스왑하려면 숫자만 바꿔주면 된다.
			- ```sudo chmod 600 /.swapfile```
			- ```sudo mkswap /.swapfile```
		- ```sudo swapon /.swapfile``` 실행하여 스왑 공간 활성화
	- 만약 기존 스왑이 없다면
		- 위 단계에서 ```fallocate```부터 ```swapon```까지만 실행하면 됨
- 이후 ```sudo yum search```든 ```install```이든 실행하면 잘 된다. 적어도 나는 그랬다.
	- -y 설정 걸어놓고 nohup &으로 백그라운드 실행시키면 메모리 사용량이 6~7GB까지 올라가는것을 확인할 수 있다. 이러니까 컴퓨터가 멈추지...
	- 나는 파이썬 3.12를 설치하고 싶었다. Oracle Linux 8에서 기본 제공하는 파이썬은 3.6이라서... 천만다행으로 yum에서 파이썬 3.12가 찾아지기는 한다. 이걸 감사해야 하나...

## 3. 파이썬 설치 후 명령어 통합하기
- 기본값 설정으로 python과 python3 모두 3.6을 가리키게 되어 있다. 그리고 pip3만 깔려 있다. 불편하다.
	- ```which python3``` 그리고 ```which python``` 으로 각 명령어가 가리키는 실행 파일 위치를 찾아 기억한다. 기본값은 ```/user/bin/python3``` 이다.
	- ```sudo alternatives --config python3``` 그리고 ```sudo alternatives --config python``` 을 이용해 각각 python 3.12가 선택되도록 설정을 바꾼다.
		```
		[opc@instance-name ~]$ sudo alternatives --config python3

		There are 2 programs which provide 'python3'.

		  Selection    Command
		-----------------------------------------------
		*+ 1           /usr/bin/python3.6
		   2           /usr/bin/python3.12

		Enter to keep the current selection[+], or type selection number: 2
		```
	- 이후 python 3.12의 ensurepip 모듈을 활용해 pip를 설치한다.
		- ```python3.12 -m ensurepip --upgrade```

## 4. 시간대 바꾸기
- ```timedatectl list-timezones | grep Asia/Seoul``` 로 timezone 목록 중 한국이 있는지 확인한다.
- 있으면 ```sudo timedatectl set-timezone Asia/Seoul``` 한다.