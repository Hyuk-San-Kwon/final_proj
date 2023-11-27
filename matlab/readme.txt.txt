데이터
- train : test = 7 : 3 분할
- label
	- 안녕하세요
	- 만나다
	- 반갑습니다


학습방법
- preprocess.m 실행
-> lstm 실행
-> model 실행
(preprocess & model의 cnn 모델을 일치시켜야함)


평가방법
- test : 이미지 한장 넣어서 결과 값 추론
- test_dir : test 폴더 전체에 대한 정확도 출력