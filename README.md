# PyTorch MNIST 예측과 손실

[전체 강의](https://yooongza.github.io/pytorch-mnist-lecture/) · [손실 보충 자료](https://yooongza.github.io/pytorch-mnist-lecture/cross-entropy.html)

자습노트의 흐름에 맞춘 20페이지·50분 강의와 보충 5페이지입니다. 슬라이드는 1280×720 가로 고정형이며, 페이지 번호나 ← → 키로 이동합니다. N은 발표 노트, G는 목차, F는 전체 화면입니다.

실제 MNIST 테스트 #0(정답 7)과 시드 42로 초기화한 같은 모델을 사용합니다. 입력 → Flatten → Linear·ReLU → 점수 → 예측·확률 → 손실을 따라갑니다. 손계산 대신 출력의 의미와 단계 간 연결을 설명합니다. 마지막에는 같은 모델에 실제 이미지 네 장을 넣어 개별·평균 손실을 봅니다.

- 공개 페이지의 코드 버튼: 저장된 실제 PyTorch 실행 결과를 표시합니다.
- 그래프·값 비교·퀴즈: 브라우저에서 조작할 수 있습니다.
- 직접 실행: `cross-entropy-example.py`와 `samples.json`을 같은 폴더에 저장한 뒤 PyTorch 환경에서 Python 파일을 실행합니다.
- 모델: `784 → 512 → 512 → 10`, CPU, 초기 가중치로 계산. 손실 2.2641, 네 장 평균 손실 2.2993.

주교재: [PyTorch 신경망 모델 구성하기](https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html), [손실 함수와 최적화](https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html).
