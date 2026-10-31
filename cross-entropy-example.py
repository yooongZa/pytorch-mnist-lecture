# 이 Python 파일과 samples.json을 같은 폴더에 둡니다.
import json
from pathlib import Path
import torch
from torch import nn

samples = json.loads(Path(__file__).with_name("samples.json").read_text())["samples"]
batch_X = torch.tensor(
    [s["pixels"] for s in samples[:4]], dtype=torch.float32
).reshape(4, 1, 28, 28) / 255
batch_y = torch.tensor([s["label"] for s in samples[:4]], dtype=torch.long)
X, y = batch_X[:1], batch_y[:1]  # 실제 테스트 샘플 #0, 정답 7

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, images):
        flat = self.flatten(images)
        return self.layers(flat)

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)
model = DigitModel()  # PyTorch가 초기값을 준비한 모델

# X와 y는 같은 MNIST 샘플에서 가져온 값입니다.
with torch.no_grad():
    logits = model(X)
    loss = nn.CrossEntropyLoss()(logits, y)
print("점수 형태:", tuple(logits.shape))
print("예측 번호:", logits.argmax(dim=1).tolist())
print("데이터의 정답:", y.tolist())
print("손실 형태:", tuple(loss.shape))
print("손실 값:", round(loss.item(), 4))
