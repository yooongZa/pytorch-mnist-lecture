"""설명용으로 직접 정한 확률 세 가지를 비교합니다."""
import torch
from torch import nn

# 비교를 위해 직접 만든 확률입니다. 실제 모델의 출력이 아닙니다.
case_probs = torch.full((3, 10), 0.01)
case_probs[0] = 0.60 / 9
case_probs[0, 7] = 0.40
case_probs[1] = 0.10 / 9
case_probs[1, 7] = 0.90
case_probs[2, 3] = 0.91

# 위 확률을 재현하는 점수를 준비합니다.
case_logits = case_probs.log()
case_y = torch.tensor([7, 7, 7], dtype=torch.long)
case_losses = nn.CrossEntropyLoss(reduction="none")(case_logits, case_y)
case_mean_loss = nn.CrossEntropyLoss()(case_logits, case_y)

print("확률 합:", case_probs.sum(dim=1))
print("예측 클래스:", case_logits.argmax(dim=1))
print("개별 손실:", case_losses)
print("평균 손실:", round(case_mean_loss.item(), 4))
print("직접 계산:", -case_probs[:, 7].log())
