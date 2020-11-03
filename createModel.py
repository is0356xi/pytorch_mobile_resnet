import torch
import torchvision

# resnetモデルを利用する
model = torchvision.models.resnet18(pretrained=True)
# 推論モードにする
model.eval()

# サンプル入力を与える
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
# モデルを保存
traced_script_module.save("app/src/main/assets/resnet.pt")
