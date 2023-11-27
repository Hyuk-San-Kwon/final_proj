import onnx
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnxruntime


batch_size = 1
input_SequenceLength = 5 # 조정 가능
x = torch.randn(input_SequenceLength, batch_size, 3, 224, 224, requires_grad=True)
# x = torch.randn(3, 224, 224, requires_grad=True)
onnx_model = onnx.load("KSLmodel_resnet50.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("KSLmodel_resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

