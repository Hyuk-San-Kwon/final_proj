import onnx
import numpy as np
import cv2
import time
from loguru import logger

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnxruntime


batch_size = 1
input_SequenceLength = 5 # 조정 가능
IMAGE_COUNT = 0


x = torch.randn(input_SequenceLength, batch_size, 3, 224, 224, requires_grad=True)
test_size = (224, 224)
# x = torch.randn(3, 224, 224, requires_grad=True)
onnx_model = onnx.load("KSLmodel_resnet50.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("KSLmodel_resnet50.onnx")

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def infer_frame(img, inputs):
    
    global IMAGE_COUNT
    device = 'cpu'
    img_info = {"id": 0}
    img_info["file_name"] = None
    
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    
    ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    img_info["ratio"] = ratio
    
    img = preproc(img, test_size)
    img = torch.from_numpy(img[0]).unsqueeze(0)
    
    img = img.float()
    outputs = None
    if IMAGE_COUNT == 0 :
        inputs = img.unsqueeze(0)
        IMAGE_COUNT += 1       
    
        
    if IMAGE_COUNT < 5:
        img = img.unsqueeze(0)
        inputs = torch.cat([inputs, img])
        IMAGE_COUNT += 1  

    
    elif IMAGE_COUNT == 5:
        if device == "gpu":
            inputs = inputs.cuda()
        
        t0 = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        outputs = ort_session.run(None, ort_inputs)   
        
        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        inputs = inputs[1:]
        img = img.unsqueeze(0)
        inputs = torch.cat([inputs, img])
        
    return inputs, outputs

