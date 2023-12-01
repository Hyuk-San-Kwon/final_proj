import onnx
import numpy as np
import cv2
import time
from loguru import logger

from torch import nn
import torch.onnx
import onnxruntime


batch_size = 1
input_SequenceLength = 5 # 조정 가능
IMAGE_COUNT = 0
time_means = 0
cnt = 0


cuda_providers = [
    "CUDAExecutionProvider"
]
tensorrt_providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 1,
    }),
]
tensorrt_fp16_providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 1,
        'trt_fp16_enable': True,
    }),
]
# onnx.checker.check_model(onnx_model)

class Predictor(object):
    def __init__(
        self,
        device = 'cpu',
        fp16 = False
    ):
        self.device = device
        self.fp16 = fp16
        self.providers = {"CUDA": cuda_providers, "TensorRT": tensorrt_providers, "TensorRT-fp16": tensorrt_fp16_providers}
        self.test_size = (224, 224)
        onnx_model = onnx.load("KSLmodel_resnet50.onnx")
        self.key = "TensorRT-fp16"
        self.model = onnxruntime.InferenceSession("KSLmodel_resnet50.onnx", providers=self.providers[self.key])
        

    def preproc(self, img, input_size, swap=(2, 0, 1)):
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

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def infer_frame(self, img):
        
        global time_means, cnt
  
        img_info = {"id": 0}
        img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        
        img = self.preproc(img, self.test_size)
        img = torch.from_numpy(img[0]).unsqueeze(0)
        
        img = img.float().cuda()
        outputs = None
        img = img.unsqueeze(0)
        ort_inputs = {self.model.get_inputs()[0].name: self.to_numpy(img)}
        
        t0 = time.time()
        outputs = self.model.run(None, ort_inputs)
        infer_time = time.time() - t0
        logger.info("Infer time: {:.4f}s".format(infer_time))
        time_means += infer_time
        cnt += 1
        logger.info("Infer mean time : {:.4f}s".format(time_means / cnt))
    
        return outputs
        
    