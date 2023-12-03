from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import File
from fastapi import UploadFile

from loguru import logger

from uuid import UUID, uuid4
import uvicorn
from loguru import logger

import cv2
from infer import Predictor
import time
import os

# cv2 모듈 import
from cv import get_stream_video, save_video, infer

INFER = False

# FastAPI객체 생성
app = FastAPI()



# openCV에서 이미지 불러오는 함수

def video_streaming():
    
    return get_stream_video()

# 스트리밍 경로를 /video 경로로 설정.
@app.get("/{backend}")
def main():
    # StringResponse함수를 return하고,
    # 인자로 OpenCV에서 가져온 "바이트"이미지와 type을 명시
    # return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")
    output = video_streaming()
    
    output = {'output' : 'test'}
    logger.info(output)
    return output 
    

@app.post("/{style}")
def get_video(style: str, file: UploadFile = File(...)):
    
    input_file_name = save_video(file)
    infer(input_file_name)
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)