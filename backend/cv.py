import cv2
import time
import os
from infer import infer_frame
from loguru import logger

def get_stream_video():
    # camera 정의
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    inputs = 0
    while True:
        # 카메라 값 불러오기
        success, frame = cap.read()
        if not success:
            logger.info("Cam Error Occured!")
            break
        else:
            inputs, outputs = infer_frame(frame, inputs)
            logger.info(outputs)
            ret, buffer = cv2.imencode('.jpg', frame)                       
            # frame을 byte로 변경 후 특정 식??으로 변환 후에
            # yield로 하나씩 넘겨준다.
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')

def save_video(file):
    
    cap = file.file
    current_time = time.localtime()
    vis_folder = './videos'
    save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    input_file_name = save_folder + '/input.avi'
    save_file_name = save_folder + '/output.avi'
    contents = cap.read()
    with open(os.path.join(save_folder, 'input.avi'), "wb") as fp:
        fp.write(contents)
        logger.info(file.filename)
    fp.close()
    
    return input_file_name
    
    
    
def infer(cap):
    return
    