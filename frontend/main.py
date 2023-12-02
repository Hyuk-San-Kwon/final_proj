import cv2
import streamlit as st
import requests
from loguru import logger

st.title("Realtime Inference!")

websocket_url = "http://backend:8080/{backend}"

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    response = requests.get(websocket_url, stream=True)
    if response.status_code == 200:
        output = response.json()
        logger.info(output.get('output'))
        cv2.putText(frame, output.get('output'), (300, 450), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    FRAME_WINDOW.image(frame)
    