import eventlet
eventlet.monkey_patch()

import base64
import io
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

@socketio.on('frame')
def handle_frame(data):
    print("[RECEIVED] Frame from client")

    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    _, buffer = cv2.imencode('.jpg', frame)
    processed_base64 = base64.b64encode(buffer).decode('utf-8')

    emit("processed_frame", {"image": processed_base64})
    print("[SENT] Processed frame to client")

@socketio.on('connect')
def handle_connect():
    print("[CONNECTED] Client connected")

if __name__ == '__main__':
    print("Backend running on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
