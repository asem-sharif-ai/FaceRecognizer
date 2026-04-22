import cv2 
import numpy as np
import onnxruntime as ort

from pathlib import Path
from typing import Optional
from collections import deque

PATH = Path('FaceDetector/genderage.onnx')

class GenderAgeEngine:
    def __init__(self, path: Path = PATH, buffer_size: int = 30):
        if not path.exists():
            raise FileNotFoundError(f'GanderAge ONNX Model was not found at {path}')
        
        self._buffer_size = buffer_size
        
        self._model = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self._input = self._model.get_inputs()[0].name

        self._gen_buf = deque(maxlen=buffer_size)
        self._age_buf = deque(maxlen=buffer_size)

    def push(self, face_rgb: np.ndarray) -> None:
        x = cv2.resize(face_rgb, (96, 96))
        y = self._model.run(None, {self._input: x.astype(np.float32).transpose(2, 0, 1)[None]})[0][0]
        self._gen_buf.append(int(np.argmax(y[:2])))
        self._age_buf.append(float(y[2]) * 100)

    def predict(self, face_rgb: Optional[np.ndarray] = None) -> tuple[Optional[str], int]:
        if face_rgb is not None:
            self.push(face_rgb)
        
        if not self._gen_buf or len(self._gen_buf) < self._buffer_size:
            return 'UnKnown', 0

        gen = 'Male' if sum(self._gen_buf) > len(self._gen_buf) / 2 else 'Female'
        age = round(float(np.mean(self._age_buf)) / 5) * 5
        return gen, age

    def reset(self) -> None:
        self._gen_buf.clear()
        self._age_buf.clear()

    @property
    def status(self):
        return self.predict()
