import cv2
import numpy as np

from collections import deque

class ENVEngine:
    BRIGHTNESS = (0.15, 0.85)
    OCCUPANCY  = (0.02, 0.30)
    JITTER     = 0.05

    ERROR  = -1
    NORMAL = 0
    LOW    = 1
    HIGH   = 2

    __STABLE = [1, 4, 33, 263, 61, 291, 168, 197]

    def __init__(self, buffer_size: int = 30):
        self._buffer_size = buffer_size
        self._lm_history  = deque(maxlen=buffer_size)
        self._brightness  = deque(maxlen=buffer_size)
        self._occupancy   = deque(maxlen=buffer_size)

    def push(self, img_bgr: np.ndarray, landmarks: list) -> None:
        h, w = img_bgr.shape[:2]
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]

        self._lm_history.append(np.array(
            [[landmarks[i][0] * w, landmarks[i][1] * h] for i in ENVEngine.__STABLE],
            dtype=np.float32,
        ))
        self._brightness.append(float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].mean()) / 255.0)
        self._occupancy.append(((max(xs) - min(xs)) * w) * ((max(ys) - min(ys)) * h) / (w * h))

    def reset(self) -> None:
        self._lm_history.clear()
        self._brightness.clear()
        self._occupancy.clear()

    @property
    def status(self) -> dict:
        if not self._lm_history:
            return {
            'brightness'     : ENVEngine.ERROR,
            'brightness_flag': ENVEngine.ERROR,
            'occupancy'      : ENVEngine.ERROR,
            'occupancy_flag' : ENVEngine.ERROR,
            'jitter'         : ENVEngine.ERROR,
            'jitter_flag'    : ENVEngine.ERROR,
            'reliable'       : ENVEngine.ERROR
        }

        brightness = float(np.mean(self._brightness))
        occupancy  = float(np.mean(self._occupancy))

        last_lm = self._lm_history[-1]
        face_w = last_lm[:, 0].max() - last_lm[:, 0].min()
        face_h = last_lm[:, 1].max() - last_lm[:, 1].min()
        face_d = float(np.sqrt(face_w**2 + face_h**2))

        jitter = 0.0
        if len(self._lm_history) >= 2:
            jitter = float(np.sqrt((np.diff(np.stack(self._lm_history), axis=0)**2).sum(axis=-1)).mean())
            jitter  = (jitter / face_d) if face_d > 0 else 1.0

        brightness_flag = ENVEngine.NORMAL
        if   brightness < ENVEngine.BRIGHTNESS[0]:  brightness_flag = ENVEngine.LOW
        elif brightness > ENVEngine.BRIGHTNESS[1]:  brightness_flag = ENVEngine.HIGH

        occupancy_flag = ENVEngine.NORMAL
        if   occupancy < ENVEngine.OCCUPANCY[0]:  occupancy_flag = ENVEngine.LOW
        elif occupancy > ENVEngine.OCCUPANCY[1]:  occupancy_flag = ENVEngine.HIGH

        jitter_flag = ENVEngine.NORMAL
        if jitter > ENVEngine.JITTER:  jitter_flag = ENVEngine.HIGH

        return {
            'brightness'     : round(brightness, 4),
            'brightness_flag': brightness_flag,
            'occupancy'      : round(occupancy, 4),
            'occupancy_flag' : occupancy_flag,
            'jitter'         : round(jitter, 4),
            'jitter_flag'    : jitter_flag,
            'reliable'       : all(f == ENVEngine.NORMAL for f in (brightness_flag, occupancy_flag, jitter_flag)),
        }
