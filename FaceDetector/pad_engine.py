import cv2
import numpy as np

from typing import Optional
from collections import deque
from skimage.feature import local_binary_pattern

class Weight:
    LBP        = 0.30
    HF_POWER   = 0.20
    TEXTURE    = 0.15
    COLOUR     = 0.10
    REFLECTION = 0.05
    TEMPORAL   = 0.05

class PADEngine:
    def __init__(self, threshold: float = 0.60, buffer_size: int = 60):
        self._threshold   = threshold
        self._buffer_size = buffer_size

        self._buffer = deque(maxlen=buffer_size)
        self._scores = deque(maxlen=buffer_size)

    def push(self, rgb: np.ndarray) -> None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self._buffer.append((gray.astype(np.float32), rgb.copy()))
        self._scores.append(self._score_frame(gray, rgb))

    def reset(self) -> None:
        self._buffer.clear()
        self._scores.clear()

    def ready(self) -> bool:
        return len(self._buffer) >= self._buffer_size

    def decide(self) -> tuple[bool, Optional[bool], float]:
        if not self.ready():
            return False, None, 0.0

        score = self._weight(
            (float(np.mean(list(self._scores))), (1 - Weight.TEMPORAL)),
            (self._temporal_score(),                  Weight.TEMPORAL )
        )
        return True, score >= self._threshold, round(score, 4)

    def _score_frame(self, gray: np.ndarray, rgb: np.ndarray) -> float:
        score = self._weight(
            (self._lbp_score(gray)        , Weight.LBP        ),
            (self._hf_score(gray)         , Weight.HF_POWER   ),
            (self._texture_score(gray)    , Weight.TEXTURE    ),
            (self._colour_score(rgb)      , Weight.COLOUR     ),
            (self._reflection_score(rgb)  , Weight.REFLECTION )
        ) / (Weight.LBP + Weight.HF_POWER + Weight.TEXTURE + Weight.COLOUR + Weight.REFLECTION)

        return float(np.clip(score, 0.0, 1.0))

    def _lbp_score(self, gray: np.ndarray) -> float:
        hist, _ = np.histogram(
            local_binary_pattern(gray.astype(np.uint8), 8, 1, 'uniform').ravel(),
            bins=10, range=(0, 10), density=True
        )
        hist = hist + 1e-10
        entropy = float(-np.sum(hist * np.log2(hist)))
        return float(np.clip((entropy - 2.0) / 1.5, 0.0, 1.0))

    def _hf_score(self, gray: np.ndarray) -> float:
        mag = np.abs(
            np.fft.fftshift(
                np.fft.fft2(gray.astype(np.float32))
            )
        )

        h, w   = gray.shape
        cy, cx = h // 2, w // 2
        y, x   = np.ogrid[:h, :w]
        
        lf_mask = np.sqrt((x - cx)**2 + (y - cy)**2) <= 10
        hf_mask = ~lf_mask

        hf_energy = np.sum(mag * hf_mask)
        lf_energy = np.sum(mag * lf_mask) + 1e-10
        
        score = 1.0 - abs(hf_energy / lf_energy - 2.75) / 3.0
        return float(np.clip(score, 0.0, 1.0))

    def _texture_score(self, gray: np.ndarray) -> float:
        f_gray = gray.astype(np.float32)
        sigma = np.std(cv2.absdiff(f_gray, cv2.GaussianBlur(f_gray, (5, 5), 0)))
        score = np.exp(-0.5 * (abs(sigma / (np.mean(f_gray) + 1e-5) - 0.09) / 0.05)**2)
        return float(np.clip(score, 0.0, 1.0))

    def _colour_score(self, rgb: np.ndarray) -> float:
        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        cb = ycbcr[:, :, 2]
        cr = ycbcr[:, :, 1]
        score = (np.mean((cb >= 77) & (cb <= 127)) + np.mean((cr >= 133) & (cr <= 173))) / 2
        return float(np.clip(score, 0.0, 1.0))

    def _reflection_score(self, rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, highlights = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        ratio = highlights.sum() / (255 * gray.size)
        if ratio < 0.001: return 0.5
        score = 1.0 - abs(ratio - 0.02) / 0.05
        return float(np.clip(score, 0.0, 1.0))

    def _temporal_score(self) -> float:
        if len(self._buffer) < 5: return 0.5

        frames = [g for g, _ in self._buffer]
        diff = [np.abs(frames[i] - frames[i-1]).mean() for i in range(1, len(frames))]
        mean_diff, std_diff = float(np.mean(diff)), float(np.std(diff))

        if mean_diff < 0.2: return 0.0
        
        return float(np.clip(mean_diff / 3.0, 0.0, 1.0) * 0.7 + np.clip(std_diff / 1.5, 0.0, 1.0) * 0.3)

    @staticmethod
    def _weight(*sw_pairs):
        return sum(s * w for s, w in sw_pairs)


