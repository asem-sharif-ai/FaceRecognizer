import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from typing import Optional
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from FaceDetector.pad_engine import PADEngine
from FaceDetector.env_engine import ENVEngine
from FaceDetector.ga_engine import GenderAgeEngine


PATH = Path('FaceDetector/detector.task')

class Flag:
    ERROR  = -1
    NORMAL = 0
    LOW    = 1
    HIGH   = 2

class Draw:
    BBOX       = 0
    BBOX_B     = 1
    LANDMARKS  = 2
    WIREFRAME  = 3
    OVERLAY    = 4
    OVERLAY_O  = 5
    OVERLAY_H  = 6
    OVERLAY_W  = 7
    OVERLAY_WO = 8
    OVERLAY_WH = 9

    OK      = (  0, 210,  80)  # Green   — normal / live / pass
    WARN    = (  0, 160, 255)  # Amber   — low confidence / borderline
    ERROR   = ( 20,  20, 220)  # Red     — spoof / fail / blocked
    INFO    = (235, 170,   0)  # Sky     — neutral info / idle
    UNKNOWN = (150, 150, 150)  # Gray    — undecided / buffer not ready

    WHITE   = (255, 255, 255)  # Pure White
    ICE     = (255, 235, 180)  # Light Sky Blue
    CYAN    = (255, 210,   0)  # Vivid Blue/Cyan
    SILVER  = (200, 200, 200)  # Light Gray
    MINT    = (180, 230, 100)  # Pale Turquoise
    NEON    = (  0, 255, 180)  # Bright Lime/Yellow-Green
    SKY     = (235, 170,   0)  # Cerulean
    TEAL    = (200, 180,   0)  # Deep Sky Blue
    GOLD    = (  0, 185, 230)  # Bright Gold/Orange
    LIME    = (  0, 230, 120)  # Vivid Grass Green
    GREEN   = ( 20, 210,  60)  # Lime Green
    AMBER   = (  0, 160, 255)  # Deep Amber/Orange
    PLASMA  = (255,  50, 120)  # Purple-Blue
    VIOLET  = (210,  40, 180)  # Soft Violet/Purple
    INDIGO  = (200,  60,  60)  # Deep Blue-Purple
    CORAL   = ( 50,  80, 255)  # Vibrant Red-Orange
    MAGENTA = (180,   0, 200)  # Dark Magenta
    NAVY    = (110,  30,  10)  # Dark Blue
    RED     = ( 20,  20, 220)  # Pure Red
    CRIMSON = (  0,   0, 180)  # Deep Red

    MODE  = OVERLAY_WH
    COLOR = SKY

class Detector:
    MIRROR = True
    ACTIVATE_ENV = False
    ACTIVATE_PAD = False
    ACTIVATE_GA  = False

    EULER_ANGLES = (15, 15, 20)
    EAR_RANGE    = (0.20, 0.40)
    MAR_RANGE    = (0.05, 0.40)
    YAW_OFFSET   = 0

    _FILTER = {
        'c': [],        # 'cheek'
        'n': [],        # 'nose'
        'j': ['Open']   # 'jaw'
    }

    __FACE_OVAL = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 
        152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    __L_EYE  = [33,  160, 158, 133, 153, 144]
    __R_EYE  = [362, 385, 387, 263, 373, 380]
    __MOUTH  = [61,  291, 13,  14,  78,  308]
    __STABLE = [1, 4, 33, 263, 61, 291, 168, 197]

    __OPPOSITE = {
        'HeadLeft'  : 'HeadRight',
        'HeadRight' : 'HeadLeft' ,
        'RollRight' : 'RollLeft' ,
        'RollLeft'  : 'RollRight',
    }

    def __init__(
            self,
            path  : Path  = PATH,
            size  : int   = 112,
            ratio : float = 0.85,
            smooth: float = 0.5,
            buffer: int   = 30,
        ):

        self._mp_lmr = None
        self._last_lm = None
        self._last_bbox = None

        self._frame_ts = 0
        self._pad_engine = PADEngine(buffer_size=buffer)
        self._env_engine = ENVEngine(buffer_size=buffer)
        self._ga_engine  = GenderAgeEngine(buffer_size=buffer*2)

        self._size = size
        self._ratio = ratio
        self._smooth = smooth

        self._lm_history = deque(maxlen=10)

        self.__load(path)

    def __load(self, model_path: Path) -> None:
        if model_path.exists():
            try:
                self._mp_lmr = mp_vision.FaceLandmarker.create_from_options(
                    mp_vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=mp_vision.RunningMode.VIDEO,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                    )
                )
            except Exception as e:
                raise ImportError(f'MediaPipe FaceLandmarker Error: {e}.')

    def __ema(self, old, new):
        a, b = self._smooth, 1 - self._smooth
        return tuple(a*o + b*n for o, n in zip(old, new))

    def __lm_xy(lm, w, h):
        if isinstance(lm, tuple):
            return lm[0] * w, lm[1] * h
        return lm.x * w, lm.y * h

    def detect(self, img_bgr: np.ndarray) -> tuple[Optional[list], Optional[tuple], Optional[dict], Optional[np.ndarray]]:
        '''
        Process a single BGR frame
        Returns
        -------
        - Optional[list]        MediaPipe Face 478 Mesh Points
        - Optional[tuple]       Face Bounding Box 
        - Optional[dict]        Face Formatted Blendshapes
        - Optional[np.ndarray]  Facial Transformation Matrixes
        '''
        
        N = (None, None, None, None)

        if self._mp_lmr is None:
            return N

        self._frame_ts += 33  # ~30 FPS in MS
        result = self._mp_lmr.detect_for_video(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
            self._frame_ts
        )

        if not result.face_landmarks:
            return N

        landmarks = result.face_landmarks[0]
        h, w = img_bgr.shape[:2]

        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        pad = int((x2 - x1) * 0.15)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        bbox = (x1, y1, x2 - x1, y2 - y1)

        if self._last_lm is None:
            self._last_lm = [(l.x, l.y, l.z) for l in landmarks]
            self._last_bbox = bbox
        else:
            self._last_lm = [self.__ema(p, (c.x, c.y, c.z)) for p, c in zip(self._last_lm, landmarks)]
            self._last_bbox = self.__ema(self._last_bbox, bbox)

        blendshapes = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = category.category_name
                if name.startswith('_'): continue   # '_neutral'
                
                idx = next((i for i, c in enumerate(name) if c.isupper()), len(name))
                group = name[:idx]
                feature = name[idx:] if idx < len(name) else name

                if group[0] in Detector._FILTER.keys():
                    if not (filtered := Detector._FILTER[group[0]]) or feature not in filtered:
                        continue

                if group not in blendshapes:
                    blendshapes[group] = {}

                blendshapes[group][feature] = round(category.score, 4)

        transformation = None
        if result.facial_transformation_matrixes:
            transformation = result.facial_transformation_matrixes[0].reshape(4, 4)

        if Detector.ACTIVATE_ENV:
            self._env_engine.push(img_bgr, self._last_lm)

        return self._last_lm, self._last_bbox, blendshapes, transformation

    def align(self, img_bgr: np.ndarray, landmarks: list, bbox: tuple) -> Optional[np.ndarray]:
        '''
        Align face to **canonical pose** using **eye corners** from **MediaPipe 478-point mesh**
        Returns `aligned` `sizexsize` `RGB` face crop with a bit of margin (no masking).
        '''
        def _align_face(
                img_bgr  : np.ndarray,
                landmarks: list,
                size     : int,
                ratio    : float
            ) -> Optional[np.ndarray]:

            h, w = img_bgr.shape[:2]
            def lm(idx):
                return np.array(list(Detector.__lm_xy(landmarks[idx], w, h)), dtype=np.float32)

            try:
                face_points = np.array([lm(i) for i in self.__FACE_OVAL])
                l_eye = (lm(33) + lm(133)) / 2
                r_eye = (lm(362) + lm(263)) / 2
            except:
                return None

            # 1. Compute rotation to level eyes
            dY = r_eye[1] - l_eye[1]
            dX = r_eye[0] - l_eye[0]
            angle = float(np.degrees(np.arctan2(dY, dX)))

            # 2. Bounding box of rotated face points
            mass_center = np.mean(face_points, axis=0)
            rotation_mx = cv2.getRotationMatrix2D(tuple(mass_center), angle, 1.0)

            rotated_pts = np.hstack(
                [face_points, np.ones(shape=(len(face_points), 1))]
            ).dot(rotation_mx.T)

            min_x, min_y = np.min(rotated_pts, axis=0)
            max_x, max_y = np.max(rotated_pts, axis=0)

            # 3. Scale to size×size with a bit of margin
            scale = (size * ratio) / max(max_x - min_x, max_y - min_y-10)

            # 4. Final affine matrix
            M = cv2.getRotationMatrix2D(tuple(mass_center), angle, scale)
            M[0, 2] += (size // 2) - mass_center[0]
            M[1, 2] += (size // 2) - mass_center[1]

            # 5. Warp: fill border with face-region mean
            aligned = cv2.warpAffine(
                img_bgr, M, (size, size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101
            )
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        if landmarks is not None:
            if (aligned := _align_face(img_bgr, landmarks, self._size, self._ratio)) is not None:
                self._push_engines(aligned)
                return aligned

        x, y, w, h = [int(v) for v in bbox]
        if (crop := img_bgr[y:y+h, x:x+w]).size != 0:
            aligned = cv2.cvtColor(cv2.resize(crop, (self._size, self._size)), cv2.COLOR_BGR2RGB)
            self._push_engines(aligned)
            return aligned

        return None

    def convex_hull(self, img_bgr: np.ndarray, landmarks: list) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        
        pts = np.array([list(Detector.__lm_xy(landmarks[i], w, h)) for i in Detector.__FACE_OVAL], dtype=np.int32)

        final = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(final, cv2.convexHull(pts), 255)
        return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    def gradient(self, img_bgr: np.ndarray, landmarks: list) -> np.ndarray:
        h, w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mag  = cv2.normalize(
            np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)**2),
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        mask  = cv2.cvtColor(self.convex_hull(img_bgr, landmarks), cv2.COLOR_BGR2GRAY)
        final = np.zeros((h, w), dtype=np.uint8)
        final[mask == 255] = mag[mask == 255]
        return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    def _push_engines(self, aligned_rgb: np.ndarray):
        if Detector.ACTIVATE_PAD:
            self._pad_engine.push(aligned_rgb)
        if Detector.ACTIVATE_GA:
            self._ga_engine.push(aligned_rgb)

    @property
    def env_status(self) -> tuple[bool, dict]:
        return Detector.ACTIVATE_ENV, self._env_engine.status

    @property
    def pad_status(self) -> tuple[bool, dict]:
        r, l, s = self._pad_engine.decide()
        return Detector.ACTIVATE_PAD, {'ready': r,  'live': l, 'score': s}

    @property
    def ga_status(self) -> tuple[bool, str, int]:
        return Detector.ACTIVATE_GA, *self._ga_engine.status

    @staticmethod
    def euler_angles(
        frame    : np.ndarray,
        landmarks: list,
        transformation: Optional[np.ndarray],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:

        def solve(frame, landmarks):
            OBJ_PTS = np.array([
                (   0.0,    0.0,    0.0),
                (   0.0, -330.0,  -65.0),
                (-225.0,  170.0, -135.0),
                ( 225.0,  170.0, -135.0),
                (-150.0, -150.0, -125.0),
                ( 150.0, -150.0, -125.0),
            ], dtype='double')

            h, w = frame.shape[:2]

            IMG_PTS = np.array(
                [list(Detector.__lm_xy(landmarks[i], w, h)) for i in [1, 152, 33, 263, 61, 291]],
                dtype='double',
            )

            fw = float(w)
            ok, rvec, tvec = cv2.solvePnP(
                OBJ_PTS, IMG_PTS,
                np.array([[fw, 0, fw/2], [0, fw, h/2], [0, 0, 1]], dtype='double'),
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return None, None, None

            rmat, _ = cv2.Rodrigues(rvec)
            proj    = np.hstack((rmat, tvec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

            yaw_raw = float(euler[1][0])

            pitch_raw = float(euler[0][0])
            if   pitch_raw < -90:  pitch_raw = -(180 + pitch_raw)
            elif pitch_raw >  90:  pitch_raw =  (180 - pitch_raw)

            roll_raw = float(euler[2][0])
            if   roll_raw < -90:  roll_raw = -(180 + roll_raw)
            elif roll_raw >  90:  roll_raw =  (180 - roll_raw)

            return round(yaw_raw, 4), round(-pitch_raw, 4), round(roll_raw, 4)

        def label(y, p, r, mirror=False) -> tuple[str, str, str]:
            if None in (y, p, r): return None, None, None

            L = Detector.EULER_ANGLES
            lr = ud = rl = 'Neutral'

            if y >  L[0]:  lr = 'HeadLeft'
            if y < -L[0]:  lr = 'HeadRight'

            if p >  L[1]:  ud = 'HeadDown'
            if p < -L[1]:  ud = 'HeadUp'

            if r >  L[2]:  rl = 'RollRight'
            if r < -L[2]:  rl = 'RollLeft'

            if mirror:
                lr = Detector.__OPPOSITE.get(lr, 'Neutral')
                rl = Detector.__OPPOSITE.get(rl, 'Neutral')

            return lr, ud, rl

        if transformation is None:
            y, p, r = solve(frame, landmarks)
            ys, ps, rs = label(y, p, r, Detector.MIRROR)
        else:
            R = transformation[:3, :3]
            y = float(np.degrees(np.arcsin(-R[2, 0]))) - Detector.YAW_OFFSET
            p = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            r = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            ys, ps, rs = label(y, p, r, False)
        
        return y, p, r, ys, ps, rs

    @staticmethod
    def eye_aspect_ratio(
        frame    : np.ndarray,
        landmarks: list,
        threshold: Optional[tuple[float, float]] = None
    ) -> tuple[float, float, str, str]:

        h, w = frame.shape[:2]

        def label(r, t):
            if r <= t[0]:
                return Flag.LOW
            elif r >= t[1]:
                return Flag.HIGH
            return Flag.NORMAL

        def calculate(indices) -> float:
            try:
                p = [np.array(Detector.__lm_xy(landmarks[i], w, h)) for i in indices]
                r = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * np.linalg.norm(p[0] - p[3]))
                return r, label(r, threshold or Detector.EAR_RANGE)
            except:
                return 0.0, Flag.ERROR
            
        l_ear, l_ear_lbl = calculate(Detector.__L_EYE)
        r_ear, r_ear_lbl = calculate(Detector.__R_EYE)

        return round(l_ear, 4), round(r_ear, 4), l_ear_lbl, r_ear_lbl

    @staticmethod
    def mouth_aspect_ratio(
        frame    : np.ndarray,
        landmarks: list,
        threshold: Optional[float] = None
    ) -> tuple[float, str]:

        h, w = frame.shape[:2]
        try:
            t = threshold or Detector.MAR_RANGE
            p = [np.array(Detector.__lm_xy(landmarks[i], w, h)) for i in Detector.__MOUTH]
            r = np.linalg.norm(p[2] - p[3]) / np.linalg.norm(p[4] - p[5])
            return round(r, 4), (Flag.NORMAL if r <= t[0] else Flag.HIGH if r >= t[1] else Flag.LOW)
        except:
            return 0.0, Flag.ERROR

    @staticmethod
    def draw(
        img_bgr  : np.ndarray,
        landmarks: list,
        bbox     : Optional[tuple],
    ) -> np.ndarray:

        if landmarks is None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mode, color = Draw.MODE, Draw.COLOR
        final, (h, w) = img_bgr.copy(), img_bgr.shape[:2]

        def _pt_xy(idx):
            return (int(landmarks[idx][0]*w), int(landmarks[idx][1]*h))

        def _smooth(pts, radius=10):
            hi = cv2.convexHull(pts, returnPoints=False).flatten()
            hp = pts[hi].astype(np.float32)
            n  = len(hp)

            smoothed = []
            for i in range(n):
                p0 = hp[(i - 1) % n]
                p1 = hp[i]
                p2 = hp[(i + 1) % n]

                d0 = np.linalg.norm(p1 - p0)
                d1 = np.linalg.norm(p2 - p1)
                r  = min(radius, d0 * 0.45, d1 * 0.45)

                a = p1 + (p0 - p1) / d0 * r
                b = p1 + (p2 - p1) / d1 * r

                smoothed.extend([a, b])

            return np.array(smoothed, dtype=np.int32)

        def _mask(poly):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            return mask

        def _blend(layer, alpha):
            cv2.addWeighted(layer, alpha, final, 1-alpha, 0, final)

        smooth = _smooth(np.array([_pt_xy(i) for i in Detector._Detector__FACE_OVAL], dtype=np.int32))

        def _bbox():
            x, y, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(final, (x, y), (x+bw, y+bh), color, 1, cv2.LINE_AA)

        def _brackets():
            xs, ys = smooth[:,0], smooth[:,1]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            for cx, cy, hx, vy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1),(x2, y2, -1, -1)]:
                cv2.line(final, (cx, cy), (cx+hx*20, cy), color, 2, cv2.LINE_AA)
                cv2.line(final, (cx, cy), (cx, cy+vy*20), color, 2, cv2.LINE_AA)

        def _overlay(alpha=0.10):
            mask   = _mask(smooth)
            layer  = final.copy()
            layer[mask==255] = color

            region = final.copy()
            cv2.addWeighted(layer, alpha, region, 1-alpha, 0, region)

            final[mask==255] = region[mask==255]

        def _outline(thick=2):
            cv2.polylines(final, [smooth], True, color, thick, cv2.LINE_AA)

        def _halo():
            halo = final.copy()
            cv2.polylines(halo, [smooth], True, color, 10, cv2.LINE_AA)
    
            _blend(halo, 0.10)
            _outline()

        def _landmarks():
            for lm in landmarks:
                cv2.circle(final, (int(lm[0]*w), int(lm[1]*h)), 1, color, -1, cv2.LINE_AA)

        def _wireframe(alpha=0.90):
            layer = final.copy()
            for a, b in [
                (10, 1),   (1,   152), (10,  133), (10,  362),
                (33, 133), (362, 263), (33,  1),   (263, 1),
                (33, 61),  (263, 291), (61,  291), (61,  152), (291, 152),
                (33, 234), (263, 454), (234, 454), (127, 356)
            ]:
                cv2.line(layer, _pt_xy(a), _pt_xy(b), color, 1, cv2.LINE_AA)
            _blend(layer, alpha)

        if   mode == Draw.BBOX:
            _bbox()

        elif mode == Draw.BBOX_B:
            _bbox()
            _brackets()

        elif mode == Draw.LANDMARKS:
            _landmarks()

        elif mode == Draw.WIREFRAME:
            _wireframe()

        elif mode == Draw.OVERLAY:
            _overlay()

        elif mode == Draw.OVERLAY_O:
            _overlay()
            _outline()

        elif mode == Draw.OVERLAY_H:
            _overlay()
            _halo()

        elif mode == Draw.OVERLAY_W:
            _overlay()
            _wireframe()

        elif mode == Draw.OVERLAY_WO:
            _overlay()
            _wireframe()
            _outline()

        elif mode == Draw.OVERLAY_WH:
            _overlay()
            _wireframe()
            _halo()

        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
