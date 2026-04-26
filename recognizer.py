import shutil
import numpy as np

from pathlib import Path
from typing import Optional

from collections import deque
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

from whitelist import WhitelistDB

from FaceDetector.detector import Detector, PATH as DETECTOR_PATH
from embedder   import _Embedder, PATH as EMBEDDER_PATH
from adapter    import _Adapter , PATH as ADAPTER_PATH
from normalize  import _normalize_l2

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a, b))

class Recognizer:
    '''
    Parameters
    ----------
    `database`    : WhitelistDB
        SQLite whitelist object.
    -----
    `use_adapter` : bool
        Load fine-tune adapter if available. Default True.
    -----
    `buffer_size` : int
        Number of frames kept in recognition and PAD buffers.
        Better to be within 15 to 30.
    -----
    `threshold`   : float
        Cosine similarity threshold for identity matching.
        Average is 0.70:0.80 (appropriate for InsightFace ArcFace 512-d embeddings).
        Higher is stricter (fewer false accepts, more "unknown" results).
        Lower is more lenient (more accepts, higher false positive risk).
    '''

    def __init__(
        self,
        database    : WhitelistDB = WhitelistDB(),
        use_adapter : bool  = True,
        buffer_size : int   = 30,
        threshold   : float = 0.75,
    ):

        self._use_adapter = use_adapter
        self._buffer_size = buffer_size
        self._threshold   = threshold

        self._ensure_models()

        Detector.ACTIVATE_PAD = True

        self._database = database
        self._detector = Detector()
        self._embedder = _Embedder()
        self._adapter  = _Adapter()

        self._db_cache = None

        if use_adapter:
            self._adapter.load()

        self._emb_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._pad_buffer: deque[bool]       = deque(maxlen=buffer_size)

    #! ::::: ::::: Enrolment ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def meet(
        self,
        name    : str,
        frames  : list[np.ndarray],
        chat_id : int,
        password: str,
        callback: callable
    ) -> bool:
        '''
        Process a list of **BGR** frames for a **single person**.
        Ensure each frame contains only the target person's face.
        Intended for live enrolment when no prepared images are available.
        Call `tune()` after enrolments to update the adapter.

        Returns
        -------
        True if enrolment succeeded, otherwise False.
        '''

        try:
            vectors = []
            for idx, frame in enumerate(frames):
                if (emb := self._extract(frame)) is not None:
                    vectors.append(emb)
                    success = True
                else:
                    success = False

                callback({'code': 1, 'index': idx, 'success': success})

            if vectors:
                user_id = self._database.add_user(name, chat_id, password)
                self._database.add_embeddings(user_id, vectors)
                self._invalidate()
                callback({'code': 2, 'success': True})
                return True

            callback({'code': 2, 'success': False})
            return False

        except Exception:
            callback({'code': 2, 'success': False})
            return False

    def forget(self, user_id: str) -> bool:
        ''' **Remove** a user, **Update** the database, and **Invalidate** the adapter '''
        ok = self._database.remove_user(user_id)
        if ok:
            self._invalidate()
            if ADAPTER_PATH.exists():
                ADAPTER_PATH.unlink()
        return ok

    @property
    def list(self) -> list[dict]:
        ''' `[{id, name, password_hash, chat_id, embedding_count}, ...]` for all users '''
        return self._database.all_users()

    #! ::::: ::::: Fine-Tuning ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def tune(
        self,
        epochs     : int   = 30,
        margin     : float = 0.35,
        learn_rate : float = 1e-3,
        batch_size : int   = 16,
        callback   : callable = print
    ) -> float:
        '''
        Fine-tune the adapter MLP using current enrolled embeddings.

        Train the adapter with triplet loss to improve inter-class separation 
        and intra-class compactness. Call after `meet()` or `forget()` to
        re-calibrate the embedding space.

        Requirements
        ------------
        At least 2 enrolled persons.
        Each person must have at least +1 embedding (more = better generalization).

        Returns
        -------
        float : Final triplet loss value (lower = better class separation).
        '''
        gallery = self._database.all_embeddings()

        if len(gallery) < 2:
            return 0.0

        loss = self._adapter.train(
            gallery=gallery,
            epochs=epochs,
            margin=margin,
            learn_rate=learn_rate,
            batch_size=batch_size,
            callback=callback
        )
        self._adapter.save()
        self._db_cache = None

        return loss

    #! ::::: ::::: Recognition ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::C

    def look(self, frame: np.ndarray) -> dict:
        '''
        Inference Pipeline.
        Analyse one BGR frame. Call repeatedly for reliable results.
        Typically 15-30 frames once buffer is filled.

        Returns
        -------
        dict {
            'located'    : bool,       # Whether MediaPipe model detected a face
            'decided'    : bool,       # Whether a final decision was made
            'known'      : bool,       # If identity matched above threshold
            'id'         : str|None,   # Matched user ID (or None)
            'name'       : str|None,   # Matched user display name (or None)
            'chat_id'    : int|None,   # Matched user chat ID (or None)
            'confidence' : float,      # Cosine similarity score (0-1)
            'pad'        : bool|None,  # True if classified as unreal person
            'pad_score'  : float,      # Liveness confidence score
        }
        '''
        result = {
            'located'    : False,
            'decided'    : False,
            'known'      : False,
            'id'         : None,
            'name'       : None,
            'chat_id'    : None,
            'confidence' : 0.0,
            'pad'        : None,
            'pad_score'  : 0.0,
        }

        landmarks, bbox, _, _ = self._detector.detect(frame)
        if bbox is None:
            return result

        aligned = self._detector.align(frame, landmarks, bbox)
        if aligned is None:
            return result

        result['located'] = True

        self._emb_buffer.append(self._embedder.embed(aligned))

        avg_emb = _normalize_l2(np.mean(list(self._emb_buffer), axis=0))
        name, sim, uid = self._match(avg_emb)

        result['decided'] = True

        pad, pad_score = self.pad_status()

        if name is not None and sim >= self._threshold and pad is not None:
            user = self._database.get_by_id(uid)
            result.update(
                known      = True,
                id         = uid,
                name       = name,
                chat_id    = user['chat_id'] if user else None,
                confidence = round(sim, 3),
                pad        = pad,
                pad_score  = round(pad_score, 3),
            )
        else:
            result.update(
                confidence = round(sim, 3) if sim is not None else 0.0,
                pad        = pad,
                pad_score  = round(pad_score, 3),
            )

        return result

    def reset(self) -> None:
        self._emb_buffer.clear()

    def pad_status(self):
        _, status = self._detector.pad_status
        self._pad_buffer.append(status['live'])

        if len(self._pad_buffer) >= self._buffer_size:
            return all(self._pad_buffer), status['score']

        return None, 0.0

    #! ::::: ::::: Processes ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def _extract(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if img_bgr.shape == (self._detector._size, self._detector._size):
            return self._embedder.embed(img_bgr)

        landmarks, bbox, _, _ = self._detector.detect(img_bgr)
        if bbox is None:
            return None

        aligned = self._detector.align(img_bgr, landmarks, bbox)
        if aligned is None:
            return None

        return self._embedder.embed(aligned)

    def _match(self, emb: np.ndarray) -> tuple[Optional[str], float, Optional[str]]:
        ''' `emb`: **Average Raw** Embedding (Real-Time Input) '''

        if not self._cache_gallery():
            return None, 0.0, None

        q = self._adapter.project(emb)
        b_name, b_sim, b_id = None, -1.0, None
        for name, proj_embs in self._db_cache.items():
            p_sim = float(np.mean(sorted([_cosine_sim(q, e) for e in proj_embs], reverse=True)[:3]))
            if p_sim > b_sim:
                b_sim, b_name, b_id = p_sim, name, self._id_map.get(name)

        return b_name, b_sim, b_id

    def _cache_gallery(self) -> None:
        if self._db_cache is None:
            self._db_cache = {
                name: [self._adapter.project(e) for e in embs]
                for name, embs in self._database.all_embeddings().items()
            }
            self._id_map = {u['name']: u['id'] for u in self._database.all_users()}

        return self._db_cache

    def _invalidate(self) -> None:
        self._db_cache = None
        self._adapter.invalidate()

    #! ::::: ::::: Helper ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def _ensure_models(self) -> None:
        if not DETECTOR_PATH.exists():
            raise SystemError(f'MediaPipe Detector Model Was Not Found: {DETECTOR_PATH}.')

        if not EMBEDDER_PATH.exists():
            try:
                app = FaceAnalysis(name='buffalo_l')
                app.prepare(ctx_id=-1)
                src = Path.home() / '.insightface' / 'models' / 'buffalo_l' / 'w600k_r50.onnx'
                shutil.copy2(src, EMBEDDER_PATH)
            except Exception as e:
                raise SystemError(f'ArcFace ONNX Error: {e}')

    def __repr__(self) -> str:
        return (
            f'FaceRecognizer('
            f'KnownUsers={len(self._database.all_users())}, '
            f'Embedder={"ArcFace-R50-ONNX" if self._embedder._ort else "NULL"}, '
            f'Adapter={"Active" if self._adapter.trained else "Idle"}, '
            f'Threshold={self._threshold})'
        )
