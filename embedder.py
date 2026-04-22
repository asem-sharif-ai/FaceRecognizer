import numpy as np
import onnxruntime as ort
from pathlib import Path

from normalize import _normalize_l2

PATH = Path('w600k_r50.onnx')

class _Embedder:
    '''
    Produces a `512-d L2-normalized embedding` from a `112x112 RGB` face.

    Model
    -----
    InsightFace w600k_r50.onnx
        - Intra-class cosine similarity: ~0.60:0.90
        - Inter-class cosine similarity: ~0.15:0.30
    '''

    INPUT_SIZE = (112, 112)

    def __init__(self, model_path: Path = PATH):
        self._ort   = None
        self._input = None
        self._dim   = 256

        self._load(model_path)

    def _load(self, path: Path) -> None:
        '''
        Load ArcFace ONNX model if available.

        Uses CPUExecutionProvider by default.
        Attempts NNAPIExecutionProvider on Pi if present.
        '''
        if not path.exists():
            raise ImportError('ArcFace ONNX not found.')
        try:
            s_options = ort.SessionOptions()
            s_options.inter_op_num_threads = 4
            s_options.intra_op_num_threads = 4

            providers = ['CPUExecutionProvider']
            available = [p[0] if isinstance(p, tuple) else p for p in ort.get_available_providers()]

            if 'NNAPIExecutionProvider' in available:
                providers = ['NNAPIExecutionProvider', 'CPUExecutionProvider']

            self._ort = ort.InferenceSession(str(path), sess_options=s_options, providers=providers)
            self._input = self._ort.get_inputs()[0].name
            self._dim = self._ort.run(
                None,
                {self._input: np.zeros((1, 3, 112, 112), dtype=np.float32)}
            )[0].shape[-1]

        except Exception as e:
            raise ImportError(f'ArcFace ONNX Error: {e}.')

    def embed(self, image: np.ndarray) -> np.ndarray:
        '''
        Input
        -----
        `image`: np.ndarray - Aligned 112x112 RGB face image.

        Output
        ------
        `embedding`: np.ndarray - L2-normalized 512-d embedding.

        Steps
        -----
            1. Convert RGB uint8 to float32.
            2. Subtract 127.5 and divide by 128.
            3. Convert HWC to CHW and add batch dim.
            4. Process ONNX model.
            5. L2-normalize output.
        '''
        if self._ort is not None:
            img = image
            x = img.astype(np.float32)
            x = (x - 127.5) / 128.0
            x = x.transpose(2, 0, 1)[None]   # (1, 3, 112, 112)
            out = self._ort.run(None, {self._input: x})[0][0]
            return _normalize_l2(out.astype(np.float64))

    @property
    def dim(self) -> int:
        return self._dim
