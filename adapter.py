import numpy as np
from pathlib import Path

from normalize import _normalize_l2, _denominator

PATH = Path('adapter.npz')

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

class _Adapter:
    '''
    **Lightweight MLP** adapter trained with `Online Hard Triplet Loss`.
    **Architecture**: `Linear(512:256, ReLU) ~ Linear(256:128)`.

    After training, all stored embeddings are projected through the adapter
    before similarity comparison, significantly increasing inter-class distance.

    The adapter `adapter.npz` must be re-trained (`.invalidate()`) whenever persons are added or removed.
    '''

    IN_DIM, H_DIM, OUT_DIM = 512, 256, 128

    def __init__(self):
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = None
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def project(self, emb: np.ndarray) -> np.ndarray:
        '''
        **Projects original ArcFace 512-d embedding into an optimized 128-d space**
        '''
        if self._trained:
            try:
                z1 = emb @ self._W1.T + self._b1
                h  = _relu(z1)
                z2 = h @ self._W2.T + self._b2
                return _normalize_l2(z2)
            except:
                return emb
        return emb

    def invalidate(self) -> None:
        self._trained = False

    def save(self, path: Path = PATH) -> None:
        if self._trained:
            np.savez(
                str(path),
                W1=self._W1, b1=self._b1,
                W2=self._W2, b2=self._b2
            )
    
    def load(self, path: Path = PATH) -> bool:
        if path.exists():
            try:
                data = np.load(str(path))
                self._W1 = data['W1'].astype(np.float32)
                self._b1 = data['b1'].astype(np.float32)
                self._W2 = data['W2'].astype(np.float32)
                self._b2 = data['b2'].astype(np.float32)
                self._trained = True
                return True
            except:
                return False
        else:
            return False

    def train(
        self,
        gallery: dict[str, list[np.ndarray]],
        epochs:  int   = 30,
        margin:  float = 0.30,
        learn_rate: float = 1e-3,
        batch_size: int   = 16,
        callback: callable = print
    ) -> float:
        '''
        Param `gallery`: {name: [emb, ...]} - ArcFace 512-d raw embeddings
        Returns final mean triplet loss.
        '''

        if len(list(gallery.keys())) < 2:
            return 0.0

        # Flatten to (X=[emb, ...], Y=[id, ...]) for all persons
        X = np.array(
            [e for _, embs in gallery.items() for e in embs],
            dtype=np.float32
        )
        Y = np.array(
            [pid for pid, (_, embs) in enumerate(gallery.items()) for _ in range(len(embs))],
            dtype=np.int32
        )

        if len(X) < 3:
            return 0.0

        rng = np.random.default_rng(42)

        # He initialisation (For ReLu)
        self._W1 = rng.standard_normal((self.H_DIM, self.IN_DIM)).astype(np.float32) * np.sqrt(2 / self.IN_DIM)
        self._b1 = np.zeros(self.H_DIM, dtype=np.float32)
        self._W2 = rng.standard_normal((self.OUT_DIM, self.H_DIM)).astype(np.float32) * np.sqrt(2 / self.H_DIM)
        self._b2 = np.zeros(self.OUT_DIM, dtype=np.float32)

        # Adam state
        mW1 = np.zeros_like(self._W1); vW1 = np.zeros_like(self._W1)
        mb1 = np.zeros_like(self._b1); vb1 = np.zeros_like(self._b1)
        mW2 = np.zeros_like(self._W2); vW2 = np.zeros_like(self._W2)
        mb2 = np.zeros_like(self._b2); vb2 = np.zeros_like(self._b2)
        adam_b1, adam_b2, adam_eps, t = 0.9, 0.999, 1e-8, 0

        def forward(x):
            z1 = x @ self._W1.T + self._b1
            h  = _relu(z1)
            z2 = h @ self._W2.T + self._b2
            norm = _denominator(z2)
            proj = z2 / norm
            return z1, h, norm, proj

        def adam_step(p, m, v, g, t):
            m = adam_b1 * m + (1 - adam_b1) * g
            v = adam_b2 * v + (1 - adam_b2) * g**2
            m_hat = m / (1 - adam_b1**t)
            v_hat = v / (1 - adam_b2**t)
            p -= learn_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
            return p, m, v

        final_loss, prev_loss, n = 0.0, None, len(X)
        for epoch in range(epochs):
            idx = rng.permutation(n)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n, batch_size):
                bi = idx[start:start+batch_size]
                bx, bl = X[bi], Y[bi]   # Batch data (embeddings and labels)

                z1, h, norm, proj = forward(bx)  # (B, OUT_DIM)

                # Mining (pairwise squared distances in projected space)
                dot  = proj @ proj.T                          # Similarity Matrix
                sq   = np.sum(proj**2, axis=1, keepdims=True) # Squared L2 Norm
                dist = np.maximum(sq + sq.T - 2 * dot, 0.0)   # Distance Matrix

                same = (bl[:, None] == bl[None, :])   # (B, B) bool
                diff = ~same

                # Hard Positive: same label, largest distance
                dist_p = np.where(same, dist, -1.0)
                np.fill_diagonal(dist_p, -1.0)
                hp_idx = np.argmax(dist_p, axis=1)

                # Hard Negative: diff label, smallest distance
                dist_n = np.where(diff, dist, np.inf)
                hn_idx = np.argmin(dist_n, axis=1)

                dp = dist[np.arange(len(bi)), hp_idx]
                dn = dist[np.arange(len(bi)), hn_idx]

                losses = np.maximum(0, dp - dn + margin)
                active = (losses > 0) & (dist_p > -1.0).any(axis=1)

                if not active.any():
                    continue

                loss = losses[active].mean()
                epoch_loss += loss
                n_batches  += 1

                # Backprop | dL/d(dp) = 1, dL/d(dn) = -1 for active triplets
                dL = np.zeros(len(bi), dtype=np.float32)
                dL[active] = 1.0 / active.sum()

                proj_a = proj           # (B, OUT_DIM)
                proj_p = proj[hp_idx]   # Hard Positives
                proj_n = proj[hn_idx]   # Hard Negatives

                # Backpropagation
                dOut = (
                     2 * dL[:, None] * (proj_a - proj_p)   # Pull Positives
                    -2 * dL[:, None] * (proj_a - proj_n)   # Push Negatives
                ) * active[:, None]

                # Chain through W2          
                dRaw = (dOut - np.einsum('bi,bi->b', dOut, proj)[:, None] * proj) / norm 

                dW2 = dRaw.T @ h
                db2 = dRaw.sum(axis=0)
                dh2 = dRaw @ self._W2

                # Chain through W1 & ReLU
                dh1 = dh2 * (z1 > 0).astype(np.float32)
                dW1 = dh1.T @ bx
                db1 = dh1.sum(axis=0)

                # Adam updates
                t += 1
                self._W1, mW1, vW1 = adam_step(self._W1, mW1, vW1, dW1, t)
                self._b1, mb1, vb1 = adam_step(self._b1, mb1, vb1, db1, t)
                self._W2, mW2, vW2 = adam_step(self._W2, mW2, vW2, dW2, t)
                self._b2, mb2, vb2 = adam_step(self._b2, mb2, vb2, db2, t)

            if n_batches:
                final_loss = epoch_loss / n_batches
                delta = 0.0 if prev_loss is None else (final_loss - prev_loss)
                prev_loss = final_loss
                callback({'code': 4, 'epoch': epoch, 'loss': final_loss, 'delta': delta})

        self._trained = True
        return final_loss
