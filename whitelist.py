import uuid
import hashlib
import sqlite3
import numpy as np
from pathlib import Path

PATH = Path('whitelist.db')

class WhitelistDB:
    def __init__(self, db_path:Path=PATH, embed_dim:int=512, password:str='lynxDMS'):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._dim  = embed_dim
        self._build(password)

    def _build(self, password: str) -> None:
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT    PRIMARY KEY,
                name          TEXT    NOT NULL UNIQUE,
                chat_id       INTEGER NOT NULL,
                password_hash TEXT    NOT NULL
            )
        ''')
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT    NOT NULL REFERENCES users(id),
                vector  BLOB    NOT NULL
            )
        ''')
        self._conn.commit()
        self._ensure_admin(password)

    def _ensure_admin(self, password: str) -> None:
        row = self._conn.execute('SELECT COUNT(*) FROM users').fetchone()
        if row[0] == 0:
            self._conn.execute(
                'INSERT INTO users(id, name, chat_id, password_hash) VALUES (?,?,?,?)',
                ('00', 'Admin', 0, self._hash(password))
            )
            self._conn.commit()

    def _next_id(self) -> str:
        existing = {r[0] for r in self._conn.execute('SELECT id FROM users').fetchall()}
        for i in range(100):
            cid = f'{i:02d}'
            if cid not in existing:
                return cid
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _hash(plaintext: str) -> str:
        return hashlib.md5(plaintext.encode()).hexdigest()

    #! ::::: ::::: Embeddings ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def add_embedding(self, user_id: str, vector: np.ndarray) -> None:
        blob = vector.astype(np.float64).tobytes()
        with self._conn:
            self._conn.execute(
                'INSERT INTO embeddings(user_id, vector) VALUES (?,?)', 
                (user_id, blob)
            )

    def add_embeddings(self, user_id: str, vectors: list[np.ndarray]) -> None:
        blobs = [(user_id, v.astype(np.float64).tobytes()) for v in vectors]
        with self._conn:
            self._conn.executemany(
                'INSERT INTO embeddings(user_id, vector) VALUES (?,?)', blobs
            )

    def embedding_count(self, name: str) -> int | None:
        '''Returns embedding count for the given name, or None if the user does not exist.'''
        user = self._conn.execute('SELECT id FROM users WHERE name=?', (name,)).fetchone()
        if not user:
            return None
        row = self._conn.execute(
            'SELECT COUNT(*) FROM embeddings WHERE user_id=?', (user[0],)
        ).fetchone()
        return row[0]

    def all_embeddings(self) -> dict[str, list[np.ndarray]]:
        '''Returns {name: [emb1, emb2, ...]} with float64 vectors.'''
        cur = self._conn.execute('''
            SELECT u.name, e.vector
            FROM embeddings e
            JOIN users u ON u.id = e.user_id
        ''')
        result: dict[str, list[np.ndarray]] = {}
        for name, blob in cur.fetchall():
            vec = np.frombuffer(blob, dtype=np.float64)
            result.setdefault(name, []).append(vec)
        return result

    #! ::::: ::::: Users ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def add_user(self, name: str, chat_id: int, password: str) -> str:
        '''
        Add a new user and return their id.
        If the user already exists, update chat_id and return existing id.
        Password is hashed with MD5 before storage.
        '''
        row = self._conn.execute('SELECT id FROM users WHERE name=?', (name,)).fetchone()

        if row:
            self._conn.execute(
                'UPDATE users SET chat_id=? WHERE id=?', (chat_id, row[0])
            )
            self._conn.commit()
            return row[0]

        uid = self._next_id()
        self._conn.execute(
            'INSERT INTO users(id, name, chat_id, password_hash) VALUES (?,?,?,?)',
            (uid, name, chat_id, self._hash(password))
        )
        self._conn.commit()
        return uid

    def update_user(self, user_id: str, name: str=None, password: str=None, chat_id: int=None) -> bool:
        '''
        Update one or more fields for an existing user by id.
        Returns False if the user was not found.
        '''
        fields, values = [], []

        if name     is not None:
            fields.append('name=?')
            values.append(name)
        
        if password is not None:
            fields.append('password_hash=?')
            values.append(self._hash(password))
        
        if chat_id  is not None:
            fields.append('chat_id=?')
            values.append(chat_id)

        if not fields:
            return False

        values.append(user_id)
        cur = self._conn.execute(
            f'UPDATE users SET {", ".join(fields)} WHERE id=?', values
        )
        self._conn.commit()
        return cur.rowcount > 0

    def remove_user(self, user_id: str) -> bool:
        '''Remove a user and all their embeddings. Returns False if not found.'''
        self._conn.execute('DELETE FROM embeddings WHERE user_id=?', (user_id,))
        cur = self._conn.execute('DELETE FROM users WHERE id=?', (user_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def all_users(self) -> list[dict]:
        '''
        Returns a list of all users, each as: {id, name, password_hash, chat_id, embedding_count}
        '''
        cur = self._conn.execute('''
            SELECT u.id, u.name, u.password_hash, u.chat_id,
                   COUNT(e.rowid) AS embedding_count
            FROM users u
            LEFT JOIN embeddings e ON e.user_id = u.id
            GROUP BY u.id
            ORDER BY u.id
        ''')
        return [
            {
                'id':              r[0],
                'name':            r[1],
                'password_hash':   r[2],
                'chat_id':         r[3],
                'embedding_count': r[4],
            }
            for r in cur.fetchall()
        ]

    def get_user(self, name: str, plaintext: str) -> dict | None:
        row = self._conn.execute(
            'SELECT id, name, chat_id FROM users WHERE name=? AND password_hash=?',
            (name, self._hash(plaintext))
        ).fetchone()

        if not row:
            return None
        return {'id': row[0], 'name': row[1], 'chat_id': row[2]}

    def get_by_id(self, user_id: str) -> dict | None:
        '''
        Return everything about a single user by their id:
            {id, name, password_hash, chat_id, embedding_count, embeddings}
        embeddings is a list of float64 np.ndarray vectors.
        Returns None if the user does not exist.
        '''
        row = self._conn.execute(
            'SELECT id, name, password_hash, chat_id FROM users WHERE id=?', (user_id,)
        ).fetchone()

        if not row:
            return None

        emb_rows = self._conn.execute(
            'SELECT vector FROM embeddings WHERE user_id=?', (user_id,)
        ).fetchall()
        embeddings = [np.frombuffer(r[0], dtype=np.float64) for r in emb_rows]

        return {
            'id':              row[0],
            'name':            row[1],
            'password_hash':   row[2],
            'chat_id':         row[3],
            'embedding_count': len(embeddings),
            'embeddings':      embeddings,
        }

    #! ::::: ::::: Password ::::: ::::: ::::: ::::: ::::: ::::: ::::: :::::

    def verify_password(self, user_id: str, plaintext: str) -> bool:
        row = self._conn.execute('SELECT password_hash FROM users WHERE id=?', (user_id,)).fetchone()
        if not row:
            return False
        return row[0] == self._hash(plaintext)

    def reset_password(self, user_id: str, new_password: str) -> bool:
        cur = self._conn.execute(
            'UPDATE users SET password_hash=? WHERE id=?',
            (self._hash(new_password), user_id)
        )
        self._conn.commit()
        return cur.rowcount > 0
