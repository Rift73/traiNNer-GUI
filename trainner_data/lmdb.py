"""Share read-only LMDB environments within a loader process, never across forks."""
from __future__ import annotations

import os
from threading import Lock
from weakref import WeakSet, WeakValueDictionary


class _Environment:
    def __init__(self, folder: str, options: dict):
        self.options = options.copy()
        import lmdb
        self.env = lmdb.open(folder, **options)


_readers = WeakSet()
_clients: WeakValueDictionary[str, _Environment] = WeakValueDictionary()
_lock = Lock()


def _after_fork() -> None:
    global _clients, _lock
    for reader in tuple(_readers):
        reader.clients.clear()
        reader.pid = None
    _clients = WeakValueDictionary()
    _lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class PairedLmdbReader:
    def __init__(self, options: dict):
        self.options = {"readonly": True, "create": False, "readahead": False, "lock": False}
        self.options.update({k:v for k,v in options.items() if k not in ("type","db_paths","client_keys")})
        if self.options.get("readonly", True) is not True:
            raise ValueError("Training datasets require read-only LMDB access.")
        self.pid: int | None = None
        self.clients: dict[str, _Environment] = {}
        _readers.add(self)

    def __getstate__(self):
        return {"options":self.options, "pid":None, "clients":{}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        _readers.add(self)

    def read(self, folder: str, key: str) -> bytes:
        pid = os.getpid()
        if self.pid != pid:
            self.clients.clear()
            self.pid = pid
        client = self.clients.get(folder)
        if client is None:
            real = os.path.realpath(folder)
            with _lock:
                client = _clients.get(real)
                if client is None:
                    client = _Environment(real, self.options)
                    _clients[real] = client
                elif self.options != client.options:
                    raise ValueError(f"Datasets sharing {real} must use identical LMDB reader options.")
                self.clients[folder] = client
        with client.env.begin() as txn:
            value = txn.get(key.encode("ascii"))
        if value is None:
            raise FileNotFoundError(f"Missing LMDB record {key!r} in {folder}.")
        return value
