"""Thread-safe BK-tree index for 256-bit pHash Hamming searches."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from phash_match import hamming_distance


class _Node:
    def __init__(self, value: str, artwork_ids: set[str] | None = None) -> None:
        self.value = value
        self.artwork_ids = artwork_ids or set()
        self.children: dict[int, _Node] = {}


class PHashIndex:
    def __init__(self, persistence_path: str | Path | None = None) -> None:
        self.path = Path(persistence_path) if persistence_path else None
        self._entries: dict[str, str] = {}
        self._root: _Node | None = None
        self._lock = RLock()
        if self.path and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for artwork_id, value in payload.get("entries", {}).items():
                self._add_no_lock(artwork_id, value)

    def _add_no_lock(self, artwork_id: str, value: str) -> None:
        self._entries[artwork_id] = value
        if self._root is None:
            self._root = _Node(value, {artwork_id})
            return
        node = self._root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.artwork_ids.add(artwork_id)
                return
            if distance not in node.children:
                node.children[distance] = _Node(value, {artwork_id})
                return
            node = node.children[distance]

    def upsert(self, artwork_id: str, value: str) -> None:
        # Rebuilding keeps replacement/removal semantics correct and is cheap
        # for the PoC; the query path is logarithmic-ish through the BK-tree.
        hamming_distance(value, value)
        with self._lock:
            self._entries[artwork_id] = value
            entries = dict(self._entries)
            self._root = None
            self._entries = {}
            for key, item in entries.items():
                self._add_no_lock(key, item)
            self._persist_no_lock()

    def _persist_no_lock(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"version": 1, "entries": self._entries}, indent=2), encoding="utf-8")

    def search(self, value: str, max_distance: int = 20, limit: int = 10) -> list[dict]:
        hamming_distance(value, value)
        with self._lock:
            results: list[tuple[int, str]] = []

            def visit(node: _Node) -> None:
                distance = hamming_distance(value, node.value)
                if distance <= max_distance:
                    results.extend((distance, artwork_id) for artwork_id in node.artwork_ids)
                low, high = distance - max_distance, distance + max_distance
                for edge, child in node.children.items():
                    if low <= edge <= high:
                        visit(child)

            if self._root:
                visit(self._root)
            results.sort(key=lambda item: (item[0], item[1]))
            return [
                {"artworkId": artwork_id, "perceptualHash": self._entries[artwork_id], "distance": distance}
                for distance, artwork_id in results[:limit]
            ]

    def __len__(self) -> int:
        return len(self._entries)
