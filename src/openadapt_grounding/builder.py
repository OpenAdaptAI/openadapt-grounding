"""Registry builder with temporal clustering."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from openadapt_grounding.types import Bounds, Element, RegistryEntry


class Registry:
    """Collection of stable UI elements."""

    def __init__(self, entries: List[RegistryEntry]):
        self.entries = entries
        self._by_text: Dict[str, RegistryEntry] = {}
        self._by_uid: Dict[str, RegistryEntry] = {}

        for entry in entries:
            self._by_uid[entry.uid] = entry
            if entry.text:
                # Index by lowercase text for case-insensitive lookup
                self._by_text[entry.text.lower()] = entry

    def get_by_text(self, text: str) -> Optional[RegistryEntry]:
        """Find entry by text (case-insensitive)."""
        return self._by_text.get(text.lower())

    def get_by_uid(self, uid: str) -> Optional[RegistryEntry]:
        """Find entry by UID."""
        return self._by_uid.get(uid)

    def find_similar_text(self, query: str) -> Optional[RegistryEntry]:
        """Find entry with text containing the query (case-insensitive)."""
        query_lower = query.lower()
        for text, entry in self._by_text.items():
            if query_lower in text or text in query_lower:
                return entry
        return None

    def save(self, path: Union[str, Path]) -> None:
        """Save registry to JSON file."""
        data = {"entries": [e.to_dict() for e in self.entries]}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Registry":
        """Load registry from JSON file."""
        data = json.loads(Path(path).read_text())
        entries = [RegistryEntry.from_dict(e) for e in data["entries"]]
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


class RegistryBuilder:
    """Build a registry from multiple frames of detections."""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: Min IoU to consider two elements the same (for non-text elements)
        """
        self.iou_threshold = iou_threshold
        self.frames: List[List[Element]] = []

    def add_frame(self, elements: List[Element]) -> None:
        """Add a frame's worth of element detections."""
        self.frames.append(elements)

    def build(self, min_stability: float = 0.5) -> Registry:
        """
        Build registry, keeping only stable elements.

        Args:
            min_stability: Min fraction of frames an element must appear in (0.0-1.0)

        Returns:
            Registry of stable elements
        """
        if not self.frames:
            return Registry([])

        # Cluster elements across frames
        clusters = self._cluster_elements()

        # Filter by stability
        total_frames = len(self.frames)
        stable_entries = []

        for cluster_id, elements in clusters.items():
            stability = len(elements) / total_frames
            if stability >= min_stability:
                entry = self._make_entry(cluster_id, elements, total_frames)
                stable_entries.append(entry)

        return Registry(stable_entries)

    def _cluster_elements(self) -> Dict[str, List[Element]]:
        """
        Group elements across frames into clusters.

        Strategy:
        1. Elements with identical text -> same cluster
        2. Elements without text -> cluster by IoU overlap
        """
        clusters: Dict[str, List[Element]] = defaultdict(list)

        # First pass: cluster by text
        text_to_cluster: Dict[str, str] = {}

        for frame_idx, frame in enumerate(self.frames):
            for elem in frame:
                if elem.text:
                    text_key = elem.text.lower().strip()
                    if text_key not in text_to_cluster:
                        # New text, create cluster
                        cluster_id = self._make_cluster_id(elem.text, elem.element_type)
                        text_to_cluster[text_key] = cluster_id
                    clusters[text_to_cluster[text_key]].append(elem)

        # Second pass: cluster non-text elements by spatial overlap
        # For simplicity, we use a greedy approach
        non_text_elements: List[tuple] = []  # (frame_idx, element)
        for frame_idx, frame in enumerate(self.frames):
            for elem in frame:
                if not elem.text:
                    non_text_elements.append((frame_idx, elem))

        # Greedy clustering for non-text elements
        spatial_clusters: List[List[Element]] = []
        used = set()

        for i, (frame_i, elem_i) in enumerate(non_text_elements):
            if i in used:
                continue

            cluster = [elem_i]
            used.add(i)

            for j, (frame_j, elem_j) in enumerate(non_text_elements):
                if j in used or frame_j == frame_i:
                    continue
                # Check if overlaps with any element in cluster
                if any(elem_j.iou(c) >= self.iou_threshold for c in cluster):
                    cluster.append(elem_j)
                    used.add(j)

            if len(cluster) > 1:  # Only keep clusters with multiple detections
                spatial_clusters.append(cluster)

        # Add spatial clusters
        for idx, cluster in enumerate(spatial_clusters):
            cluster_id = f"spatial_{idx}"
            clusters[cluster_id] = cluster

        return clusters

    def _make_cluster_id(self, text: Optional[str], element_type: str) -> str:
        """Generate a deterministic cluster ID."""
        key = f"{text or ''}:{element_type}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _make_entry(
        self, cluster_id: str, elements: List[Element], total_frames: int
    ) -> RegistryEntry:
        """Create a registry entry from a cluster of elements."""
        # Use most common text
        texts = [e.text for e in elements if e.text]
        text = max(set(texts), key=texts.count) if texts else None

        # Use most common type
        types = [e.element_type for e in elements]
        element_type = max(set(types), key=types.count)

        # Average bounds
        avg_bounds = self._average_bounds([e.bounds for e in elements])

        return RegistryEntry(
            uid=cluster_id,
            text=text,
            bounds=avg_bounds,
            element_type=element_type,
            detection_count=len(elements),
            total_frames=total_frames,
        )

    def _average_bounds(self, bounds_list: List[Bounds]) -> Bounds:
        """Compute average bounds."""
        n = len(bounds_list)
        if n == 0:
            return (0.0, 0.0, 0.0, 0.0)

        avg_x = sum(b[0] for b in bounds_list) / n
        avg_y = sum(b[1] for b in bounds_list) / n
        avg_w = sum(b[2] for b in bounds_list) / n
        avg_h = sum(b[3] for b in bounds_list) / n

        return (avg_x, avg_y, avg_w, avg_h)
