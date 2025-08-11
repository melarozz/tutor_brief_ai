import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List
from .segment import Segment

def cluster_speakers(segments: List[Segment], min_speakers=1, max_speakers=8):
    embeddings = np.vstack([s.embedding for s in segments])
    best_k = min(max_speakers, len(segments))
    if best_k <= 1:
        labels = np.zeros(len(segments), dtype=int)
    else:
        k = min(best_k, max(1, int(math.sqrt(len(segments)))))
        k = max(min_speakers, k)
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = cl.fit_predict(embeddings)
    return labels

def merge_adjacent_segments(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg.speaker == prev.speaker and abs(seg.start - prev.end) < 0.5:
            prev.end = seg.end
            prev.text = (prev.text + " " + seg.text).strip()
        else:
            out.append(seg)
    return out
