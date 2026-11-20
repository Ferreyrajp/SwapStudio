"""Reemplazo minimo de `decord` basado en OpenCV.

El preprocesado oficial de Wan2.2 importa `decord`, que no tiene wheels para muchas
combinaciones de Windows/Python. Solo usa: VideoReader(path), len(), get_avg_fps(),
get_frame_timestamp(-1) y get_batch(idxs).asnumpy(). Esta carpeta solo se agrega al
PYTHONPATH si el decord real no esta instalado.
"""
import cv2
import numpy as np


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class VideoReader:
    def __init__(self, path, *args, **kwargs):
        self._path = str(path)
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir el video: {path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self):
        return self._n

    def get_avg_fps(self):
        return self._fps

    def get_frame_timestamp(self, idx):
        i = idx if idx >= 0 else self._n + idx
        return np.array([i / self._fps, (i + 1) / self._fps])

    def get_batch(self, idxs):
        idxs = [int(i) for i in idxs]
        wanted = set(idxs)
        last = max(wanted) if wanted else -1
        keep = {}
        cap = cv2.VideoCapture(self._path)
        i = 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                keep[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        if not keep:
            raise RuntimeError("No se pudo leer ningun frame")
        fallback = keep[max(keep)]
        return _Batch(np.stack([keep.get(j, fallback) for j in idxs]))
