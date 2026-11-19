"""Módulo de visão computacional do CitizenTech: triagem de fissuras em concreto.

Apoia a TRIAGEM de locais para inspeção. A presença de fissura na imagem, isoladamente,
não permite determinar a gravidade do dano nem a segurança estrutural da edificação.
"""
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _build(arch):
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    elif arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError(arch)
    return m


class CrackDetector:
    def __init__(self, ckpt_path, device=None, threshold=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(str(ckpt_path), map_location="cpu")
        self.arch, self.size = ck["arch"], int(ck["img_size"])
        self.threshold = float(threshold if threshold is not None else 0.60)
        self.model = _build(self.arch)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def predict_rgb_batch(self, arrays, batch_size=64):
        """arrays: lista de imagens RGB uint8 (H, W, 3). Devolve a probabilidade de fissura de cada uma."""
        out = []
        for i in range(0, len(arrays), batch_size):
            chunk = [cv2.resize(np.ascontiguousarray(a), (self.size, self.size)) for a in arrays[i:i + batch_size]]
            x = torch.from_numpy(np.stack(chunk)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            out.append(torch.sigmoid(self.model(x).squeeze(1)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_image(self, img):
        """img: caminho, PIL.Image ou array RGB. Devolve classe, confiança e se deve ser inspecionada."""
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        arr = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)
        p = float(self.predict_rgb_batch([arr])[0])
        flag = p >= self.threshold
        return {"prob_fissura": p, "classe": "com fissura" if flag else "sem fissura",
                "confianca": p if flag else 1.0 - p, "inspecionar": bool(flag), "limiar": self.threshold}

    def scan_frame(self, frame_bgr, tile=227, stride=None):
        """Varre o quadro em blocos tile x tile e classifica cada bloco (localização aproximada)."""
        h, w = frame_bgr.shape[:2]
        tile = int(min(tile, h, w))
        stride = int(stride or tile)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pos = [(x, y) for y in range(0, h - tile + 1, stride) for x in range(0, w - tile + 1, stride)]
        probs = self.predict_rgb_batch([rgb[y:y + tile, x:x + tile] for x, y in pos])
        return [{"x": x, "y": y, "size": tile, "prob": float(p), "flag": bool(p >= self.threshold)}
                for (x, y), p in zip(pos, probs)]

    def annotate_frame(self, frame_bgr, dets, extra_text=""):
        out = frame_bgr.copy()
        flagged = [d for d in dets if d["flag"]]
        for d in flagged:
            x, y, s = d["x"], d["y"], d["size"]
            cv2.rectangle(out, (x, y), (x + s, y + s), (0, 0, 255), 3)
            cv2.putText(out, f"{d['prob']:.0%}", (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        bar = f"Triagem: {len(flagged)}/{len(dets)} trechos para inspecao | limiar={self.threshold:.2f} {extra_text}"
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, bar, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def process_video(self, src, dst, tile=227, frame_stride=1, max_frames=None, fps_out=None,
                      progress_callback=None):
        """Processa um vídeo (ou sequência) e grava o vídeo anotado. Devolve estatísticas de desempenho."""
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise IOError(f"não foi possível abrir {src}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_out or fps_in, (W, H))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"não foi possível criar o vídeo de saída {dst}")
        i, n_proc, t_inf, last, log = 0, 0, 0.0, [], []
        t_all = time.perf_counter()
        while True:
            ok, fr = cap.read()
            if not ok or (max_frames is not None and i >= max_frames):
                break
            if i % frame_stride == 0:
                t0 = time.perf_counter()
                last = self.scan_frame(fr, tile)
                self._sync()
                t_inf += time.perf_counter() - t0
                n_proc += 1
            log.append({"frame": i, "blocos": len(last), "suspeitos": sum(d["flag"] for d in last),
                        "prob_max": max([d["prob"] for d in last], default=0.0)})
            writer.write(self.annotate_frame(fr, last))
            i += 1
            if progress_callback is not None:
                progress_callback(i)
        cap.release()
        writer.release()
        total = time.perf_counter() - t_all
        return {"frames": i, "frames_processados": n_proc, "tempo_total_s": total,
                "fps_pipeline": i / max(total, 1e-9),
                "ms_inferencia_por_quadro": 1000 * t_inf / max(n_proc, 1),
                "fps_inferencia": n_proc / max(t_inf, 1e-9),
                "resolucao": f"{W}x{H}", "dispositivo": str(self.device), "log": log}


def reencode_h264(src, dst):
    """Converte o mp4 para H.264 (toca em navegadores). Precisa do ffmpeg; devolve True se deu certo."""
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", str(dst)])
        return r.returncode == 0
    except FileNotFoundError:
        return False
