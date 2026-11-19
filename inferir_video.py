import time
from pathlib import Path

import cv2

from citizentech_crack_module import CrackDetector

project_dir = Path(__file__).resolve().parent
src = project_dir / "video_model.mp4"
checkpoint = project_dir / "crack_model.pt"
dst = project_dir / "outputs" / "video_model_inferido_metade.mp4"

if not src.is_file():
    raise FileNotFoundError(f"Vídeo de entrada não encontrado: {src}")
if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint do modelo não encontrado: {checkpoint}")

cap = cv2.VideoCapture(str(src))
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {src}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

if total_frames <= 0:
    raise RuntimeError("Não foi possível determinar a quantidade de quadros do vídeo.")

# Arredonda para cima para garantir pelo menos metade, inclusive em vídeos ímpares.
half_frames = (total_frames + 1) // 2

dst.parent.mkdir(parents=True, exist_ok=True)

print(f"Carregando modelo e preparando inferência de {half_frames:,} quadros...")
det = CrackDetector(checkpoint)
started_at = time.perf_counter()
progress_state = {"last_update": 0.0}


def show_progress(frames_done):
    now = time.perf_counter()
    if frames_done < half_frames and now - progress_state["last_update"] < 0.25:
        return
    progress_state["last_update"] = now

    fraction = min(frames_done / half_frames, 1.0)
    percent = fraction * 100
    bar_width = 32
    filled = int(bar_width * fraction)
    elapsed = now - started_at
    remaining = elapsed / frames_done * (half_frames - frames_done) if frames_done else 0
    bar = f"[{'#' * filled}{'-' * (bar_width - filled)}]"
    print(
        f"\r{bar} {percent:5.1f}% | {frames_done:,}/{half_frames:,} quadros"
        f" | decorrido {elapsed:.0f}s | restante ~{remaining:.0f}s",
        end="",
        flush=True,
    )


print(f"Dispositivo: {det.device} | Vídeo: {src.name}")
stats = det.process_video(
    src,
    dst,
    frame_stride=1,       # inferência em todos os quadros dessa parte
    max_frames=half_frames,
    progress_callback=show_progress,
)
print()

print(f"Vídeo salvo em: {dst}")
print(f"Quadros processados: {stats['frames_processados']:,} de {total_frames:,} do vídeo")
print(f"Tempo: {stats['tempo_total_s']:.1f}s | FPS do pipeline: {stats['fps_pipeline']:.1f}")