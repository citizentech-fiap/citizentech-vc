import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from citizentech_crack_module import CrackDetector, reencode_h264

st.set_page_config(page_title="CitizenTech · Inspeção visual", layout="wide")
st.title("CitizenTech · Triagem visual de fissuras em concreto")
st.caption("Apoio à triagem de locais para inspeção. A presença de fissura na imagem, isoladamente, "
           "não determina a gravidade do dano nem a segurança estrutural.")

ckpt = st.sidebar.text_input("Checkpoint do modelo", "crack_model.pt")
if not Path(ckpt).exists():
    st.error(f"Arquivo '{ckpt}' não encontrado. Coloque o crack_model.pt na mesma pasta do app.")
    st.stop()


@st.cache_resource
def load(path):
    return CrackDetector(path)


det = load(ckpt)
det.threshold = st.sidebar.slider("Limiar de confiança", 0.05, 0.95, float(det.threshold), 0.01)
st.sidebar.write(f"Modelo: {det.arch} · dispositivo: {det.device}")

tab_img, tab_vid = st.tabs(["Imagem", "Vídeo / sequência"])

with tab_img:
    up = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"], key="img")
    if up:
        img = Image.open(up).convert("RGB")
        res = det.predict_image(img)
        c1, c2 = st.columns([2, 1])
        arr = np.array(img)
        if min(arr.shape[:2]) > 300:       # imagem grande: varre em blocos e marca os trechos suspeitos
            bgr = arr[:, :, ::-1].copy()
            dets = det.scan_frame(bgr)
            c1.image(det.annotate_frame(bgr, dets)[:, :, ::-1], caption="Trechos suspeitos (caixas vermelhas)")
            c2.metric("Trechos para inspeção", f"{sum(d['flag'] for d in dets)} de {len(dets)}")
            c2.metric("Maior confiança", f"{max(d['prob'] for d in dets):.1%}")
        else:
            c1.image(img, caption="Imagem enviada")
            c2.metric("Classe", res["classe"])
            c2.metric("Confiança", f"{res['confianca']:.1%}")
            c2.metric("p(fissura)", f"{res['prob_fissura']:.1%}")
            (c2.warning if res["inspecionar"] else c2.success)(
                "Possível necessidade de inspeção" if res["inspecionar"] else "Sem indício de fissura")

with tab_vid:
    upv = st.file_uploader("Envie um vídeo curto", type=["mp4", "avi", "mov"], key="vid")
    stride = st.number_input("Processar 1 a cada N quadros", 1, 30, 1)
    if upv and st.button("Processar vídeo"):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            src.write_bytes(upv.read())
            raw, web = Path(tmp) / "out_raw.mp4", Path(tmp) / "out_web.mp4"
            with st.spinner("Processando..."):
                stats = det.process_video(src, raw, frame_stride=int(stride))
            final = web if reencode_h264(raw, web) else raw
            st.video(final.read_bytes())
        c = st.columns(4)
        c[0].metric("Quadros", stats["frames"])
        c[1].metric("FPS do pipeline", f"{stats['fps_pipeline']:.1f}")
        c[2].metric("ms de inferência/quadro", f"{stats['ms_inferencia_por_quadro']:.0f}")
        c[3].metric("Resolução", stats["resolucao"])
        st.line_chart({"trechos suspeitos por quadro": [r["suspeitos"] for r in stats["log"]]})
