"""
Whisper · Asistente conversacional + Explorador Encoder/Decoder (Streamlit)

Instalación:
    pip install streamlit openai-whisper matplotlib pandas imageio-ffmpeg
    (requiere ffmpeg instalado en el sistema)

Ejecución:
    streamlit run app_whisper_streamlit.py
"""
import os
import subprocess
import tempfile
import time
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import whisper
from whisper.model import MultiHeadAttention

# Desactivamos SDPA para que cada capa de atención devuelva la matriz QK^T
# (necesaria para visualizar la atención cruzada).
if hasattr(MultiHeadAttention, "use_sdpa"):
    MultiHeadAttention.use_sdpa = False

st.set_page_config(page_title="Whisper · Encoder/Decoder", page_icon="🎙️", layout="wide")


def cargar_audio(ruta: str, sr: int = 16000) -> np.ndarray:
    """Igual que whisper.load_audio, pero si no hay ffmpeg en el PATH usa el
    binario que trae la librería imageio-ffmpeg (pip install imageio-ffmpeg)."""
    try:
        return whisper.load_audio(ruta, sr)
    except FileNotFoundError:
        try:
            import imageio_ffmpeg
        except ImportError:
            st.error("No se encontró ffmpeg. Instala el respaldo con: "
                     "python -m pip install imageio-ffmpeg")
            st.stop()
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-threads", "0",
               "-i", ruta, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
               "-ar", str(sr), "-"]
        salida = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(salida, np.int16).flatten().astype(np.float32) / 32768.0


@st.cache_resource(show_spinner="Cargando modelo Whisper…")
def cargar_modelo(nombre: str):
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(nombre, device=dispositivo)


# ================================================================ ASISTENTE (CHATBOT)
EXTENSIONES = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]
NOMBRE_TAREA = {"transcribe": "transcribir", "translate": "traducir al inglés"}


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def detectar_intencion(texto: str):
    """Clasifica el mensaje por palabras clave: 'translate', 'transcribe' o None."""
    t = sin_tildes(texto)
    if any(p in t for p in ["traduc", "translat", "ingles", "english", "otro idioma"]):
        return "translate"
    if any(p in t for p in ["transcrib", "transcripc", "texto", "escrib", "dictad",
                            "que dice", "subtitul", "acta"]):
        return "transcribe"
    return None


def decir(texto: str):
    st.session_state.chat.append({"role": "assistant", "content": texto})


def procesar_audio(modelo, datos: bytes, nombre: str, tarea: str):
    sufijo = os.path.splitext(nombre)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
        tmp.write(datos)
        ruta = tmp.name
    try:
        audio = cargar_audio(ruta)
    finally:
        os.unlink(ruta)
    t0 = time.time()
    r = modelo.transcribe(audio, task=tarea, fp16=False)
    return r["text"].strip(), r.get("language", "?"), len(audio) / 16000, time.time() - t0


def responder_con_resultado(modelo, datos, nombre, tarea):
    with st.spinner("Procesando audio…"):
        texto, idioma, dur, seg = procesar_audio(modelo, datos, nombre, tarea)
    etiqueta = "Transcripción" if tarea == "transcribe" else "Traducción al inglés"
    decir(
        f"**{etiqueta}** — idioma detectado: `{idioma}`, {dur:.1f} s de audio, "
        f"procesado en {seg:.1f} s:\n\n> {texto or '(no se detectó voz en el audio)'}\n\n"
        f"¿Quieres enviar otro audio? Sigo en modo **{NOMBRE_TAREA[tarea]}**, "
        f"o dime si prefieres cambiar de tarea."
    )


def elegir_tarea(modelo, tarea):
    ss = st.session_state
    ss.tarea_chat = tarea
    if ss.audio_pendiente:
        datos, nombre = ss.audio_pendiente
        ss.audio_pendiente = None
        decir(f"Perfecto, voy a **{NOMBRE_TAREA[tarea]}** el audio que me enviaste.")
        responder_con_resultado(modelo, datos, nombre, tarea)
    else:
        nota = (" Ten en cuenta que Whisper traduce **únicamente hacia el inglés**."
                if tarea == "translate" else "")
        decir(f"Perfecto, vamos a **{NOMBRE_TAREA[tarea]}**.{nota} Envíame el audio: "
              "grábalo con el micrófono de abajo o adjúntalo con el clip 📎 de la barra de mensajes.")


def recibir_audio(modelo, datos, nombre):
    ss = st.session_state
    ss.chat.append({"role": "user", "content": f"🎧 {nombre}", "audio": datos})
    if ss.tarea_chat is None:
        ss.audio_pendiente = (datos, nombre)
        decir("¡Recibí tu audio! ¿Quieres que lo **transcriba** o que lo **traduzca al inglés**?")
    else:
        responder_con_resultado(modelo, datos, nombre, ss.tarea_chat)


def asistente(modelo):
    ss = st.session_state
    if "chat" not in ss:
        ss.chat = [{"role": "assistant", "content":
                    "¡Hola! Soy tu asistente de audio. Puedo **transcribir** un audio "
                    "(pasarlo a texto en su idioma original) o **traducirlo al inglés**. "
                    "¿Qué te gustaría hacer?"}]
        ss.tarea_chat = None
        ss.audio_pendiente = None
        ss.mic_n = 0

    st.title("💬 Asistente de audio con Whisper")
    st.caption("Cuéntame qué necesitas y envíame el audio por este mismo chat.")

    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("audio"):
                st.audio(m["audio"])

    if ss.tarea_chat is None:
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("📝 Transcribir"):
            elegir_tarea(modelo, "transcribe")
            st.rerun()
        if b2.button("🌐 Traducir al inglés"):
            elegir_tarea(modelo, "translate")
            st.rerun()
    elif hasattr(st, "audio_input"):
        with st.chat_message("assistant"):
            mic = st.audio_input("Graba tu audio aquí", key=f"mic_{ss.mic_n}")
        if mic is not None:
            ss.mic_n += 1  # nueva clave: el grabador queda limpio para el siguiente audio
            recibir_audio(modelo, mic.getvalue(), "grabacion.wav")
            st.rerun()

    try:
        entrada = st.chat_input("Escribe un mensaje o adjunta un audio…",
                                accept_file=True, file_type=EXTENSIONES)
    except TypeError:  # versiones antiguas de Streamlit
        entrada = st.chat_input("Escribe un mensaje…")

    if entrada:
        texto = entrada if isinstance(entrada, str) else (entrada.text or "")
        archivos = [] if isinstance(entrada, str) else list(entrada.files)
        if texto:
            ss.chat.append({"role": "user", "content": texto})
            intencion = detectar_intencion(texto)
            if intencion and archivos:
                ss.tarea_chat = intencion
            elif intencion:
                elegir_tarea(modelo, intencion)
            elif not archivos:
                decir("No logré identificar si quieres **transcribir** o **traducir**. "
                      "¿Me lo confirmas? Puedes escribirlo o usar los botones.")
        for f in archivos:
            recibir_audio(modelo, f.getvalue(), f.name)
        st.rerun()


# ---------------------------------------------------------------- Barra lateral
st.sidebar.header("Configuración")
modo = st.sidebar.radio("Modo", ["💬 Asistente", "🔬 Explorador"], index=0)
nombre_modelo = st.sidebar.selectbox("Modelo", ["base", "tiny", "small"], index=0)
idioma_sel = st.sidebar.selectbox("Idioma", ["Auto (detectar)", "es", "en", "pt", "fr"])
tarea = st.sidebar.selectbox("Tarea", ["transcribe", "translate"])

modelo = cargar_modelo(nombre_modelo)
dims = modelo.dims

if modo.startswith("💬"):
    if st.sidebar.button("🗑️ Nueva conversación"):
        st.session_state.pop("chat", None)
        st.rerun()
    asistente(modelo)
    st.stop()

st.title("🎙️ Whisper · Explorador Encoder/Decoder")
st.caption(
    f"Modelo **{nombre_modelo}** en **{modelo.device}** · encoder: {dims.n_audio_layer} capas, "
    f"{dims.n_audio_head} cabezas · entrada mel ({dims.n_mels}, 3000) → features "
    f"({dims.n_audio_ctx} × {dims.n_audio_state}) · decoder: {dims.n_text_layer} capas · "
    f"vocabulario {dims.n_vocab} tokens · contexto {dims.n_text_ctx} tokens"
)

# ---------------------------------------------------------------- Entrada de audio
st.subheader("Sube o graba tu audio")
c1, c2 = st.columns(2)
archivo = c1.file_uploader("Archivo de audio", type=["wav", "mp3", "m4a", "flac", "ogg", "webm"])
grabacion = c2.audio_input("Grabar con micrófono") if hasattr(st, "audio_input") else None
fuente = grabacion or archivo

if fuente is None:
    st.info("Sube un archivo o graba un audio para comenzar.")
    st.stop()

st.audio(fuente)

if not st.button("Transcribir y analizar", type="primary"):
    st.stop()

# Guardamos el audio en un archivo temporal para que ffmpeg lo lea
sufijo = os.path.splitext(getattr(fuente, "name", "audio.wav"))[1] or ".wav"
with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
    tmp.write(fuente.getvalue())
    ruta = tmp.name

t0 = time.time()
with st.spinner("Procesando audio…"):
    # ---- 1. Entrada: audio -> espectrograma Mel (80, 3000)
    audio = cargar_audio(ruta)
    duracion = len(audio) / whisper.audio.SAMPLE_RATE
    audio_30s = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_30s, n_mels=dims.n_mels).to(modelo.device)

    # ---- Detección de idioma
    _, probs = modelo.detect_language(mel)
    idioma = max(probs, key=probs.get) if idioma_sel.startswith("Auto") else idioma_sel

    # ---- 2. Encoder: audio -> 1500 representaciones de 512 dimensiones
    with torch.no_grad():
        audio_features = modelo.embed_audio(mel.unsqueeze(0))  # (1, 1500, 512)

    # ---- 3. Decoder: generación autoregresiva greedy (T = 0)
    tok = whisper.tokenizer.get_tokenizer(
        modelo.is_multilingual, num_languages=modelo.num_languages,
        language=idioma, task=tarea,
    )
    tokens = list(tok.sot_sequence_including_notimestamps)
    n_inicial = len(tokens)
    logprobs = []
    confianza = []  # probabilidad del token elegido y la segunda mejor opción
    with torch.no_grad():
        for _ in range(dims.n_text_ctx // 2):
            x = torch.tensor([tokens], device=modelo.device)
            logits = modelo.decoder(x, audio_features)[0, -1].float()
            logits[tok.timestamp_begin:] = -np.inf  # sin marcas de tiempo
            lp = torch.log_softmax(logits, dim=-1)
            siguiente = int(logits.argmax())
            if siguiente == tok.eot:
                break
            tokens.append(siguiente)
            logprobs.append(float(lp[siguiente]))
            top2 = torch.topk(lp, 2)
            confianza.append({
                "token elegido": tok.decode([siguiente]),
                "probabilidad": float(top2.values[0].exp()),
                "segunda opción": tok.decode([int(top2.indices[1])]),
                "prob. segunda": float(top2.values[1].exp()),
            })

    # ---- 4. Atención cruzada: Q (decoder) · K (encoder) -> pesos sobre el audio
    matrices_qk = []
    def gancho_cross(m, i, o):
        if o[1] is not None:
            matrices_qk.append(o[1].detach().float())
    ganchos = [b.cross_attn.register_forward_hook(gancho_cross) for b in modelo.decoder.blocks]
    with torch.no_grad():
        modelo.decoder(torch.tensor([tokens], device=modelo.device), audio_features)
    for g in ganchos:
        g.remove()

tiempo = time.time() - t0
os.unlink(ruta)

generados = tokens[n_inicial:]
texto = tok.decode(generados).strip()
piezas = [tok.decode([t]) for t in generados]

# ================================================================ 1. ENTRADAS
st.header("1 · Entradas")
m1, m2, m3 = st.columns(3)
m1.metric("Duración del audio", f"{duracion:.2f} s")
m2.metric("Muestreo", f"{whisper.audio.SAMPLE_RATE} Hz")
m3.metric("Entrada al encoder", f"{tuple(mel.shape)}")

fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(mel.cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
ax.axvline(duracion * 100, color="white", ls="--", lw=1)
ax.set_xlabel("Frames (1 frame = 10 ms)")
ax.set_ylabel("Banda Mel")
ax.set_title("Espectrograma Mel de 80 bandas (la línea marca el fin del audio real)")
st.pyplot(fig)

# ================================================================ 2. SALIDAS
st.header("2 · Salidas")
st.success(texto or "(sin texto)")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Idioma detectado", idioma, f"{probs.get(idioma, 0) * 100:.1f} %")
s2.metric("Tokens generados", len(generados))
s3.metric("Log-prob media", f"{np.mean(logprobs):.4f}" if logprobs else "—")
s4.metric("Tiempo de análisis", f"{tiempo:.2f} s")

top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
st.bar_chart(pd.DataFrame({"probabilidad": [p for _, p in top]}, index=[k for k, _ in top]))

st.markdown("**Tokens generados (subpalabras):**")
st.markdown(" ".join(f"`{p}`" for p in piezas))
st.caption("Secuencia inicial: " + " ".join(tok.decode([t]) for t in tokens[:n_inicial]))

st.subheader("Confianza del decoder en cada token")
if confianza:
    df_conf = pd.DataFrame(confianza)
    colores = ["#2e9e6b" if p >= 0.8 else "#e0a526" if p >= 0.5 else "#d64545"
               for p in df_conf["probabilidad"]]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.bar(range(len(df_conf)), df_conf["probabilidad"] * 100, color=colores)
    ax.set_xticks(range(len(df_conf)), df_conf["token elegido"], rotation=60, ha="right")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.set_title("Probabilidad con la que el modelo eligió cada token (verde ≥ 80 %, "
                 "amarillo 50–80 %, rojo < 50 %)")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "En cada paso el decoder calcula la probabilidad de los 51.865 tokens del vocabulario "
        "y, con la estrategia greedy (T = 0), elige el más probable. Una barra alta indica que el "
        "modelo estaba seguro; una barra baja indica duda entre varias opciones, algo típico "
        "en nombres propios poco comunes."
    )
    st.dataframe(
        df_conf.style.format({"probabilidad": "{:.1%}", "prob. segunda": "{:.1%}"}),
        hide_index=True,
    )

# ================================================================ 3. Q, K, V
st.header("3 · Interpretación de Q, K y V")
st.latex(r"\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")

d, h = dims.n_audio_state, dims.n_audio_head
dk, n_a, n_t = d // h, audio_features.shape[1], len(tokens)
st.table(pd.DataFrame([
    {"Atención": "Self-attention del encoder", "Q viene de": "audio", "K y V vienen de": "audio",
     "Forma Q": f"({n_a}, {d})", "Forma K/V": f"({n_a}, {d})"},
    {"Atención": "Self-attention del decoder (con máscara)", "Q viene de": "tokens",
     "K y V vienen de": "tokens anteriores", "Forma Q": f"({n_t}, {d})", "Forma K/V": f"({n_t}, {d})"},
    {"Atención": "Cross-attention", "Q viene de": "tokens (decoder)",
     "K y V vienen de": "salida del encoder", "Forma Q": f"({n_t}, {d})", "Forma K/V": f"({n_a}, {d})"},
]))
st.caption(f"Cada una de las {h} cabezas trabaja con d_k = {d} / {h} = {dk} dimensiones.")

if matrices_qk:
    st.subheader("Atención cruzada: qué parte del audio consulta cada token")
    qk = torch.stack(matrices_qk)[:, 0]              # (capas, cabezas, n_tokens, 1500)
    pesos = torch.softmax(qk, dim=-1)
    mitad = pesos.shape[0] // 2
    pesos = pesos[mitad:].mean(dim=(0, 1))            # promedio de las últimas capas y cabezas
    filas = pesos[n_inicial - 1: len(tokens) - 1].cpu().numpy()  # query que predijo cada token
    fin = min(n_a, int(duracion / 0.02) + 10)
    filas = filas[:, :fin]
    filas = filas / filas.max(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.3 * len(piezas))))
    ax.imshow(filas, aspect="auto", cmap="magma",
              extent=[0, fin * 0.02, len(piezas) - 0.5, -0.5])
    ax.set_yticks(range(len(piezas)), piezas)
    ax.set_xlabel("Tiempo del audio (s)")
    ax.set_ylabel("Token generado")
    st.pyplot(fig)
    st.caption(
        "Cada fila es un token del decoder (Q) y cada columna una posición del encoder (K). "
        "Las zonas claras indican de qué segmento del audio se toma la información (V) "
        "para generar ese token; por eso se forma una diagonal que sigue el habla."
    )
else:
    st.warning("Esta versión de whisper no expone los pesos de atención cruzada.")
