from __future__ import annotations

import html
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from groq_service import (
    GroqConfigurationError,
    GroqGenerationError,
    generate_guidance,
)
from model_service import ModelLoadError, predict_image
from pdf_service import create_pdf_report


st.set_page_config(
    page_title="AgroDetect Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --page: #f8f5ee;
            --paper: #fffdf9;
            --ink: #2b170d;
            --body: #4d372b;
            --muted: #806b5e;
            --line: #e4dacf;
            --soft: #f0ebe3;
            --green: #286342;
            --orange: #c55f17;
        }

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {
            background: var(--page) !important;
            color: var(--ink) !important;
        }

        [data-testid="stHeader"] {
            background: var(--page) !important;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stToolbar"] button,
        [data-testid="stHeaderActionElements"] button {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--paper) !important;
        }

        .block-container {
            width: min(100%, 1840px);
            max-width: 1840px;
            padding: 1.65rem 2rem 1.4rem;
        }

        /* Fuerza contraste legible aunque Streamlit esté configurado en modo oscuro. */
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp small,
        .stApp li,
        .stApp div[data-testid="stMarkdownContainer"],
        .stApp div[data-testid="stCaptionContainer"],
        .stApp div[data-testid="stWidgetLabel"],
        .stApp div[data-testid="stAlertContent"] {
            color: var(--body);
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp strong,
        .stApp b {
            color: var(--ink);
        }

        hr {
            border-color: var(--line) !important;
            margin: .15rem 0 1.35rem !important;
        }

        .section-title {
            margin: 0;
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(1.85rem, 2.55vw, 2.9rem);
            line-height: 1.1;
            font-weight: 700;
            letter-spacing: -.025em;
        }

        .section-copy {
            color: var(--body);
            font-size: .91rem;
            line-height: 1.55;
            margin: .65rem 0 1.25rem;
            max-width: 760px;
        }

        .micro-label {
            color: var(--muted);
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .result-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .diagnosis-name {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(1.75rem, 2.45vw, 2.75rem);
            line-height: 1.08;
            font-weight: 700;
            letter-spacing: -.02em;
            margin-top: .55rem;
        }

        .scientific-name {
            color: var(--muted);
            font-size: .78rem;
            font-style: italic;
            margin-top: .45rem;
        }

        .confidence-number {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2rem, 2.7vw, 3rem);
            font-weight: 700;
            line-height: 1;
            text-align: right;
            white-space: nowrap;
        }

        .confidence-label {
            color: var(--ink);
            font-size: .61rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            text-align: right;
            margin-top: .4rem;
        }

        .date-label {
            color: var(--muted);
            font-size: .62rem;
            letter-spacing: .08em;
            text-align: right;
            margin-bottom: 1.15rem;
        }

        .orientation-panel {
            background: var(--soft);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.25rem .3rem;
            margin-top: .7rem;
        }

        .orientation-heading {
            display: flex;
            align-items: center;
            gap: .55rem;
            color: var(--muted);
            font-size: .67rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .orientation-icon {
            display: inline-grid;
            place-items: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #5e3b0a;
            color: #fff !important;
            font-size: .72rem;
        }

        .orientation-intro {
            color: var(--body);
            font-size: .81rem;
            line-height: 1.5;
            margin: .4rem 0 .35rem;
        }

        .guide-row {
            display: grid;
            grid-template-columns: 34px minmax(0, 1fr);
            gap: .8rem;
            padding: .92rem 0;
            border-top: 1px solid #ded4c9;
        }

        .guide-row:first-of-type {
            border-top: 0;
        }

        .guide-badge {
            display: grid;
            place-items: center;
            width: 30px;
            height: 30px;
            border-radius: 9px;
            background: var(--green);
            color: #fff !important;
            font-size: .69rem;
            font-weight: 800;
        }

        .guide-title {
            color: var(--ink);
            font-size: .83rem;
            font-weight: 800;
            margin-bottom: .27rem;
        }

        .guide-text {
            color: var(--body);
            font-size: .84rem;
            line-height: 1.58;
        }

        .empty-result {
            min-height: 390px;
            display: grid;
            place-items: center;
            background: rgba(255, 253, 249, .48);
            border: 1px dashed #d8cbbd;
            border-radius: 18px;
            padding: 2rem;
            text-align: center;
        }

        .empty-title {
            color: var(--ink);
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: .4rem;
        }

        .empty-copy {
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.55;
            max-width: 420px;
        }

        .history-label {
            color: var(--muted);
            font-size: .64rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin: 1.3rem 0 .55rem;
        }

        .history-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 11px;
            padding: .72rem .9rem;
            color: var(--body);
            font-size: .78rem;
            margin-bottom: .4rem;
        }

        .history-main {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .history-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--orange);
            margin-right: .55rem;
        }

        .history-time {
            color: var(--muted);
            font-size: .68rem;
            white-space: nowrap;
        }

        .technical-notice {
            background: #e8f0e9;
            border-left: 4px solid var(--green);
            border-radius: 8px;
            color: #294937;
            font-size: .78rem;
            line-height: 1.55;
            padding: .72rem .9rem;
            margin-top: .8rem;
        }

        .footer-line {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: .67rem;
            letter-spacing: .04em;
            margin-top: 1.5rem;
            padding-top: 1rem;
        }

        /* Radio buttons */
        div[role="radiogroup"] {
            gap: 1.2rem;
            margin-bottom: .65rem;
        }

        div[role="radiogroup"] label,
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span {
            color: var(--body) !important;
            font-size: .79rem !important;
        }

        /* Uploader */
        [data-testid="stFileUploader"] {
            margin-top: .2rem;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label p,
        [data-testid="stFileUploader"] label span {
            color: var(--body) !important;
            font-size: .78rem !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--paper) !important;
            border: 1px dashed #ccbbaa !important;
            border-radius: 13px !important;
            min-height: 82px;
        }

        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzoneInstructions"] *,
        [data-testid="stFileUploaderFile"] * {
            color: var(--body) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #fff !important;
            color: var(--ink) !important;
            border: 1px solid #cdbdae !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: #fff !important;
            border: 1px solid var(--line);
            border-radius: 10px;
        }

        [data-testid="stCameraInput"] label,
        [data-testid="stCameraInput"] p,
        [data-testid="stCameraInput"] span {
            color: var(--body) !important;
        }

        [data-testid="stCameraInput"] button {
            background: #fff !important;
            color: var(--ink) !important;
            border: 1px solid #cdbdae !important;
        }

        /* Imágenes */
        [data-testid="stImage"] {
            margin-top: .7rem;
        }

        [data-testid="stImage"] img {
            width: 100%;
            max-height: 610px;
            object-fit: contain;
            background: #eee7dc;
            border: 1px solid var(--line);
            border-radius: 13px;
        }

        [data-testid="stImageCaption"] {
            color: var(--muted) !important;
        }

        /* Botones */
        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 999px !important;
            font-weight: 800 !important;
        }

        .stButton > button[kind="primary"] {
            background: var(--ink) !important;
            color: #fff !important;
            border: 1px solid var(--ink) !important;
        }

        .stButton > button[kind="primary"]:disabled {
            background: #baa99c !important;
            color: #fff !important;
            border-color: #baa99c !important;
            opacity: .72;
        }

        .stDownloadButton > button {
            background: var(--paper) !important;
            color: var(--ink) !important;
            border: 1px solid #cdbdae !important;
        }

        /* Alertas, expanders y captions */
        [data-testid="stAlert"] {
            border-radius: 11px !important;
        }

        [data-testid="stAlert"] *,
        [data-testid="stExpander"] *,
        [data-testid="stCaptionContainer"] * {
            color: var(--body) !important;
        }

        [data-testid="stExpander"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 11px;
        }

        [data-testid="stSpinner"] *,
        [data-testid="stStatusWidget"] * {
            color: var(--body) !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 1.25rem 1rem;
            }

            .result-heading {
                margin-top: 1.2rem;
            }

            .empty-result {
                min-height: 230px;
            }

            .history-row {
                align-items: flex-start;
                flex-direction: column;
                gap: .3rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_image(uploaded_file) -> tuple[Image.Image, bytes]:
    image_bytes = uploaded_file.getvalue()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("El archivo seleccionado no es una imagen válida.") from exc

    return image, image_bytes


def safe_text(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def guidance_html(guidance: dict) -> str:
    """
    Construye el panel como una sola cadena HTML sin sangría inicial.
    De esta manera Streamlit lo interpreta como HTML y no como un bloque de código.
    """
    sections = [
        (
            "Diferenciación a simple vista",
            guidance["diferenciacion_visual"],
        ),
        (
            "Manejo agronómico preventivo y correctivo",
            guidance["manejo_preventivo"],
        ),
        (
            "Consulta a un técnico",
            guidance["alerta_tecnica"],
        ),
        (
            "Monitoreo y seguimiento",
            guidance["seguimiento"],
        ),
        (
            "Buenas prácticas y trazabilidad",
            guidance["buenas_practicas"],
        ),
    ]

    parts = [
        '<div class="orientation-panel">',
        '<div class="orientation-heading">',
        '<span class="orientation-icon">!</span>',
        "Orientación y manejo preventivo",
        "</div>",
        f'<div class="orientation-intro">{safe_text(guidance["descripcion"])}</div>',
    ]

    for index, (title, text) in enumerate(sections, start=1):
        parts.extend(
            [
                '<div class="guide-row">',
                f'<div class="guide-badge">{index:02d}</div>',
                "<div>",
                f'<div class="guide-title">{safe_text(title)}</div>',
                f'<div class="guide-text">{safe_text(text)}</div>',
                "</div>",
                "</div>",
            ]
        )

    parts.append("</div>")
    return "".join(parts)


def history_html(history: list[dict]) -> str:
    if not history:
        return (
            '<div class="history-row">'
            '<span class="history-main">Todavía no se han realizado diagnósticos.</span>'
            "</div>"
        )

    parts = []

    for item in history:
        readable_time = datetime.fromisoformat(item["timestamp"]).strftime(
            "%d/%m/%Y %H:%M"
        )

        parts.extend(
            [
                '<div class="history-row">',
                '<span class="history-main">',
                '<span class="history-dot"></span>',
                f'<b>{safe_text(item["display_name"])}</b>',
                f' · {item["confidence"]:.1f}%',
                "</span>",
                f'<span class="history-time">{readable_time}</span>',
                "</div>",
            ]
        )

    return "".join(parts)


if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "history" not in st.session_state:
    st.session_state.history = []


left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    st.markdown(
        '<h1 class="section-title">Captura de Imagen Foliar</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-copy">
            Posicione la hoja de café bajo luz natural. El modelo detectará signos
            correspondientes a las clases con las que fue entrenado: Minador, Phoma,
            Roya o una hoja aparentemente sana.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_source = st.radio(
        "Fuente de la imagen",
        ["Subir archivo", "Usar cámara"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_source == "Subir archivo":
        uploaded_file = st.file_uploader(
            "Seleccione una imagen",
            type=["jpg", "jpeg", "png", "webp"],
        )
    else:
        uploaded_file = st.camera_input("Tome una fotografía de la hoja")

    image = None
    image_bytes = None
    image_name = "hoja_cafe.jpg"

    if uploaded_file is not None:
        try:
            image, image_bytes = read_image(uploaded_file)
            image_name = getattr(uploaded_file, "name", image_name)

            st.image(
                image,
                caption=image_name,
                use_container_width=True,
            )
        except ValueError as exc:
            st.error(str(exc))

    analyze_button = st.button(
        "Analizar hoja",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )

    st.markdown(
        """
        <div class="section-copy" style="font-size:.76rem;margin-top:.55rem;">
            Para mejores resultados: utilice buena iluminación, enfoque una sola hoja,
            evite filtros y procure que la lesión sea claramente visible.
        </div>
        """,
        unsafe_allow_html=True,
    )


with right_column:
    result = st.session_state.last_result

    if analyze_button and image is not None and image_bytes is not None:
        with st.spinner("Analizando la imagen con TensorFlow..."):
            try:
                prediction = predict_image(image)
            except ModelLoadError as exc:
                st.error(str(exc))
                prediction = None
            except Exception as exc:
                st.error(f"No fue posible analizar la imagen: {exc}")
                prediction = None

        guidance = None
        guidance_error = None

        if prediction is not None:
            with st.spinner("Generando orientación técnica con Groq..."):
                try:
                    guidance = generate_guidance(prediction)
                except GroqConfigurationError as exc:
                    guidance_error = str(exc)
                except GroqGenerationError as exc:
                    guidance_error = f"Groq no pudo generar la orientación: {exc}"
                except Exception as exc:
                    guidance_error = f"Error inesperado al consultar Groq: {exc}"

            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prediction": prediction,
                "guidance": guidance,
                "guidance_error": guidance_error,
                "image_bytes": image_bytes,
                "image_name": image_name,
            }

            st.session_state.last_result = result
            st.session_state.history.insert(
                0,
                {
                    "timestamp": result["timestamp"],
                    "display_name": prediction["display_name"],
                    "confidence": prediction["confidence"],
                },
            )
            st.session_state.history = st.session_state.history[:5]

    if result is None:
        st.markdown(
            """
            <div class="result-heading">
                <div class="micro-label">Último diagnóstico</div>
                <div class="date-label">Sin análisis reciente</div>
            </div>

            <div class="empty-result">
                <div>
                    <div class="empty-title">Esperando una imagen</div>
                    <div class="empty-copy">
                        Cargue o capture una hoja de café y presione
                        <b>Analizar hoja</b>. Aquí aparecerán la clase detectada,
                        el porcentaje de confianza y la orientación técnica.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        prediction = result["prediction"]
        guidance = result["guidance"]
        generated_time = datetime.fromisoformat(result["timestamp"]).strftime(
            "%d/%m/%Y %H:%M"
        )

        heading_left, heading_right = st.columns([3.25, 1])

        with heading_left:
            st.markdown(
                '<div class="micro-label">Último diagnóstico</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="diagnosis-name">{safe_text(prediction["display_name"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="scientific-name">{safe_text(prediction["scientific_name"])}</div>',
                unsafe_allow_html=True,
            )

        with heading_right:
            st.markdown(
                f'<div class="date-label">{generated_time}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence-number">{prediction["confidence"]:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="confidence-label">confianza</div>',
                unsafe_allow_html=True,
            )

        if prediction["confidence"] < 60:
            st.warning(
                "La confianza es baja. Tome otra fotografía y solicite una "
                "revisión técnica antes de aplicar medidas de control."
            )

        if result["guidance_error"]:
            st.error(result["guidance_error"])

        if guidance:
            st.markdown(
                guidance_html(guidance),
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="technical-notice">
                    <b>Aviso:</b> este sistema es una herramienta académica de apoyo.
                    La predicción no sustituye la inspección de un agrónomo,
                    un técnico de IHCAFE ni las recomendaciones oficiales.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Ver las tres predicciones principales"):
                for item in prediction["top_predictions"]:
                    st.write(
                        f'**{item["display_name"]}:** {item["confidence"]:.1f}%'
                    )

            pdf_bytes = create_pdf_report(
                image_bytes=result["image_bytes"],
                prediction=prediction,
                guidance=guidance,
                generated_at=result["timestamp"],
            )

            st.download_button(
                "Descargar informe PDF",
                data=pdf_bytes,
                file_name="diagnostico_foliar_cafe.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.markdown(
        '<div class="history-label">Historial reciente</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        history_html(st.session_state.history),
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="footer-line">
        © 2026 AGRODETECT · PROYECTO ACADÉMICO · TENSORFLOW · STREAMLIT · GROQ
    </div>
    """,
    unsafe_allow_html=True,
)
