from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from disease_catalog import get_disease_information


MODEL_PATH = Path(__file__).resolve().parent / "coffee_disease_model.keras"
CLASS_NAMES_PATH = Path(__file__).resolve().parent / "class_names.json"
IMAGE_SIZE = (224, 224)


class ModelLoadError(RuntimeError):
    pass


@st.cache_resource(show_spinner=False)
def load_classifier():
    if not MODEL_PATH.exists():
        raise ModelLoadError(
            "No se encontró `coffee_disease_model.keras` en la raíz del repositorio."
        )

    if not CLASS_NAMES_PATH.exists():
        raise ModelLoadError(
            "No se encontró `class_names.json` en la raíz del repositorio."
        )

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        # Importación diferida para que la interfaz aparezca antes de cargar TensorFlow.
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        class_names = json.loads(
            CLASS_NAMES_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ModelLoadError(
            f"No fue posible cargar el modelo TensorFlow: {exc}"
        ) from exc

    if not isinstance(class_names, list) or not class_names:
        raise ModelLoadError(
            "`class_names.json` debe contener una lista de clases."
        )

    try:
        output_units = int(model.output_shape[-1])
    except Exception as exc:
        raise ModelLoadError(
            f"No fue posible leer la salida del modelo: {exc}"
        ) from exc

    if output_units != len(class_names):
        raise ModelLoadError(
            f"El modelo tiene {output_units} salidas, pero el archivo de clases "
            f"contiene {len(class_names)} nombres."
        )

    return model, class_names


def predict_image(image: Image.Image) -> dict:
    model, class_names = load_classifier()

    prepared_image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(prepared_image, dtype=np.float32)
    batch = np.expand_dims(image_array, axis=0)

    try:
        probabilities = model.predict(batch, verbose=0)[0]
    except Exception as exc:
        raise RuntimeError(
            f"El modelo no pudo procesar la imagen: {exc}"
        ) from exc

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 1:
        raise RuntimeError("La salida del modelo no tiene el formato esperado.")

    if len(probabilities) != len(class_names):
        raise RuntimeError(
            "La cantidad de predicciones no coincide con las clases configuradas."
        )

    ranked_indices = np.argsort(probabilities)[::-1]
    best_index = int(ranked_indices[0])
    raw_class = str(class_names[best_index])

    disease_key, display_name, scientific_name = get_disease_information(
        raw_class
    )

    top_predictions = []

    for index in ranked_indices[: min(3, len(ranked_indices))]:
        index = int(index)
        _, top_display_name, _ = get_disease_information(
            str(class_names[index])
        )

        top_predictions.append(
            {
                "raw_class": str(class_names[index]),
                "display_name": top_display_name,
                "confidence": float(probabilities[index] * 100),
            }
        )

    return {
        "raw_class": raw_class,
        "class_key": disease_key,
        "display_name": display_name,
        "scientific_name": scientific_name,
        "confidence": float(probabilities[best_index] * 100),
        "top_predictions": top_predictions,
    }
