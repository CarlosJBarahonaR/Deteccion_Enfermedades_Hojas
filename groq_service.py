from __future__ import annotations

import json
import os
import re
from typing import Any

import streamlit as st
from groq import Groq


DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqConfigurationError(RuntimeError):
    pass


class GroqGenerationError(RuntimeError):
    pass


def get_setting(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except (FileNotFoundError, KeyError):
        pass

    return os.getenv(name, default)


def extract_json(content: str) -> dict:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise GroqGenerationError(
                "La respuesta no contiene un objeto JSON válido."
            ) from exc

        try:
            result = json.loads(cleaned[first_brace:last_brace + 1])
        except json.JSONDecodeError as second_exc:
            raise GroqGenerationError(
                "No fue posible interpretar la respuesta JSON de Groq."
            ) from second_exc

    return result


def validate_guidance(payload: Any) -> dict:
    required_fields = [
        "descripcion",
        "diferenciacion_visual",
        "manejo_preventivo",
        "buenas_practicas",
        "seguimiento",
        "alerta_tecnica",
    ]

    if not isinstance(payload, dict):
        raise GroqGenerationError("Groq no devolvió un objeto JSON.")

    missing_fields = [
        field for field in required_fields if field not in payload
    ]

    if missing_fields:
        raise GroqGenerationError(
            "La respuesta está incompleta. Faltan: "
            + ", ".join(missing_fields)
        )

    result = {}

    for field in required_fields:
        value = str(payload[field]).strip()

        if not value:
            raise GroqGenerationError(
                f"El campo `{field}` llegó vacío."
            )

        result[field] = value

    return result


def generate_guidance(prediction: dict) -> dict:
    api_key = get_setting("GROQ_API_KEY")
    model_name = get_setting("GROQ_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise GroqConfigurationError(
            "Falta configurar `GROQ_API_KEY` en Streamlit. "
            "Abra Manage app → Settings → Secrets y agregue la clave."
        )

    client = Groq(api_key=api_key)

    system_prompt = """
Eres un asistente técnico agrícola especializado en cultivo de café.
Redacta orientación preventiva y prudente en español. La clasificación
proviene de un modelo académico de visión y no es un diagnóstico definitivo.
No inventes datos de clima, suelo, variedad, altitud, severidad o laboratorio.
No recomiendes marcas comerciales ni dosis exactas de agroquímicos.
Cuando corresponda, indica que se debe seguir la etiqueta, la normativa local
y la orientación de un técnico de IHCAFE o un profesional autorizado.
"""

    user_prompt = f"""
Resultado del modelo:
- Clase: {prediction["display_name"]}
- Referencia: {prediction["scientific_name"]}
- Confianza: {prediction["confidence"]:.1f} %

Devuelve únicamente un objeto JSON con estas seis claves:
- descripcion
- diferenciacion_visual
- manejo_preventivo
- buenas_practicas
- seguimiento
- alerta_tecnica

Cada valor debe ser un texto breve, claro y útil para un productor de café.
Si el resultado es una hoja aparentemente sana, aclara que solo significa
que no se observan signos claros de las cuatro clases entrenadas.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                },
                {
                    "role": "user",
                    "content": user_prompt.strip(),
                },
            ],
            temperature=0.2,
            max_completion_tokens=1100,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise GroqGenerationError("Groq devolvió una respuesta vacía.")

        return validate_guidance(extract_json(content))

    except GroqGenerationError:
        raise
    except Exception as exc:
        raise GroqGenerationError(str(exc)) from exc
