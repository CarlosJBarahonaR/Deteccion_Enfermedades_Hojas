from __future__ import annotations

import unicodedata


DISEASE_CATALOG = {
    "minador": {
        "display_name": "Minador de la hoja",
        "scientific_name": "Leucoptera coffeella",
    },
    "phoma": {
        "display_name": "Phoma",
        "scientific_name": "Phoma spp.",
    },
    "roya": {
        "display_name": "Roya del cafeto",
        "scientific_name": "Hemileia vastatrix",
    },
    "sana": {
        "display_name": "Hoja aparentemente sana",
        "scientific_name": "Sin signos visibles de las clases entrenadas",
    },
}


def normalize_class_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    return (
        normalized.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def get_disease_information(raw_class: str) -> tuple[str, str, str]:
    key = normalize_class_name(raw_class)

    information = DISEASE_CATALOG.get(
        key,
        {
            "display_name": raw_class.replace("_", " ").title(),
            "scientific_name": "Clase identificada por el modelo",
        },
    )

    return (
        key,
        information["display_name"],
        information["scientific_name"],
    )
