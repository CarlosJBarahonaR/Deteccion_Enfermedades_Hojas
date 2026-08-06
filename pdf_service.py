from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_pdf_report(
    image_bytes: bytes,
    prediction: dict,
    guidance: dict,
    generated_at: str,
) -> bytes:
    output = BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=40,
        bottomMargin=40,
        title="Diagnóstico foliar de café",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#28170F"),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor("#215C3A"),
        spaceBefore=9,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=13.5,
        textColor=colors.HexColor("#49362B"),
    )

    story = [
        Paragraph("Informe de diagnóstico foliar de café", title_style),
        Paragraph(
            f"Fecha de generación: {escape(generated_at.replace('T', ' '))}",
            body_style,
        ),
        Spacer(1, 10),
    ]

    source_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    source_image.thumbnail((900, 650))

    image_buffer = BytesIO()
    source_image.save(image_buffer, format="JPEG", quality=88)
    image_buffer.seek(0)

    ratio = source_image.height / max(source_image.width, 1)
    image_width = 5.6 * inch
    image_height = min(image_width * ratio, 3.7 * inch)

    story.append(
        PdfImage(
            image_buffer,
            width=image_width,
            height=image_height,
        )
    )
    story.append(Spacer(1, 12))

    summary = [
        ["Resultado", prediction["display_name"]],
        ["Referencia", prediction["scientific_name"]],
        ["Confianza", f'{prediction["confidence"]:.1f}%'],
    ]

    table = Table(summary, colWidths=[1.25 * inch, 4.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F0E9")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#215C3A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.3),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8CEC2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 10))

    sections = [
        ("Descripción", guidance["descripcion"]),
        ("Diferenciación visual", guidance["diferenciacion_visual"]),
        ("Manejo preventivo", guidance["manejo_preventivo"]),
        ("Buenas prácticas", guidance["buenas_practicas"]),
        ("Seguimiento y monitoreo", guidance["seguimiento"]),
        ("Cuándo consultar a un técnico", guidance["alerta_tecnica"]),
    ]

    for title, text in sections:
        story.append(Paragraph(escape(title), section_style))
        story.append(Paragraph(escape(str(text)), body_style))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "<b>Aviso:</b> herramienta académica de apoyo. No sustituye "
            "una inspección agronómica ni las indicaciones oficiales aplicables.",
            body_style,
        )
    )

    document.build(story)
    return output.getvalue()
