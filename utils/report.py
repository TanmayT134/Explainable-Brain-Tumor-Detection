import os
import tempfile

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer,
    Table, TableStyle, KeepTogether, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime


def generate_report(image_path, heatmap_path, result, tumor_type, confidence, probabilities):

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # ==========================
    # 🏥 PAGE 1
    # ==========================
    elements.append(Paragraph("<b>AI RADIOLOGY DIAGNOSTIC REPORT</b>", styles['Title']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Paragraph("System: AI-Based Brain MRI Analyzer", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Patient Info
    elements.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['Heading2']))
    elements.append(Paragraph("Patient ID: BT-2026-001", styles['Normal']))
    elements.append(Paragraph("Modality: MRI Brain", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Summary
    conf_level = "High" if confidence > 90 else "Moderate" if confidence > 70 else "Low"

    summary_data = [
        ["Result", result],
        ["Tumor Type", tumor_type],
        ["Confidence", f"{confidence:.2f}% ({conf_level})"]
    ]

    table = Table(summary_data, colWidths=[150, 250])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ]))

    elements.append(Paragraph("<b>DIAGNOSTIC SUMMARY</b>", styles['Heading2']))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Probabilities
    prob_data = [["Class", "Probability (%)"]]
    for label, prob in probabilities.items():
        prob_data.append([label, f"{prob:.2f}"])

    prob_table = Table(prob_data, colWidths=[200, 200])
    prob_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ]))

    elements.append(Paragraph("<b>CLASS PROBABILITY DISTRIBUTION</b>", styles['Heading2']))
    elements.append(prob_table)
    elements.append(Spacer(1, 12))

    # Confusion
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_probs) > 1 and (sorted_probs[0][1] - sorted_probs[1][1]) < 15:
        elements.append(Paragraph("<b>MODEL UNCERTAINTY</b>", styles['Heading2']))
        elements.append(Paragraph(
            f"The model shows possible confusion between {sorted_probs[0][0]} and {sorted_probs[1][0]}.",
            styles['Normal']
        ))

    # 🔥 FORCE PAGE 2
    elements.append(PageBreak())

    # ==========================
    # 🖼 PAGE 2 (CLINICAL IMAGE ANALYSIS)
    # ==========================

    elements.append(Paragraph("<b>IMAGE ANALYSIS</b>", styles['Heading2']))
    elements.append(Spacer(1, 12))

    img1 = Image(image_path, width=240, height=240)

    if heatmap_path:
        img2 = Image(heatmap_path, width=240, height=240)

        image_table = Table(
            [
                [img1, img2],
                [
                    Paragraph("<b>Figure 1:</b> MRI Brain Scan", styles['Normal']),
                    Paragraph("<b>Figure 2:</b> Grad-CAM Attention Map", styles['Normal'])
                ],
                [
                    Paragraph("The original MRI scan used as input to the model.", styles['Normal']),
                    Paragraph("Visualization of regions that influenced the model's prediction.", styles['Normal'])
                ]
            ],
            colWidths=[270, 270]
        )

    else:
        image_table = Table(
            [
                [img1],
                [Paragraph("<b>Figure 1:</b> MRI Brain Scan", styles['Normal'])],
                [Paragraph("The original MRI scan used as input to the model.", styles['Normal'])]
            ],
            colWidths=[540]
        )

    image_table.setStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,1), (-1,-1), 8),
    ])

    elements.append(image_table)

    elements.append(Spacer(1, 12))

    # Clinical note
    elements.append(Paragraph(
        "<b>Note:</b> Grad-CAM highlights regions influencing the model's decision and does not represent exact tumor boundaries.",
        styles['Normal']
    ))

    # 🔥 FORCE PAGE 3
    elements.append(PageBreak())

    # ==========================
    # 📋 PAGE 3
    # ==========================

    # Impression
    if result == "No Tumor Detected":
        impression_text = "No radiological evidence of intracranial tumor detected."
    else:
        impression_text = f"Findings suggest presence of {tumor_type} tumor. Clinical validation recommended."

    impression_block = [
        Paragraph("<b>IMPRESSION</b>", styles['Heading2']),
        Spacer(1, 6),
        Paragraph(impression_text, styles['Normal']),
        Spacer(1, 15)
    ]
    elements.append(KeepTogether(impression_block))

    # Limitations
    limitations_block = [
        Paragraph("<b>LIMITATIONS</b>", styles['Heading2']),
        Spacer(1, 6),
        Paragraph(
            "This AI system is trained on limited data and is intended for assistance only. "
            "Grad-CAM provides approximate explanations and not exact tumor boundaries.",
            styles['Normal']
        ),
        Spacer(1, 15)
    ]
    elements.append(KeepTogether(limitations_block))

    # ==========================
    # 🖊 SIGNATURE SECTION WITH STAMP
    # ==========================
    elements.append(Spacer(1, 40))

    stamp_path = os.path.join("assets", "ai_stamp.png")

    try:
        stamp = Image(stamp_path, width=70, height=70)
    except:
        stamp = Paragraph("AI VERIFIED", styles['Normal'])

    signature_table = Table(
        [
            [stamp, ""],
            ["AI Diagnostic System", "Authorized Medical Reviewer"],
            ["(Digitally Generated)", "(Signature Required)"]
        ],
        colWidths=[250, 250]
    )

    signature_table.setStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ])
    
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Normal']
    ))

    elements.append(signature_table)

    # Disclaimer
    elements.append(Paragraph("<b>DISCLAIMER</b>", styles['Heading2']))
    elements.append(Paragraph(
        "This report is generated using an AI-based system and should not replace professional medical diagnosis.",
        styles['Normal']
    ))

    doc.build(elements)

    return temp_pdf.name