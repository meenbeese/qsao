import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    HRFlowable
)
from reportlab.lib import colors

# =============================================================================
# PDF REPORT GENERATOR — KC ROYALS ROSTER ANALYSIS
# =============================================================================

class PDFReportGenerator:
    """Convert JSON analysis report to a professional, consistent PDF"""

    def __init__(self, json_file_path):
        self.json_path = json_file_path
        self.report_data = self._load_json()
        self.pdf_path = None

        # Color palette
        self.primary_color = colors.HexColor("#1f4788")
        self.secondary_color = colors.HexColor("#d4a574")
        self.dark_text = colors.HexColor("#1a1a1a")
        self.light_gray = colors.HexColor("#f5f5f5")

        self._init_styles()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def _init_styles(self):
        self.styles = {
            "title": ParagraphStyle(
                "Title", fontSize=40, fontName="Helvetica-Bold",
                textColor=self.primary_color, spaceAfter=16
            ),
            "subtitle": ParagraphStyle(
                "Subtitle", fontSize=26, fontName="Helvetica-Bold",
                textColor=self.secondary_color, spaceAfter=24
            ),
            "section": ParagraphStyle(
                "Section", fontSize=22, fontName="Helvetica-Bold",
                textColor=self.primary_color, spaceAfter=10
            ),
            "subsection": ParagraphStyle(
                "Subsection", fontSize=14, fontName="Helvetica-Bold",
                textColor=self.primary_color, spaceAfter=6
            ),
            "body": ParagraphStyle(
                "Body", fontSize=10, fontName="Helvetica",
                textColor=self.dark_text, leading=14, spaceAfter=8
            ),
            "meta": ParagraphStyle(
                "Meta", fontSize=9, fontName="Helvetica",
                textColor=colors.HexColor("#666666"), spaceAfter=6
            )
        }

    def _styled_table(self, data, col_widths):
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, self.light_gray]),
            ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#d0d0d0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    # -------------------------------------------------------------------------
    # Pages
    # -------------------------------------------------------------------------

    def _create_title_page(self, story):
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph("KC ROYALS", self.styles["title"]))
        story.append(Paragraph("2025 Roster Analysis", self.styles["subtitle"]))

        story.append(HRFlowable(width=4 * inch, thickness=2,
                                color=self.secondary_color, hAlign="CENTER"))

        story.append(Spacer(1, 0.6 * inch))
        story.append(Paragraph(
            "Analytics-Driven Strategic Recommendations",
            self.styles["body"]
        ))

        story.append(Spacer(1, 1 * inch))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}",
            self.styles["meta"]
        ))
        story.append(Paragraph(
            "<b>Document Type:</b> Confidential – Strategic Analysis",
            self.styles["meta"]
        ))

        story.append(PageBreak())
        return story

    def _create_executive_summary_page(self, story):
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles["section"]))
        story.append(HRFlowable(width=7 * inch, thickness=1.25,
                                color=self.secondary_color))
        story.append(Spacer(1, 0.25 * inch))

        exec_summary = self.report_data.get("Executive Summary", {})

        for section, metrics in exec_summary.items():
            story.append(Paragraph(section, self.styles["subsection"]))

            table_data = [["Metric", "Value"]]
            for k, v in metrics.items():
                table_data.append([str(k), str(v)])

            story.append(self._styled_table(
                table_data, [3.5 * inch, 2.5 * inch]
            ))
            story.append(Spacer(1, 0.3 * inch))

        story.append(PageBreak())
        return story

    def _create_section_page(self, story, section_num, section_title, section_data):
        story.append(Paragraph(
            f"SECTION {section_num}: {section_title}",
            self.styles["section"]
        ))
        story.append(HRFlowable(width=7 * inch, thickness=1.25,
                                color=self.secondary_color))
        story.append(Spacer(1, 0.25 * inch))

        for subsection, content in section_data.items():
            story.append(Paragraph(subsection, self.styles["subsection"]))

            # WAR impact fields (text, not tables)
            if subsection in ["Short Term WAR", "Long Term WAR"]:
                story.append(Paragraph(
                    f"<b>{subsection} Impact:</b> {content}",
                    self.styles["body"]
                ))
                story.append(Spacer(1, 0.2 * inch))
                continue

            # Dictionary → key/value table
            if isinstance(content, dict):
                table_data = [["Item", "Value"]]
                for k, v in content.items():
                    table_data.append([str(k), str(v)])
                story.append(self._styled_table(
                    table_data, [3.2 * inch, 2.8 * inch]
                ))

            # List handling
            elif isinstance(content, list):

                # Empty list
                if len(content) == 0:
                    story.append(Paragraph("None", self.styles["body"]))

                # List of player dictionaries → proper table
                elif isinstance(content[0], dict):
                    headers = list(content[0].keys())
                    table_data = [headers]
                    for row in content:
                        table_data.append([str(row.get(h, "")) for h in headers])

                    story.append(self._styled_table(
                        table_data,
                        [2.2 * inch for _ in headers]
                    ))

                # Simple list
                else:
                    for item in content:
                        story.append(
                            Paragraph(f"• {item}", self.styles["body"])
                        )

            # String block
            elif isinstance(content, str):
                formatted = content.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
                story.append(Paragraph(formatted, self.styles["body"]))

            else:
                story.append(Paragraph(str(content), self.styles["body"]))

            story.append(Spacer(1, 0.3 * inch))

        story.append(PageBreak())
        return story

    # -------------------------------------------------------------------------
    # PDF build
    # -------------------------------------------------------------------------

    def generate_pdf(self, output_filename=None):
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_filename = f"KCRoyals_RosterAnalysis_2025_{timestamp}.pdf"

        os.makedirs("./reports", exist_ok=True)
        self.pdf_path = os.path.join("./reports", output_filename)

        doc = SimpleDocTemplate(
            self.pdf_path,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []
        self._create_title_page(story)
        self._create_executive_summary_page(story)

        self._create_section_page(
            story, 1, "TEAM ANALYSIS",
            self.report_data.get("Section 1: Team Analysis", {})
        )
        self._create_section_page(
            story, 2, "PLAYER EVALUATION FRAMEWORK",
            self.report_data.get("Section 2: Evaluation Framework", {})
        )
        self._create_section_page(
            story, 3, "ROSTER OPTIMIZATION & CONTRACT DECISIONS",
            self.report_data.get("Section 3: Roster Optimization", {})
        )
        self._create_section_page(
            story, 4, "TRADE PROPOSALS",
            self.report_data.get("Section 4: Trade Proposals", {})
        )

        doc.build(story)
        return self.pdf_path


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    import glob

    json_files = glob.glob("./reports/kc_analysis_report_*.json")

    if not json_files:
        print("❌ No JSON report found.")
    else:
        latest_json = max(json_files, key=os.path.getctime)
        generator = PDFReportGenerator(latest_json)
        pdf_path = generator.generate_pdf()

        if pdf_path:
            print(f"✅ PDF generated: {pdf_path}")
        else:
            print("❌ PDF generation failed.")
