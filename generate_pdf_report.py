import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, HRFlowable
from reportlab.lib import colors

# ============================================================================
# PDF REPORT GENERATOR FOR KC ROYALS ANALYSIS
# ============================================================================

class PDFReportGenerator:
    """Convert JSON analysis report to professional PDF"""
    
    def __init__(self, json_file_path):
        self.json_path = json_file_path
        self.report_data = self._load_json()
        self.pdf_path = None
        # Define consistent color scheme
        self.primary_color = colors.HexColor('#1f4788')  # KC Royals blue
        self.secondary_color = colors.HexColor('#d4a574')  # Gold accent
        self.dark_text = colors.HexColor('#1a1a1a')
        self.light_gray = colors.HexColor('#f5f5f5')
        
    def _load_json(self):
        """Load JSON report data"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _get_heading_style(self, size=24, color=None):
        """Get consistent heading style"""
        if color is None:
            color = self.primary_color
        return ParagraphStyle(
            'CustomHeading',
            fontSize=size,
            textColor=color,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
    
    def _get_body_style(self, size=11):
        """Get consistent body text style"""
        return ParagraphStyle(
            'CustomBody',
            fontSize=size,
            textColor=self.dark_text,
            spaceAfter=10,
            leading=16
        )
    
    def _create_title_page(self, story):
        """Create professional title page"""
        # Add vertical spacing
        story.append(Spacer(1, 1.8 * inch))
        
        # Main title
        title_style = ParagraphStyle(
            'MainTitle',
            fontSize=48,
            textColor=self.primary_color,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("KC ROYALS", title_style))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'SubTitle',
            fontSize=32,
            textColor=self.secondary_color,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("2025 Roster Analysis", subtitle_style))
        
        # Divider
        story.append(HRFlowable(width=4 * inch, thickness=2, color=self.secondary_color, 
                               lineCap='butt', hAlign='CENTER', vAlign='MIDDLE'))
        
        story.append(Spacer(1, 0.3 * inch))
        
        # Descriptive subtitle
        desc_style = ParagraphStyle(
            'Description',
            fontSize=14,
            textColor=self.dark_text,
            spaceAfter=30,
            fontName='Helvetica'
        )
        story.append(Paragraph("Analytics-Driven Strategic Recommendations", desc_style))
        
        story.append(Spacer(1, 1.2 * inch))
        
        # Report details
        details_style = ParagraphStyle(
            'Details',
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            spaceAfter=6,
            fontName='Helvetica'
        )
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", details_style))
        story.append(Paragraph("<b>Document Type:</b> Confidential - Strategic Analysis", details_style))
        
        story.append(PageBreak())
        return story
    
    def _create_executive_summary_page(self, story):
        """Create executive summary page"""
        # Page title with horizontal line
        title_style = self._get_heading_style(26)
        story.append(Paragraph("EXECUTIVE SUMMARY", title_style))
        story.append(HRFlowable(width=7 * inch, thickness=1.5, color=self.secondary_color, 
                               lineCap='butt', hAlign='LEFT'))
        story.append(Spacer(1, 0.25 * inch))
        
        exec_summary = self.report_data.get('Executive Summary', {})
        body_style = self._get_body_style(10)
        
        # Create summary sections with improved formatting
        for section, metrics in exec_summary.items():
            # Section subheading
            section_style = ParagraphStyle(
                'SectionSubHeading',
                fontSize=13,
                textColor=self.primary_color,
                spaceAfter=8,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph(section, section_style))
            
            # Create table for metrics
            table_data = [['Metric', 'Value']]
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    table_data.append([metric, str(value)])
            
            if len(table_data) > 1:
                table = Table(table_data, colWidths=[3.2 * inch, 2.3 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 14),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), self.light_gray),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_gray]),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ]))
                story.append(table)
            
            story.append(Spacer(1, 0.35 * inch))
        
        story.append(PageBreak())
        return story
    
    def _create_section_page(self, story, section_num, section_title, section_data):
        """Create a section page with structured data"""
        # Section header with line
        heading_style = self._get_heading_style(26)
        story.append(Paragraph(f"SECTION {section_num}: {section_title}", heading_style))
        story.append(HRFlowable(width=7 * inch, thickness=1.5, color=self.secondary_color, 
                               lineCap='butt', hAlign='LEFT'))
        story.append(Spacer(1, 0.2 * inch))
        
        body_style = self._get_body_style(10)
        subsection_style = ParagraphStyle(
            'SubSection',
            fontSize=12,
            textColor=self.primary_color,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        # Display section data
        if isinstance(section_data, dict):
            for subsection, content in section_data.items():
                # Subsection title with background
                story.append(Paragraph(subsection, subsection_style))
                
                if isinstance(content, dict):
                    # Create table for dict content
                    table_data = [['Item', 'Value']]
                    for key, val in content.items():
                        table_data.append([str(key), str(val)])
                    
                    if len(table_data) > 1:
                        table = Table(table_data, colWidths=[2.8 * inch, 2.7 * inch])
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), self.secondary_color),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                            ('TOPPADDING', (0, 0), (-1, 0), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.light_gray]),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ('TOPPADDING', (0, 1), (-1, -1), 8),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                        ]))
                        story.append(table)
                        
                elif isinstance(content, list):
                    # Handle list content with improved formatting
                    list_items = ', '.join(map(str, content[:5]))  # First 5 items
                    if len(content) > 5:
                        list_items += f", ... and {len(content) - 5} more"
                    story.append(Paragraph(f"<b>•</b> {list_items}", body_style))
                    
                elif isinstance(content, str):
                    # Handle long text content
                    if len(content) > 300:
                        # Preserve formatting for long text
                        formatted_content = content.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
                        story.append(Paragraph(formatted_content, body_style))
                    else:
                        story.append(Paragraph(content, body_style))
                else:
                    story.append(Paragraph(str(content), body_style))
                
                story.append(Spacer(1, 0.2 * inch))
        
        story.append(PageBreak())
        return story
    
    def generate_pdf(self, output_filename=None):
        """Generate complete PDF report with improved naming"""
        if output_filename is None:
            # Better filename format: KCRoyals_RosterAnalysis_2025_YYYYMMDD.pdf
            timestamp = datetime.now().strftime("%Y%m%d")
            output_filename = f"KCRoyals_RosterAnalysis_2025_{timestamp}.pdf"
        
        # Create output path
        os.makedirs('./reports', exist_ok=True)
        self.pdf_path = os.path.join('./reports', output_filename)
        
        # Create PDF document with proper margins
        doc = SimpleDocTemplate(
            self.pdf_path,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title="KC Royals 2025 Roster Analysis",
            author="Analytics Department",
            subject="Strategic Roster Analysis & Recommendations"
        )
        
        # Build story (content)
        story = []
        
        # Title page
        story = self._create_title_page(story)
        
        # Executive Summary
        story = self._create_executive_summary_page(story)
        
        # Section 1: Team Analysis
        if 'Section 1: Team Analysis' in self.report_data:
            story = self._create_section_page(
                story, 1, "TEAM ANALYSIS",
                self.report_data['Section 1: Team Analysis']
            )
        
        # Section 2: Evaluation Framework
        if 'Section 2: Evaluation Framework' in self.report_data:
            story = self._create_section_page(
                story, 2, "PLAYER EVALUATION FRAMEWORK",
                self.report_data['Section 2: Evaluation Framework']
            )
        
        # Section 3: Roster Optimization
        if 'Section 3: Roster Optimization' in self.report_data:
            story = self._create_section_page(
                story, 3, "ROSTER OPTIMIZATION & CONTRACT DECISIONS",
                self.report_data['Section 3: Roster Optimization']
            )
        
        # Section 4: Trade Proposals
        if 'Section 4: Trade Proposals' in self.report_data:
            story = self._create_section_page(
                story, 4, "TRADE PROPOSALS",
                self.report_data['Section 4: Trade Proposals']
            )
        
        # Build PDF
        try:
            doc.build(story)
            return self.pdf_path
        except Exception as e:
            print(f"Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# EXECUTE PDF GENERATION
# ============================================================================

if __name__ == "__main__":
    import glob
    
    print("\n" + "="*80)
    print("KC ROYALS ROSTER ANALYSIS - PDF REPORT GENERATOR")
    print("="*80)
    
    # Find the most recent JSON report
    json_files = glob.glob('./reports/kc_analysis_report_*.json')
    
    if not json_files:
        print("\n❌ No JSON report found in ./reports/")
        print("Please run generate_kc_analysis_report.py first")
    else:
        # Get most recent JSON file
        latest_json = max(json_files, key=os.path.getctime)
        print(f"\n📄 Found JSON report: {latest_json}")
        
        # Generate PDF
        print("\n🔄 Generating professional PDF report...")
        pdf_gen = PDFReportGenerator(latest_json)
        pdf_path = pdf_gen.generate_pdf()
        
        if pdf_path:
            file_size_kb = os.path.getsize(pdf_path) / 1024
            print(f"\n✅ PDF report successfully generated!")
            print(f"📁 Location: {pdf_path}")
            print(f"📊 File size: {file_size_kb:.1f} KB")
            print("\n" + "="*80)
            print("REPORT GENERATION COMPLETE")
            print("="*80)
        else:
            print("\n❌ Failed to generate PDF")