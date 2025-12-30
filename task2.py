import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import csv

# Create sample data CSV file
def create_sample_data():
    """
    Create a sample CSV file for demonstration
    """
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'Quantity': [25, 150, 80, 45, 120],
        'Price_per_Unit': [800, 25, 75, 300, 120],
        'Total_Sales': [20000, 3750, 6000, 13500, 14400]
    }

    df = pd.DataFrame(data)
    df.to_csv('sales_data.csv', index=False)
    return df

# Read and analyze data
def analyze_data(csv_file):
    """
    Read data from CSV and perform analysis
    """
    df = pd.read_csv(csv_file)

    analysis = {
        'total_revenue': df['Total_Sales'].sum(),
        'avg_price': df['Price_per_Unit'].mean(),
        'total_items': df['Quantity'].sum(),
        'top_product': df.loc[df['Total_Sales'].idxmax(), 'Product'],
        'highest_revenue': df['Total_Sales'].max(),
        'data': df
    }

    return analysis

# Generate PDF report
def generate_pdf_report(analysis, output_file='Sales_Report.pdf'):
    """
    Generate a formatted PDF report from analysis
    """
    doc = SimpleDocTemplate(output_file, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()

    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Add title
    story.append(Paragraph("Sales Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Revenue', f"${analysis['total_revenue']:,.2f}"],
        ['Total Items Sold', f"{analysis['total_items']:,}"],
        ['Average Price per Unit', f"${analysis['avg_price']:.2f}"],
        ['Top Performing Product', analysis['top_product']],
        ['Highest Single Product Revenue', f"${analysis['highest_revenue']:,.2f}"]
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))

    # Detailed Data Table
    story.append(Paragraph("Detailed Sales Data", heading_style))

    data_table_content = [['Product', 'Quantity', 'Price/Unit', 'Total Sales']]
    for _, row in analysis['data'].iterrows():
        data_table_content.append([
            row['Product'],
            str(row['Quantity']),
            f"${row['Price_per_Unit']:.2f}",
            f"${row['Total_Sales']:,.2f}"
        ])

    data_table = Table(data_table_content, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))

    story.append(data_table)
    story.append(Spacer(1, 0.3*inch))

    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendation_text = f"""
    Based on the analysis, the following recommendations are made:
    <br/><br/>
    1. <b>Focus on Top Products:</b> {analysis['top_product']} is the highest revenue generator.
    Consider increasing stock and marketing efforts.
    <br/><br/>
    2. <b>Inventory Optimization:</b> Review low-performing products and consider promotional strategies.
    <br/><br/>
    3. <b>Price Strategy:</b> The average product price is ${analysis['avg_price']:.2f}.
    Monitor market trends and adjust pricing accordingly.
    """

    story.append(Paragraph(recommendation_text, styles['Normal']))

    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated successfully: {output_file}")

# Main execution
if __name__ == "__main__":
    print("Creating sample data...")
    df = create_sample_data()
    print("✓ Sample data created")

    print("Analyzing data...")
    analysis = analyze_data('sales_data.csv')
    print("✓ Data analysis complete")

    print("Generating PDF report...")
    generate_pdf_report(analysis)
