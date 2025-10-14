"""
Create sample spreadsheet for testing
"""

import openpyxl
from openpyxl.styles import Font, PatternFill

def create_sample_spreadsheet():
    """Create a sample financial spreadsheet for testing"""
    wb = openpyxl.Workbook()
    
    # Revenue Analysis Sheet
    ws1 = wb.active
    if not ws1:
        raise Exception("Failed to create worksheet")
    ws1.title = "Revenue Analysis"
    
    # Headers
    headers = ['Month', 'Q1 Revenue', 'Q2 Revenue', 'Total Revenue', 'Gross Profit Margin', 'Net Profit Margin']
    for col, header in enumerate(headers, 1):
        cell = ws1.cell(1, col, header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid')
    
    # Data
    months = ['January', 'February', 'March', 'April', 'May', 'June']
    q1_revenue = [50000, 52000, 54000, 0, 0, 0]
    q2_revenue = [0, 0, 0, 56000, 58000, 60000]
    
    for row, month in enumerate(months, 2):
        ws1.cell(row, 1, month)
        ws1.cell(row, 2, q1_revenue[row-2])
        ws1.cell(row, 3, q2_revenue[row-2])
        
        # Total Revenue Formula
        ws1.cell(row, 4, f"=B{row}+C{row}")
        
        # Gross Profit Margin (assuming 40%)
        ws1.cell(row, 5, 0.40)
        
        # Net Profit Margin Formula
        ws1.cell(row, 6, f"=D{row}*E{row}")
    
    # Add summary row
    summary_row = len(months) + 3
    ws1.cell(summary_row, 1, "Total")
    ws1.cell(summary_row, 1).font = Font(bold=True)
    ws1.cell(summary_row, 2, f"=SUM(B2:B7)")
    ws1.cell(summary_row, 3, f"=SUM(C2:C7)")
    ws1.cell(summary_row, 4, f"=SUM(D2:D7)")
    
    # Cost Analysis Sheet
    ws2 = wb.create_sheet("Cost Analysis")
    
    cost_headers = ['Category', 'Budget', 'Actual', 'Variance', 'Variance %']
    for col, header in enumerate(cost_headers, 1):
        cell = ws2.cell(1, col, header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid')
    
    categories = ['Marketing', 'Operations', 'R&D', 'Sales']
    budgets = [25000, 30000, 15000, 20000]
    actuals = [27000, 28000, 16000, 19000]
    
    for row, (cat, budget, actual) in enumerate(zip(categories, budgets, actuals), 2):
        ws2.cell(row, 1, cat)
        ws2.cell(row, 2, budget)
        ws2.cell(row, 3, actual)
        ws2.cell(row, 4, f"=C{row}-B{row}")  # Variance
        ws2.cell(row, 5, f"=D{row}/B{row}")  # Variance %
    
    # Efficiency Metrics Sheet
    ws3 = wb.create_sheet("Efficiency Metrics")
    
    efficiency_headers = ['Metric', 'Value', 'Calculation']
    for col, header in enumerate(efficiency_headers, 1):
        cell = ws3.cell(1, col, header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid')
    
    metrics = [
        ('ROI', '=0.25', 'Return on Investment'),
        ('Asset Turnover', '=1.5', 'Revenue / Total Assets'),
        ('Operating Margin', '=0.35', 'Operating Income / Revenue'),
        ('EBITDA Margin', '=0.42', 'EBITDA / Revenue')
    ]
    
    for row, (metric, formula, calc) in enumerate(metrics, 2):
        ws3.cell(row, 1, metric)
        ws3.cell(row, 2, formula)
        ws3.cell(row, 3, calc)
    
    # Save
    wb.save('sample_financial_data.xlsx')
    print("✓ Created sample_financial_data.xlsx")
    print("\nThe spreadsheet contains:")
    print("  - Revenue Analysis sheet with profit margins")
    print("  - Cost Analysis sheet with budget vs actual")
    print("  - Efficiency Metrics sheet with ROI and margins")
    print("\nTry these queries:")
    print("  python main.py --file sample_financial_data.xlsx --search 'find profitability metrics'")
    print("  python main.py --file sample_financial_data.xlsx --search 'show budget variance'")
    print("  python main.py --file sample_financial_data.xlsx --search 'efficiency ratios'")

if __name__ == "__main__":
    create_sample_spreadsheet()