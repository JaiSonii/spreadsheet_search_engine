"""
Spreadsheet Parser
Extracts data, formulas, and structure from spreadsheet files
"""

import openpyxl
from openpyxl.utils import get_column_letter
import pandas as pd
from typing import Dict, List, Any
import re


class SpreadsheetParser:
    def __init__(self):
        self.formula_patterns = {
            'sum': r'SUM\s*\(',
            'average': r'AVERAGE\s*\(',
            'count': r'COUNT(IF|A)?\s*\(',
            'vlookup': r'VLOOKUP\s*\(',
            'if': r'IF\s*\(',
            'percentage': r'[\d\.]+\s*/\s*[\d\.]+'
        }

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse spreadsheet and extract all relevant data"""
        workbook = openpyxl.load_workbook(file_path, data_only=False)
        
        parsed_data = {
            'sheets': [],
            'cells': [],
            'formulas': [],
            'headers': {},
            'sheet_context': {}, # NEW: For workbook-level context
            'metadata': {
                'filename': file_path,
                'sheet_count': len(workbook.sheetnames)
            }
        }
        
        for sheet_name in workbook.sheetnames:
            # NEW: Analyze sheet name for concepts
            sheet_concepts = self._analyze_sheet_name(sheet_name)
            parsed_data['sheet_context'][sheet_name] = sheet_concepts
            
            sheet = workbook[sheet_name]
            # Pass sheet_concepts to the sheet parser
            sheet_data = self._parse_sheet(sheet, sheet_name, sheet_concepts)
            parsed_data['sheets'].append(sheet_data)
            parsed_data['cells'].extend(sheet_data['cells'])
            parsed_data['formulas'].extend(sheet_data['formulas'])
            parsed_data['headers'][sheet_name] = sheet_data['headers']
        
        return parsed_data

    def _analyze_sheet_name(self, sheet_name: str) -> List[str]:
        """Extracts concepts like 'budget', 'actual', 'forecast' from a sheet name."""
        name_lower = sheet_name.lower()
        concepts = []
        if 'budget' in name_lower:
            concepts.append('budget')
        if 'actual' in name_lower:
            concepts.append('actual')
        if 'forecast' in name_lower or 'fcst' in name_lower:
            concepts.append('forecast')
        if 'p&l' in name_lower or 'income' in name_lower:
            concepts.append('p&l_statement')
        return concepts

    def _parse_sheet(self, sheet, sheet_name: str, sheet_concepts: List[str]) -> Dict[str, Any]:
        """Parse individual sheet"""
        sheet_data = {
            'name': sheet_name,
            'cells': [],
            'formulas': [],
            'headers': []
        }
        
        # Extract headers (first row typically)
        headers = {}
        for col_idx, cell in enumerate(sheet[1], 1):
            if cell.value:
                headers[col_idx] = str(cell.value).strip()
                sheet_data['headers'].append({
                    'column': get_column_letter(col_idx),
                    'value': str(cell.value).strip()
                })
        
        # Parse all cells
        for row_idx, row in enumerate(sheet.iter_rows(min_row=1), 1):
            for col_idx, cell in enumerate(row, 1):
                if cell.value is not None:
                    # Pass sheet_concepts down to the cell parser
                    cell_data = self._parse_cell(
                        cell, sheet_name, row_idx, col_idx, headers.get(col_idx), sheet_concepts
                    )
                    sheet_data['cells'].append(cell_data)
                    
                    if cell_data.get('formula'):
                        sheet_data['formulas'].append(cell_data)
        
        return sheet_data

    def _parse_cell(self, cell, sheet_name: str, row: int, col: int, header: str = "", sheet_concepts: List[str] = []) -> Dict[str, Any]:
        """Parse individual cell with context"""
        cell_ref = f"{get_column_letter(col)}{row}"
        location = f"{sheet_name}!{cell_ref}"
        
        cell_data = {
            'sheet': sheet_name,
            'location': location,
            'cell_ref': cell_ref,
            'row': row,
            'column': col,
            'column_letter': get_column_letter(col),
            'header': header,
            'value': None,
            'data_type': None,
            'formula': None,
            'formula_type': None,
            'sheet_concepts': sheet_concepts # NEW: Add sheet concepts to cell data
        }
        
        # Check if cell has formula
        if hasattr(cell, 'value') and isinstance(cell.value, str) and cell.value.startswith('='):
            cell_data['formula'] = cell.value
            cell_data['formula_type'] = self._identify_formula_type(cell.value)
            cell_data['data_type'] = 'formula'
            
            # Try to get calculated value
            try:
                cell_data['value'] = cell.value
            except:
                cell_data['value'] = None
        else:
            cell_data['value'] = cell.value
            cell_data['data_type'] = self._identify_data_type(cell.value)
        
        return cell_data

    def _identify_formula_type(self, formula: str) -> List[str]:
        """Identify types of formulas used"""
        formula_upper = formula.upper()
        types = []
        
        for formula_type, pattern in self.formula_patterns.items():
            if re.search(pattern, formula_upper):
                types.append(formula_type)
        
        return types if types else ['other']

    def _identify_data_type(self, value: Any) -> str:
        """Identify data type of cell value"""
        if value is None:
            return 'empty'
        elif isinstance(value, (int, float)):
            return 'numeric'
        elif isinstance(value, str):
            # Check if it's a date-like string
            if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', value):
                return 'date'
            return 'text'
        else:
            return 'other'