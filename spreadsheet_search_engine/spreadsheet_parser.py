"""
Enhanced Spreadsheet Parser
Extracts data, formulas, structure, dependencies, and temporal patterns
"""

import openpyxl
from openpyxl.utils import get_column_letter
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import re
import networkx as nx
from datetime import datetime
import numpy as np


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
        
        # Temporal patterns for time series detection
        self.temporal_patterns = {
            'month_names': r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            'quarter': r'\b(q[1-4]|quarter\s*[1-4])\b',
            'year': r'\b(20\d{2}|19\d{2})\b',
            'fiscal': r'\b(fy|fiscal\s*year)\s*\d{2,4}\b',
            'period': r'\b(period|pd)\s*\d+\b'
        }

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse spreadsheet and extract all relevant data including dependencies"""
        workbook = openpyxl.load_workbook(file_path, data_only=False)
        
        parsed_data = {
            'sheets': [],
            'cells': [],
            'formulas': [],
            'headers': {},
            'sheet_context': {},
            'named_ranges': {},
            'dependency_graph': None,
            'temporal_info': {},
            'hierarchies': {},
            'metadata': {
                'filename': file_path,
                'sheet_count': len(workbook.sheetnames)
            }
        }
        
        # Extract named ranges
        parsed_data['named_ranges'] = self._extract_named_ranges(workbook)
        
        # Parse each sheet
        for sheet_name in workbook.sheetnames:
            sheet_concepts = self._analyze_sheet_name(sheet_name)
            parsed_data['sheet_context'][sheet_name] = sheet_concepts
            
            sheet = workbook[sheet_name]
            sheet_data = self._parse_sheet(sheet, sheet_name, sheet_concepts)
            
            # Detect temporal patterns in this sheet
            temporal_info = self._detect_time_series(sheet_data, sheet_name)
            if temporal_info:
                parsed_data['temporal_info'][sheet_name] = temporal_info
            
            # Detect hierarchies
            hierarchies = self._detect_hierarchies(sheet_data, sheet_name)
            if hierarchies:
                parsed_data['hierarchies'][sheet_name] = hierarchies
            
            parsed_data['sheets'].append(sheet_data)
            parsed_data['cells'].extend(sheet_data['cells'])
            parsed_data['formulas'].extend(sheet_data['formulas'])
            parsed_data['headers'][sheet_name] = sheet_data['headers']
        
        # Build dependency graph
        parsed_data['dependency_graph'] = self._build_dependency_graph(parsed_data['cells'])
        
        # Enhance cells with dependency info
        self._enrich_cells_with_dependencies(parsed_data)
        
        return parsed_data

    def _extract_named_ranges(self, workbook) -> Dict[str, str]:
        """Extract Excel named ranges"""
        named_ranges = {}
        try:
            for name in workbook.defined_names.definedName:
                if name.value:
                    named_ranges[name.name] = name.value
        except Exception as e:
            print(f"Warning: Could not extract named ranges: {e}")
        return named_ranges

    def _build_dependency_graph(self, cells: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build a directed graph showing cell dependencies"""
        G = nx.DiGraph()
        
        # Create a lookup for cells by their full reference (Sheet!Cell)
        cell_lookup = {}
        for cell in cells:
            full_ref = f"{cell['sheet']}!{cell['cell_ref']}"
            cell_lookup[full_ref] = cell
            G.add_node(full_ref, **cell)
        
        # Add edges based on formula references
        for cell in cells:
            if cell.get('formula'):
                full_ref = f"{cell['sheet']}!{cell['cell_ref']}"
                dependencies = self._extract_cell_references(cell['formula'], cell['sheet'])
                
                for dep in dependencies:
                    if dep in cell_lookup:
                        # Edge from dependency TO dependent (data flows from dep to cell)
                        G.add_edge(dep, full_ref)
        
        return G

    def _extract_cell_references(self, formula: str, current_sheet: str) -> List[str]:
        """Extract all cell references from a formula"""
        references = []
        
        # Pattern for cell references (A1, $A$1, Sheet1!A1, 'Sheet Name'!A1)
        # Match cross-sheet references
        cross_sheet_pattern = r"(?:'([^']+)'|(\w+))!([A-Z]+\$?\d+)"
        cross_matches = re.finditer(cross_sheet_pattern, formula)
        for match in cross_matches:
            sheet = match.group(1) or match.group(2)
            cell_ref = match.group(3).replace('$', '')
            references.append(f"{sheet}!{cell_ref}")
        
        # Match same-sheet references
        same_sheet_pattern = r'\b([A-Z]+\$?\d+)\b'
        same_matches = re.finditer(same_sheet_pattern, formula)
        for match in same_matches:
            cell_ref = match.group(1).replace('$', '')
            # Check if not already captured as cross-sheet
            full_ref = f"{current_sheet}!{cell_ref}"
            if not any(ref.endswith(cell_ref) for ref in references):
                references.append(full_ref)
        
        # Handle range references (A1:A10)
        range_pattern = r'([A-Z]+\d+):([A-Z]+\d+)'
        range_matches = re.finditer(range_pattern, formula)
        for match in range_matches:
            # For now, just add the start and end of range
            # In production, you'd expand the entire range
            start_ref = match.group(1).replace('$', '')
            end_ref = match.group(2).replace('$', '')
            references.append(f"{current_sheet}!{start_ref}")
            references.append(f"{current_sheet}!{end_ref}")
        
        return list(set(references))

    def _detect_time_series(self, sheet_data: Dict[str, Any], sheet_name: str) -> Optional[Dict[str, Any]]:
        """Detect temporal patterns in the sheet"""
        temporal_info = {
            'time_columns': [],
            'period_type': None,
            'has_time_series': False
        }
        
        # Check headers for temporal patterns
        for header_info in sheet_data['headers']:
            header = header_info['value'].lower()
            
            for pattern_type, pattern in self.temporal_patterns.items():
                if re.search(pattern, header, re.IGNORECASE):
                    temporal_info['time_columns'].append({
                        'column': header_info['column'],
                        'header': header_info['value'],
                        'type': pattern_type
                    })
                    temporal_info['has_time_series'] = True
                    if not temporal_info['period_type']:
                        temporal_info['period_type'] = pattern_type
        
        # Check for sequential date values in columns
        for col_idx in range(1, 20):  # Check first 20 columns
            col_values = [cell.get('value') for cell in sheet_data['cells'] 
                         if cell['column'] == col_idx and cell['row'] > 1]
            
            if self._is_sequential_dates(col_values):
                col_letter = get_column_letter(col_idx)
                temporal_info['time_columns'].append({
                    'column': col_letter,
                    'header': f'Column {col_letter}',
                    'type': 'date_sequence'
                })
                temporal_info['has_time_series'] = True
        
        return temporal_info if temporal_info['has_time_series'] else None

    def _is_sequential_dates(self, values: List[Any]) -> bool:
        """Check if values form a sequential date pattern"""
        if len(values) < 3:
            return False
        
        # Check for month names
        month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december',
                      'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        str_values = [str(v).lower().strip() for v in values if v]
        month_count = sum(1 for v in str_values if any(m in v for m in month_names))
        
        if month_count >= len(str_values) * 0.6:  # At least 60% are months
            return True
        
        # Check for quarter patterns
        quarter_count = sum(1 for v in str_values if re.search(r'q[1-4]', v, re.IGNORECASE))
        if quarter_count >= len(str_values) * 0.6:
            return True
        
        return False

    def _detect_hierarchies(self, sheet_data: Dict[str, Any], sheet_name: str) -> Optional[Dict[str, Any]]:
        """Detect hierarchical relationships (subtotals, totals, categories)"""
        hierarchies = {
            'total_rows': [],
            'subtotal_rows': [],
            'categories': []
        }
        
        for cell in sheet_data['cells']:
            # Check for total/subtotal keywords
            if cell.get('value'):
                value_str = str(cell['value']).lower()
                
                if any(keyword in value_str for keyword in ['total', 'grand total', 'sum']):
                    hierarchies['total_rows'].append({
                        'row': cell['row'],
                        'cell_ref': cell['cell_ref'],
                        'label': cell['value']
                    })
                
                if any(keyword in value_str for keyword in ['subtotal', 'sub-total', 'sub total']):
                    hierarchies['subtotal_rows'].append({
                        'row': cell['row'],
                        'cell_ref': cell['cell_ref'],
                        'label': cell['value']
                    })
            
            # Check for SUM formulas that aggregate ranges (indicate hierarchy)
            if cell.get('formula'):
                sum_matches = re.finditer(r'SUM\(([A-Z]+\d+:[A-Z]+\d+)\)', cell['formula'], re.IGNORECASE)
                for match in sum_matches:
                    range_ref = match.group(1)
                    hierarchies['categories'].append({
                        'parent_cell': cell['cell_ref'],
                        'parent_row': cell['row'],
                        'aggregates': range_ref,
                        'type': 'sum_aggregation'
                    })
        
        return hierarchies if (hierarchies['total_rows'] or hierarchies['subtotal_rows'] or hierarchies['categories']) else None

    def _enrich_cells_with_dependencies(self, parsed_data: Dict[str, Any]):
        """Add dependency information to each cell"""
        G = parsed_data['dependency_graph']
        
        for cell in parsed_data['cells']:
            full_ref = f"{cell['sheet']}!{cell['cell_ref']}"
            
            if full_ref in G:
                # Predecessors: cells this cell depends on
                predecessors = list(G.predecessors(full_ref))
                cell['depends_on'] = predecessors
                cell['depends_on_count'] = len(predecessors)
                
                # Successors: cells that depend on this cell
                successors = list(G.successors(full_ref))
                cell['impacts'] = successors
                cell['impacts_count'] = len(successors)
                
                # Calculate importance based on dependencies
                cell['dependency_importance'] = self._calculate_dependency_importance(G, full_ref)
            else:
                cell['depends_on'] = []
                cell['depends_on_count'] = 0
                cell['impacts'] = []
                cell['impacts_count'] = 0
                cell['dependency_importance'] = 0.0

    def _calculate_dependency_importance(self, G: nx.DiGraph, node: str) -> float:
        """Calculate importance score based on dependency graph position"""
        # Cells that many others depend on are more important
        out_degree = G.out_degree(node)
        
        # Cells that are deep in the calculation chain are more important
        try:
            # Approximate depth by looking at descendants
            descendants = nx.descendants(G, node)
            depth_score = min(len(descendants) / 10, 1.0)
        except:
            depth_score = 0.0
        
        # Combine scores
        importance = min((out_degree * 0.1) + depth_score, 1.0)
        return importance

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
        if 'balance' in name_lower:
            concepts.append('balance_sheet')
        if 'cash' in name_lower and 'flow' in name_lower:
            concepts.append('cash_flow')
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
                    cell_data = self._parse_cell(
                        cell, sheet_name, row_idx, col_idx, headers.get(col_idx, ''), sheet_concepts 
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
            'sheet_concepts': sheet_concepts
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