"""
Semantic Analyzer
Understands business concepts, context, and meaning of spreadsheet content
"""

import re
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from typing import Optional


class SemanticAnalyzer:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Business concept knowledge base
        self.concept_map = {
            'revenue': {
                'synonyms': ['sales', 'income', 'turnover', 'receipts', 'earnings'],
                'related': ['gross revenue', 'net revenue', 'total sales'],
                'category': 'financial'
            },
            'cost': {
                'synonyms': ['expense', 'expenditure', 'spending', 'cogs', 'cost of goods sold'],
                'related': ['operating cost', 'overhead', 'direct cost', 'indirect cost'],
                'category': 'financial'
            },
            'profit': {
                'synonyms': ['earnings', 'net income', 'bottom line', 'surplus'],
                'related': ['gross profit', 'net profit', 'operating profit', 'ebitda'],
                'category': 'financial'
            },
            'margin': {
                'synonyms': ['markup', 'spread', 'profit margin'],
                'related': ['gross margin', 'net margin', 'operating margin', 'contribution margin'],
                'category': 'financial'
            },
            'efficiency': {
                'synonyms': ['productivity', 'performance', 'effectiveness'],
                'related': ['roi', 'roa', 'roe', 'asset turnover', 'inventory turnover'],
                'category': 'ratio'
            },
            'growth': {
                'synonyms': ['increase', 'expansion', 'development'],
                'related': ['yoy', 'qoq', 'cagr', 'growth rate'],
                'category': 'trend'
            },
            'budget': {
                'synonyms': ['plan', 'forecast', 'projection', 'target'],
                'related': ['budgeted', 'planned', 'estimated'],
                'category': 'planning'
            },
            'actual': {
                'synonyms': ['realized', 'achieved', 'real', 'current'],
                'related': ['actuals', 'actual results'],
                'category': 'results'
            },
            'variance': {
                'synonyms': ['difference', 'deviation', 'gap'],
                'related': ['budget variance', 'forecast variance'],
                'category': 'comparison'
            },
            'ratio': {
                'synonyms': ['proportion', 'percentage', 'rate'],
                'related': ['financial ratio', 'metric', 'kpi'],
                'category': 'metric'
            }
        }
        
        # Formula semantic patterns
        self.formula_semantics = {
            'profitability': ['margin', 'profit', 'earnings', 'ebitda'],
            'efficiency': ['turnover', 'roi', 'roa', 'roe'],
            'growth': ['yoy', 'qoq', 'cagr', 'growth'],
            'comparison': ['variance', 'vs', 'budget', 'actual'],
            'aggregation': ['sum', 'total', 'average', 'count']
        }

    def analyze(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parsed data and add semantic understanding"""
        semantic_data = parsed_data.copy()
        
        # Analyze each cell
        for cell in semantic_data['cells']:
            self._analyze_cell(cell, parsed_data)
        
        return semantic_data

    def _analyze_cell(self, cell: Dict[str, Any], full_data: Dict[str, Any]):
        """Add semantic analysis to a cell"""
        # Recognize business concepts
        cell['concepts'] = self._recognize_concepts(cell)
        
        # Incorporate sheet-level context
        if cell.get('sheet_concepts'):
            cell['concepts'].extend(cell['sheet_concepts'])
            cell['concepts'] = list(set(cell['concepts']))

        # Understand context
        cell['semantic_context'] = self._build_context(cell, full_data)
        
        # Analyze formula if present
        if cell.get('formula'):
            # FIX: Ensure header is a string, not None
            header = cell.get('header') or ''
            cell['formula_semantics'] = self._analyze_formula(cell['formula'])
            cell['concepts'].extend(self._extract_formula_concepts(cell['formula'], header))
            cell['concepts'] = list(set(cell['concepts']))
        
        # Calculate importance score
        cell['importance'] = self._calculate_importance(cell)
        
        # Generate embedding
        cell['embedding'] = self._generate_embedding(cell)

    def _recognize_concepts(self, cell: Dict[str, Any]) -> List[str]:
        """Recognize business concepts in cell"""
        concepts = []
        
        # Build searchable text from cell
        search_text = []
        if cell.get('header'):
            search_text.append(cell['header'])
        if cell.get('value'):
            search_text.append(str(cell['value']))
        if cell.get('formula'):
            search_text.append(cell['formula'])
        
        search_text = ' '.join(search_text).lower()
        
        # Check against concept map
        for concept, info in self.concept_map.items():
            if concept in search_text:
                concepts.append(concept)
                continue
            
            # Check synonyms
            for synonym in info['synonyms']:
                if synonym in search_text:
                    concepts.append(concept)
                    break
        
        # Recognize formula-based concepts
        if cell.get('formula'):
            # FIX: Ensure header is a string here as well
            header = cell.get('header') or ''
            formula_concepts = self._extract_formula_concepts(cell['formula'], header)
            concepts.extend(formula_concepts)
        
        return list(set(concepts))

    def _extract_formula_concepts(self, formula: str, header: str = "") -> List[str]:
        """Extract business concepts from formula structure, using header for context."""
        concepts = []
        formula_lower = formula.lower()
        # This line is now safe because the callers ensure header is a string
        header_lower = header.lower()

        # Check if it's a percentage/ratio calculation
        if '/' in formula:
            concepts.append('ratio')

            if 'margin' in header_lower or 'profitability' in header_lower:
                concepts.append('profitability_metric')
                concepts.append('margin_calculation')
            elif 'variance' in header_lower and '%' in header_lower:
                concepts.append('variance_analysis')

        # Check for aggregations
        if any(func in formula.upper() for func in ['SUM', 'AVERAGE', 'COUNT']):
            concepts.append('aggregation')
            if 'total' in header_lower:
                concepts.append('total_calculation')

        # Check for comparisons (Budget vs Actual)
        if '-' in formula and ('budget' in header_lower or 'actual' in header_lower or 'variance' in header_lower):
            concepts.append('variance_analysis')
            concepts.append('comparison')

        return concepts

    def _build_context(self, cell: Dict[str, Any], full_data: Dict[str, Any]) -> str:
        """Build semantic context for the cell"""
        context_parts = []
        
        context_parts.append(f"Sheet: {cell['sheet']}")
        
        if cell.get('header'):
            context_parts.append(f"Column: {cell['header']}")
        
        if cell.get('formula'):
            # FIX: Ensure header is a string before passing to explain_formula
            header = cell.get('header') or ''
            formula_meaning = self._explain_formula(cell['formula'], header)
            if formula_meaning:
                context_parts.append(formula_meaning)
        
        if cell.get('concepts'):
            context_parts.append(f"Concepts: {', '.join(cell['concepts'])}")
        
        return ' | '.join(context_parts)

    def _explain_formula(self, formula: str, header: str) -> str:
        """Generate human-readable explanation of formula"""
        formula_upper = formula.upper()
        
        if 'SUM' in formula_upper:
            base = header if header else "values"
            return f"Calculates total {base}"
        elif 'AVERAGE' in formula_upper:
            base = header if header else "values"
            return f"Calculates average {base}"
        elif '/' in formula and header:
            if any(word in header.lower() for word in ['margin', 'percent', 'ratio']):
                return f"Calculates {header.lower()}"
        elif 'IF' in formula_upper:
            return "Conditional calculation"
        elif 'VLOOKUP' in formula_upper:
            return "Lookup calculation"
        
        return "Custom calculation"

    def _analyze_formula(self, formula: str) -> Dict[str, Any]:
        """Analyze formula semantics"""
        return {
            'type': self._get_formula_type(formula),
            'complexity': self._calculate_formula_complexity(formula),
            'operations': self._extract_operations(formula)
        }

    def _get_formula_type(self, formula: str) -> str:
        """Determine primary formula type"""
        formula_upper = formula.upper()
        
        if 'SUM' in formula_upper:
            return 'aggregation'
        elif 'AVERAGE' in formula_upper:
            return 'aggregation'
        elif 'IF' in formula_upper:
            return 'conditional'
        elif 'VLOOKUP' in formula_upper or 'XLOOKUP' in formula_upper:
            return 'lookup'
        elif '/' in formula:
            return 'ratio'
        else:
            return 'calculation'

    def _calculate_formula_complexity(self, formula: str) -> int:
        """Calculate formula complexity score"""
        complexity = 0
        complexity += formula.count('(') * 2
        complexity += formula.count('+') + formula.count('-') + formula.count('*') + formula.count('/')
        if 'IF' in formula.upper():
            complexity += 3
        if 'VLOOKUP' in formula.upper():
            complexity += 4
        return complexity

    def _extract_operations(self, formula: str) -> List[str]:
        """Extract all operations from formula"""
        operations = []
        formula_upper = formula.upper()
        functions = ['SUM', 'AVERAGE', 'COUNT', 'IF', 'VLOOKUP', 'XLOOKUP', 'INDEX', 'MATCH']
        for func in functions:
            if func in formula_upper:
                operations.append(func.lower())
        return operations

    def _calculate_importance(self, cell: Dict[str, Any]) -> float:
        """Calculate importance score for cell"""
        score = 0.0
        
        if cell.get('row') == 1:
            score += 0.2
        
        if cell.get('formula'):
            score += 0.3
            if cell.get('formula_semantics'):
                complexity = cell['formula_semantics'].get('complexity', 0)
                score += min(complexity * 0.05, 0.3)
        
        if cell.get('concepts'):
            score += len(cell['concepts']) * 0.1
        
        if cell.get('header'):
            # This check is safe because if cell.get('header') is None, the block is skipped.
            header_lower = cell['header'].lower()
            important_keywords = ['total', 'revenue', 'profit', 'margin', 'cost', 'ratio']
            if any(kw in header_lower for kw in important_keywords):
                score += 0.2
        
        return min(score, 1.0)

    def _generate_embedding(self, cell: Dict[str, Any]) -> List[float]:
        """Generate embedding for cell content"""
        text_parts = []
        
        if cell.get('header'):
            text_parts.append(cell['header'])
        
        if cell.get('concepts'):
            text_parts.append(' '.join(cell['concepts']))
        
        if cell.get('value') and cell['data_type'] == 'text':
            text_parts.append(str(cell['value']))
        
        if cell.get('formula'):
            # FIX: Ensure header is a string before passing to explain_formula
            header = cell.get('header') or ''
            explanation = self._explain_formula(cell['formula'], header)
            text_parts.append(explanation)
        
        if cell.get('sheet_concepts'):
            text_parts.append(' '.join(cell['sheet_concepts']))
            
        text_parts.append(cell.get('sheet', ''))
        
        embedding_text = ' '.join(filter(None, text_parts))
        
        if not embedding_text.strip():
            embedding_text = "empty cell"
        
        return self.model.encode(embedding_text).tolist()