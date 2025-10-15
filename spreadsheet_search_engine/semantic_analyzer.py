"""
Semantic Analyzer
Understands business concepts, context, and meaning of spreadsheet content
"""

import re
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from .prompt import ANALAYZE_FORUMLA_SYSTEM_PROMPT, ANALAYZE_FORUMLA_HUMAN_PROMPT

class FormulaAnalysis(BaseModel):
    """Structured analysis of a spreadsheet formula's business meaning."""
    explanation: str = Field(description="A brief, human-readable explanation of what this formula calculates.")
    likely_concept: str = Field(description="The most likely business concept this formula represents, e.g., 'Profit Margin', 'Budget Variance', 'Total Revenue'.")
    related_concepts: List[str] = Field(description="A list of other related business concepts or tags.")


class SemanticAnalyzer:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        self.concept_map = {
            'revenue': {'synonyms': ['sales', 'income', 'turnover', 'receipts']},
            'cost': {'synonyms': ['expense', 'expenditure', 'spending', 'cogs']},
            'profit': {'synonyms': ['earnings', 'net income', 'ebitda']},
            'margin': {'synonyms': ['markup', 'spread', 'profit margin']},
            'efficiency': {'synonyms': ['productivity', 'roi', 'roa', 'roe', 'turnover']},
            'growth': {'synonyms': ['increase', 'expansion', 'yoy', 'qoq', 'cagr']},
            'budget': {'synonyms': ['plan', 'forecast', 'projection', 'target']},
            'actual': {'synonyms': ['realized', 'achieved', 'current']},
            'variance': {'synonyms': ['difference', 'deviation', 'gap']},
            'ratio': {'synonyms': ['proportion', 'percentage', 'rate', 'metric']},
        }

    def analyze(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parsed data and add semantic understanding."""
        semantic_data = parsed_data.copy()

        # Analyze each cell, passing the full data for context
        for cell in semantic_data['cells']:
            self._analyze_cell(cell, semantic_data) 

        return semantic_data

    def _analyze_cell(self, cell: Dict[str, Any], full_data: Dict[str, Any]):
        """Add semantic analysis to a single cell."""
        # Recognize basic concepts from text
        cell['concepts'] = self._recognize_concepts(cell)

        # Analyze formula if present
        if cell.get('formula'):
            header = cell.get('header') or ''
            row_context_str = self._build_row_context_string(cell, full_data)

            llm_analysis = self._analyze_formula_with_llm(cell['formula'], header, row_context_str)
            if llm_analysis:
                cell['formula_semantics'] = {
                    'type': llm_analysis.likely_concept.lower().replace(' ', '_'),
                    'explanation': llm_analysis.explanation,
                    'complexity': self._calculate_formula_complexity(cell['formula'])
                }
                # Add concepts from LLM analysis
                cell['concepts'].append(llm_analysis.likely_concept.lower().replace(' ', '_'))
                cell['concepts'].extend(llm_analysis.related_concepts)
            else: # Fallback to basic analysis
                cell['formula_semantics'] = self._analyze_formula_basic(cell['formula'])
        
        # Consolidate all concepts
        if cell.get('sheet_concepts'):
            cell['concepts'].extend(cell['sheet_concepts'])
        cell['concepts'] = sorted(list(set(c.lower() for c in cell['concepts'])))

        # Calculate importance score
        cell['importance'] = self._calculate_importance(cell)

        cell['embedding'] = self._generate_embedding(cell, full_data)

    def _recognize_concepts(self, cell: Dict[str, Any]) -> List[str]:
        """Recognize business concepts in a cell based on keywords."""
        concepts = []
        search_text = f"{cell.get('header', '')} {cell.get('value', '')}".lower()

        for concept, info in self.concept_map.items():
            if concept in search_text:
                concepts.append(concept)
                continue
            for synonym in info.get('synonyms', []):
                if synonym in search_text:
                    concepts.append(concept)
                    break
        return list(set(concepts))

    def _analyze_formula_with_llm(self, formula: str, header: str, row_context: str) -> Optional[FormulaAnalysis]:
        """Use an LLM to interpret the business meaning of a formula."""
        parser = PydanticOutputParser(pydantic_object=FormulaAnalysis)
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANALAYZE_FORUMLA_SYSTEM_PROMPT),
            ("human", ANALAYZE_FORUMLA_HUMAN_PROMPT)
        ])
        chain = prompt | self._llm | parser
        try:
            return chain.invoke({
                "formula": formula,
                "header": header,
                "row_context": row_context,
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as e:
            print(f"LLM formula analysis failed: {e}. Falling back to basic analysis.")
            return None

    def _build_row_context_string(self, cell: Dict[str, Any], full_data: Dict[str, Any]) -> str:
        """Creates a string summarizing the data in the same row as the given cell."""
        row_context = []
        for c in full_data.get('cells', []):
            # Find cells in the same row but not the cell itself
            if c['row'] == cell['row'] and c['column'] != cell['column'] and c.get('header'):
                header = c['header']
                value = c.get('value', 'N/A')
                # Only include non-empty, useful context
                if value and str(value).strip():
                    row_context.append(f"{header}: {value}")
        
        return "; ".join(row_context[:5]) # Limit to 5 for brevity

    def _generate_embedding(self, cell: Dict[str, Any], full_data: Dict[str, Any]) -> List[float]:
        """Generate a rich, context-aware embedding for the cell."""
        text_parts = []
        header = cell.get('header', '')
        
        # 1. Core identity: Header and Value/Formula
        if header:
            text_parts.append(f"Column: {header}.")
        if cell.get('formula'):
            explanation = cell.get('formula_semantics', {}).get('explanation', 'A calculation.')
            text_parts.append(f"Content: This cell is a formula that represents '{explanation}'.")
        elif cell.get('value') is not None:
            text_parts.append(f"Content: The value is '{cell['value']}'.")

        # 2. Semantic layer: Concepts
        if cell.get('concepts'):
            text_parts.append(f"Business Concepts: {', '.join(cell['concepts'])}.")

        # 3. Structural layer: Row context
        row_context_str = self._build_row_context_string(cell, full_data)
        if row_context_str:
            text_parts.append(f"Row Context: This cell is in a row with data like [{row_context_str}].")

        # 4. Location layer: Sheet context
        sheet = cell.get('sheet', '')
        sheet_concepts = cell.get('sheet_concepts', [])
        if sheet:
            text_parts.append(f"Location: Found in the '{sheet}' sheet.")
        if sheet_concepts:
            text_parts.append(f"The sheet is related to: {', '.join(sheet_concepts)}.")
        
        embedding_text = " ".join(filter(None, text_parts))
        
        return self.model.encode(embedding_text or "empty cell").tolist()

    def _calculate_importance(self, cell: Dict[str, Any]) -> float:
        """Calculate an importance score for the cell (0.0 to 1.0)."""
        score = 0.0
        # Headers are important
        if cell.get('row') == 1 and cell.get('data_type') == 'text':
            score += 0.3
        
        # Formulas are more important than raw data
        if cell.get('formula'):
            score += 0.4
            # More complex formulas are more important
            complexity = cell.get('formula_semantics', {}).get('complexity', 0)
            score += min(complexity * 0.05, 0.3)
        
        # Cells with more concepts are more significant
        if cell.get('concepts'):
            score += len(cell['concepts']) * 0.05
        
        # Keywords in headers suggest importance
        if cell.get('header'):
            important_keywords = ['total', 'summary', 'revenue', 'profit', 'margin', 'variance', 'yoy']
            if any(kw in cell['header'].lower() for kw in important_keywords):
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_formula_basic(self, formula: str) -> Dict[str, Any]:
        """Analyze formula semantics with basic pattern matching."""
        return {
            'type': self._get_formula_type_basic(formula),
            'complexity': self._calculate_formula_complexity(formula),
            'explanation': 'A custom calculation.'
        }

    def _get_formula_type_basic(self, formula: str) -> str:
        """Determine primary formula type with simple rules."""
        formula_upper = formula.upper()
        if 'SUM' in formula_upper or 'AVERAGE' in formula_upper: return 'aggregation'
        if 'IF' in formula_upper: return 'conditional'
        if 'VLOOKUP' in formula_upper or 'XLOOKUP' in formula_upper: return 'lookup'
        if '/' in formula: return 'ratio'
        if '-' in formula: return 'difference'
        return 'calculation'

    def _calculate_formula_complexity(self, formula: str) -> int:
        """Calculate a numeric score for formula complexity."""
        complexity = len(re.findall(r'[A-Z]+\(', formula)) # Functions
        complexity += len(re.findall(r'[\+\-\*\/]', formula)) # Operators
        complexity += formula.count('(') # Nesting
        return complexity