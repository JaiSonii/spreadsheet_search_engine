"""
Search Engine
Performs semantic search with intelligent ranking
"""

import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from.prompt import SYSTEM_PROMPT
import re

class StructuredQuery(BaseModel):
    metrics: List[str] = Field(description="The financial metrics or concepts mentioned, e.g., 'profit', 'variance'")
    attributes: List[str] = Field(description="Any specific attributes or categories, e.g., 'marketing', 'Q1', 'sales'")
    intent: str = Field(description="The user's primary intent, e.g., 'comparison', 'summarization', 'lookup'")

class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.cells = []
        self._llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Query expansion rules
        self.query_expansions = {
            'profitability': ['profit', 'margin', 'earnings', 'ebitda', 'net income', 'gross profit'],
            'efficiency': ['roi', 'roa', 'roe', 'turnover', 'productivity', 'utilization'],
            'cost': ['expense', 'spending', 'expenditure', 'cogs', 'overhead'],
            'revenue': ['sales', 'income', 'turnover', 'receipts'],
            'growth': ['yoy', 'qoq', 'cagr', 'increase', 'expansion'],
            'percentage': ['ratio', 'percent', '%', 'proportion'],
            'average': ['mean', 'avg'],
            'lookup': ['vlookup', 'xlookup', 'index', 'match'],
            'comparison': ['variance', 'vs', 'versus', 'budget vs actual', 'difference']
        }

    def _parse_query_with_llm(self, query: str) -> str:
        """Uses an LLM to parse the user's query and expand it intelligently."""
        parser = PydanticOutputParser(pydantic_object=StructuredQuery)

        prompt_template = ChatPromptTemplate.from_messages([
                                ("system", SYSTEM_PROMPT,),
                                ("human", "{query}"),
                    ])

        chain = prompt_template | self._llm | parser
        try:
            structured_result = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
            print(f"🤖 LLM-Parsed Result: {structured_result}")
            
            # Combine the parsed components into a search string
            search_terms = structured_result.metrics + structured_result.attributes
            if structured_result.intent:
                search_terms.append(structured_result.intent)
                
            expanded_query = " ".join(search_terms)
            print(f"🤖 LLM-Expanded Query: {expanded_query}")
            return expanded_query if expanded_query else query
            
        except Exception as e:
            print(f"LLM parsing failed: {e}. Falling back to basic expansion.")
            return self._expand_query(query)

    def build_index(self, semantic_data: Dict[str, Any]):
        """Build search index from semantically analyzed data"""
        self.cells = semantic_data['cells']
        
        # Extract embeddings
        embeddings = []
        for cell in self.cells:
            if 'embedding' in cell:
                embeddings.append(cell['embedding'])
            else:
                # Generate if missing
                embedding = self._generate_cell_embedding(cell)
                cell['embedding'] = embedding
                embeddings.append(embedding)
        
        self.index = np.array(embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        # Parse and expand query
        expanded_query = self._parse_query_with_llm(query)
        
        # Generate query embedding
        query_embedding = self.model.encode(expanded_query)
        
        # Calculate semantic similarity
        similarities = cosine_similarity(np.array([query_embedding]), self.index)[0]
        
        # Score and rank results
        scored_results = []
        for idx, similarity in enumerate(similarities):
            cell = self.cells[idx]
            
            # Calculate comprehensive score
            total_score = self._calculate_ranking_score(
                cell, query, similarity, expanded_query
            )
            
            if total_score > 0.1:  # Relevance threshold
                scored_results.append({
                    'cell': cell,
                    'score': total_score,
                    'semantic_similarity': similarity
                })
        
        # Sort by score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top K
        top_results = scored_results[:20]
        if not top_results:
            return []
        
        rerank_pairs = []
        for result in top_results:
            cell = result['cell']
            cell_content = f"{cell.get('header', '')} {cell.get('value', '')}"
            rerank_pairs.append([query, cell_content.strip()])

        rerank_scores = self._reranker.predict(rerank_pairs)

        for result, score in zip(top_results, rerank_scores):
            result['score'] = score # Overwrite with the more accurate score

        top_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results
        return [self._format_result(r, query) for r in top_results[:top_k]]

    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        for key, expansions in self.query_expansions.items():
            if key in query_lower:
                expanded_terms.extend(expansions)
        
        return ' '.join(expanded_terms)

    def _calculate_ranking_score(
        self, 
        cell: Dict[str, Any], 
        original_query: str,
        semantic_similarity: float,
        expanded_query: str
    ) -> float:
        """Calculate comprehensive ranking score"""
        
        # Base: semantic similarity (40%)
        score = semantic_similarity * 0.4
        
        # Keyword matching bonus (20%)
        keyword_score = self._keyword_match_score(cell, original_query)
        score += keyword_score * 0.2
        
        # Importance weighting (20%)
        importance = cell.get('importance', 0.5)
        score += importance * 0.2
        
        # Formula complexity bonus (10%)
        if cell.get('formula_semantics'):
            complexity = min(cell['formula_semantics'].get('complexity', 0) / 10, 1.0)
            score += complexity * 0.1
        
        # Concept matching bonus (10%)
        concept_score = self._concept_match_score(cell, original_query)
        score += concept_score * 0.1

        # NEW: Context Matching Bonus (up to 15%)
        query_attrs = set(expanded_query.lower().split())
        cell_concepts = set(cell.get('concepts', []))
        
        context_matches = query_attrs.intersection(cell_concepts)
        if context_matches:
            # Give a bonus for each matching concept, e.g., 'budget', 'actual', 'variance'
            score += len(context_matches) * 0.05
        
        return min(score, 1.0) # Ensure score doesn't exceed 1.0

    def _keyword_match_score(self, cell: Dict[str, Any], query: str) -> float:
        """Calculate keyword matching score"""
        query_words = set(query.lower().split())
        
        # Build searchable text
        cell_text = []
        if cell.get('header'):
            cell_text.append(cell['header'])
        if cell.get('value') and isinstance(cell['value'], str):
            cell_text.append(cell['value'])
        if cell.get('concepts'):
            cell_text.extend(cell['concepts'])
        
        cell_text = ' '.join(cell_text).lower()
        cell_words = set(cell_text.split())
        
        # Calculate overlap
        if not query_words:
            return 0.0
        
        matches = query_words.intersection(cell_words)
        return len(matches) / len(query_words)

    def _concept_match_score(self, cell: Dict[str, Any], query: str) -> float:
        """Calculate concept matching score"""
        query_lower = query.lower()
        cell_concepts = cell.get('concepts', [])
        
        if not cell_concepts:
            return 0.0
        
        matches = sum(1 for concept in cell_concepts if concept in query_lower)
        return matches / max(len(cell_concepts), 1)

    def _format_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format result for output"""
        cell = result['cell']
        
        # Determine concept name
        concept_name = self._determine_concept_name(cell)
        
        # Generate explanation
        explanation = self._generate_explanation(cell, query)
        
        # Format location
        location = self._format_location(cell)
        
        # Get business context
        business_context = cell.get('semantic_context', '')
        
        return {
            'concept': concept_name,
            'location': location,
            'sheet': cell['sheet'],
            'cell_ref': cell['cell_ref'],
            'value': cell.get('value'),
            'formula': cell.get('formula'),
            'score': result['score'],
            'explanation': explanation,
            'context': business_context,
            'concepts': cell.get('concepts', []),
            'importance': cell.get('importance', 0)
        }

    def _determine_concept_name(self, cell: Dict[str, Any]) -> str:
        """Determine the business concept name for the cell"""
        # Priority: header > concepts > formula explanation
        
        if cell.get('header'):
            return cell['header']
        
        if cell.get('concepts'):
            # Use most specific concept
            concepts = cell['concepts']
            priority_concepts = ['profitability', 'efficiency', 'margin', 'revenue', 'cost']
            
            for pc in priority_concepts:
                if pc in concepts:
                    return pc.title()
            
            return concepts[0].title()
        
        if cell.get('formula'):
            return "Calculation"
        
        return "Data Point"

    def _format_location(self, cell: Dict[str, Any]) -> str:
        """Format location in human-readable form"""
        sheet = cell['sheet']
        cell_ref = cell['cell_ref']
        header = cell.get('header')
        
        if header:
            return f"'{sheet}' sheet, {header} column ({cell_ref})"
        else:
            return f"'{sheet}' sheet, cell {cell_ref}"

    def _generate_explanation(self, cell: Dict[str, Any], query: str) -> str:
        """Generate explanation for why this result matches"""
        explanations = []
        
        # Check concept match
        concepts = cell.get('concepts', [])
        query_lower = query.lower()
        
        matching_concepts = [c for c in concepts if c in query_lower]
        if matching_concepts:
            explanations.append(f"Matches concept: {', '.join(matching_concepts)}")
        
        # Check formula relevance
        if cell.get('formula'):
            formula_type = cell.get('formula_semantics', {}).get('type', '')
            if formula_type:
                explanations.append(f"Contains {formula_type} formula")
        
        # Check header relevance
        if cell.get('header'):
            header_lower = cell['header'].lower()
            query_words = query_lower.split()
            if any(word in header_lower for word in query_words):
                explanations.append(f"Column header relates to search")
        
        if not explanations:
            explanations.append("Semantically similar to query")
        
        return ". ".join(explanations) + "."

    def _generate_cell_embedding(self, cell: Dict[str, Any]) -> List[float]:
        """Generate embedding for a cell (fallback)"""
        text_parts = []
        
        if cell.get('header'):
            text_parts.append(cell['header'])
        if cell.get('value'):
            text_parts.append(str(cell['value']))
        if cell.get('concepts'):
            text_parts.append(' '.join(cell['concepts']))
        
        text = ' '.join(text_parts) if text_parts else 'empty'
        return self.model.encode(text).tolist()