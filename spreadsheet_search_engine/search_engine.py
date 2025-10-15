"""
Enhanced Search Engine
Performs semantic search with intelligent ranking, aggregations, and multi-hop queries
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import duckdb
import re
import networkx as nx

from .prompt import SYSTEM_PROMPT


class StructuredQuery(BaseModel):
    metrics: List[str] = Field(description="The financial metrics or concepts mentioned")
    attributes: List[str] = Field(description="Any specific attributes or categories")
    intent: str = Field(description="The user's primary intent")
    aggregation: Optional[str] = Field(default=None, description="Aggregation type if any: sum, average, count, max, min")
    filters: Optional[List[str]] = Field(default=None, description="Filter conditions if any")
    time_context: Optional[str] = Field(default=None, description="Time period or comparison if mentioned")


class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.cells = []
        self.dependency_graph = None
        self.temporal_info = {}
        self.hierarchies = {}
        self._llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize DuckDB for aggregations
        self.db = duckdb.connect(':memory:')
        
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

    def _parse_query_with_llm(self, query: str) -> StructuredQuery:
        """Uses an LLM to parse the user's query and expand it intelligently."""
        parser = PydanticOutputParser(pydantic_object=StructuredQuery)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{query}"),
        ])

        chain = prompt_template | self._llm | parser
        try:
            structured_result = chain.invoke({
                "query": query, 
                "format_instructions": parser.get_format_instructions()
            })
            print(f"🤖 LLM-Parsed Result: {structured_result}")
            return structured_result
            
        except Exception as e:
            print(f"LLM parsing failed: {e}. Falling back to basic expansion.")
            # Return a basic structured query
            return StructuredQuery(
                metrics=[],
                attributes=[],
                intent="lookup",
                aggregation=None,
                filters=None,
                time_context=None
            )

    def build_index(self, semantic_data: Dict[str, Any]):
        """Build search index from semantically analyzed data"""
        self.cells = semantic_data['cells']
        self.dependency_graph = semantic_data.get('dependency_graph')
        self.temporal_info = semantic_data.get('temporal_info', {})
        self.hierarchies = semantic_data.get('hierarchies', {})
        
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
        
        # Build DuckDB table for aggregations
        self._build_aggregation_table()

    def _build_aggregation_table(self):
        """Create DuckDB table for fast aggregations"""
        # Prepare data for DuckDB
        rows = []
        for cell in self.cells:
            rows.append({
                'sheet': cell['sheet'],
                'cell_ref': cell['cell_ref'],
                'row': cell['row'],
                'column': cell['column'],
                'header': cell.get('header', ''),
                'value': cell.get('value'),
                'data_type': cell.get('data_type'),
                'has_formula': 1 if cell.get('formula') else 0,
                'concepts': ','.join(cell.get('concepts', [])),
                'importance': cell.get('importance', 0),
                'impacts_count': cell.get('impacts_count', 0)
            })
        
        # Create table
        self.db.execute("DROP TABLE IF EXISTS cells")
        self.db.execute("""
            CREATE TABLE cells (
                sheet VARCHAR,
                cell_ref VARCHAR,
                row INTEGER,
                column_num INTEGER,
                header VARCHAR,
                value VARCHAR,
                data_type VARCHAR,
                has_formula INTEGER,
                concepts VARCHAR,
                importance DOUBLE,
                impacts_count INTEGER
            )
        """)
        
        # Insert data
        self.db.executemany(
            "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r['sheet'], r['cell_ref'], r['row'], r['column'], r['header'], 
              str(r['value']) if r['value'] is not None else None,
              r['data_type'], r['has_formula'], r['concepts'], r['importance'], r['impacts_count'])
             for r in rows]
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search with intelligent query handling"""
        # Parse query
        structured_query = self._parse_query_with_llm(query)
        
        # Check if this is an aggregation query
        if structured_query.aggregation:
            return self._handle_aggregation_query(query, structured_query, top_k)
        
        # Check if this is a dependency/impact query
        if self._is_dependency_query(query):
            return self._handle_dependency_query(query, structured_query, top_k)
        
        # Check if this is a temporal query
        if structured_query.time_context or self._is_temporal_query(query):
            return self._handle_temporal_query(query, structured_query, top_k)
        
        # Standard semantic search
        return self._standard_search(query, structured_query, top_k)

    def _is_dependency_query(self, query: str) -> bool:
        """Check if query is asking about dependencies or impacts"""
        dependency_keywords = [
            'depends on', 'feeds into', 'impacts', 'affects', 'influences',
            'used by', 'used in', 'flows to', 'flows from', 'connected to',
            'what uses', 'what affects', 'traced to', 'drives'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in dependency_keywords)

    def _is_temporal_query(self, query: str) -> bool:
        """Check if query involves time series or trends"""
        temporal_keywords = [
            'trend', 'over time', 'growth', 'yoy', 'qoq', 'mom',
            'time series', 'historical', 'compare periods', 'quarter',
            'monthly', 'yearly', 'annual'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in temporal_keywords)

    def _handle_aggregation_query(
        self, 
        query: str, 
        structured_query: StructuredQuery, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Handle aggregation queries using DuckDB"""
        print(f"🔢 Detected aggregation query: {structured_query.aggregation}")
        
        # Build SQL query based on structured query
        agg_func = structured_query.aggregation.upper()
        
        # Find relevant column based on metrics
        metric_str = ' '.join(structured_query.metrics)
        
        # Try to find matching header
        sql = f"""
            SELECT 
                sheet,
                header,
                {agg_func}(CAST(value AS DOUBLE)) as result,
                COUNT(*) as count
            FROM cells
            WHERE data_type = 'numeric'
                AND header != ''
        """
        
        # Add filters based on attributes
        if structured_query.attributes:
            attr_conditions = ' OR '.join([
                f"LOWER(header) LIKE '%{attr.lower()}%' OR LOWER(concepts) LIKE '%{attr.lower()}%'"
                for attr in structured_query.attributes
            ])
            sql += f" AND ({attr_conditions})"
        
        # Add metric filters
        if structured_query.metrics:
            metric_conditions = ' OR '.join([
                f"LOWER(header) LIKE '%{metric.lower()}%' OR LOWER(concepts) LIKE '%{metric.lower()}%'"
                for metric in structured_query.metrics
            ])
            sql += f" AND ({metric_conditions})"
        
        sql += " GROUP BY sheet, header ORDER BY result DESC"
        
        try:
            results = self.db.execute(sql).fetchall()
            
            formatted_results = []
            for sheet, header, result, count in results[:top_k]:
                formatted_results.append({
                    'concept': f"{agg_func} of {header}",
                    'location': f"'{sheet}' sheet, {header} column",
                    'sheet': sheet,
                    'cell_ref': 'Aggregated',
                    'value': result,
                    'formula': None,
                    'score': 1.0,
                    'explanation': f"Calculated {agg_func.lower()} of {count} values in {header}",
                    'context': f"Aggregation result from {sheet} sheet",
                    'concepts': structured_query.metrics + structured_query.attributes,
                    'importance': 1.0,
                    'aggregation_type': agg_func
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Aggregation query failed: {e}")
            # Fall back to standard search
            return self._standard_search(query, structured_query, top_k)

    def _handle_dependency_query(
        self,
        query: str,
        structured_query: StructuredQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Handle queries about cell dependencies and impacts"""
        print(f"🔗 Detected dependency query")
        
        if not self.dependency_graph:
            return self._standard_search(query, structured_query, top_k)
        
        # First, find the cell being asked about using standard search
        initial_results = self._standard_search(query, structured_query, top_k=3)
        
        if not initial_results:
            return []
        
        # Get the top match
        target_cell = initial_results[0]
        full_ref = f"{target_cell['sheet']}!{target_cell['cell_ref']}"
        
        # Determine if asking about dependencies or impacts
        query_lower = query.lower()
        show_dependencies = any(word in query_lower for word in ['depends', 'uses', 'from', 'inputs'])
        show_impacts = any(word in query_lower for word in ['impacts', 'affects', 'feeds', 'drives', 'influences'])
        
        results = []
        G = self.dependency_graph
        
        if show_dependencies and full_ref in G:
            # Get cells this depends on
            predecessors = list(G.predecessors(full_ref))
            for pred_ref in predecessors[:top_k]:
                pred_cell = self._find_cell_by_ref(pred_ref)
                if pred_cell:
                    results.append({
                        'concept': f"Dependency: {pred_cell.get('header', pred_ref)}",
                        'location': pred_ref,
                        'sheet': pred_cell['sheet'],
                        'cell_ref': pred_cell['cell_ref'],
                        'value': pred_cell.get('value'),
                        'formula': pred_cell.get('formula'),
                        'score': 1.0,
                        'explanation': f"This cell is used by {target_cell['location']}",
                        'context': f"Input to calculation in {target_cell['location']}",
                        'concepts': pred_cell.get('concepts', []),
                        'importance': pred_cell.get('importance', 0.5),
                        'relationship': 'dependency'
                    })
        
        if show_impacts and full_ref in G:
            # Get cells that depend on this
            successors = list(G.successors(full_ref))
            for succ_ref in successors[:top_k]:
                succ_cell = self._find_cell_by_ref(succ_ref)
                if succ_cell:
                    results.append({
                        'concept': f"Impact: {succ_cell.get('header', succ_ref)}",
                        'location': succ_ref,
                        'sheet': succ_cell['sheet'],
                        'cell_ref': succ_cell['cell_ref'],
                        'value': succ_cell.get('value'),
                        'formula': succ_cell.get('formula'),
                        'score': 1.0,
                        'explanation': f"This cell depends on {target_cell['location']}",
                        'context': f"Downstream calculation affected by {target_cell['location']}",
                        'concepts': succ_cell.get('concepts', []),
                        'importance': succ_cell.get('importance', 0.5),
                        'relationship': 'impact'
                    })
        
        # If no specific direction, show both
        if not show_dependencies and not show_impacts:
            return self._handle_dependency_query(
                query + " impacts affects",
                structured_query,
                top_k // 2
            ) + self._handle_dependency_query(
                query + " depends on",
                structured_query,
                top_k // 2
            )
        
        return results[:top_k]

    def _handle_temporal_query(
        self,
        query: str,
        structured_query: StructuredQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Handle queries about time series and trends"""
        print(f"📊 Detected temporal query")
        
        if not self.temporal_info:
            return self._standard_search(query, structured_query, top_k)
        
        # Find sheets with time series
        temporal_sheets = [sheet for sheet, info in self.temporal_info.items() if info['has_time_series']]
        
        if not temporal_sheets:
            return self._standard_search(query, structured_query, top_k)
        
        # Filter cells to those in temporal sheets
        temporal_cells = [c for c in self.cells if c['sheet'] in temporal_sheets]
        
        # Perform search on temporal cells
        results = []
        query_embedding = self.model.encode(query)
        
        for cell in temporal_cells:
            if 'embedding' not in cell:
                continue
            
            similarity = cosine_similarity(
                np.array([query_embedding]),
                np.array([cell['embedding']])
            )[0][0]
            
            # Boost score for cells with temporal concepts
            if any(concept in cell.get('concepts', []) for concept in ['growth', 'trend', 'yoy', 'qoq']):
                similarity *= 1.2
            
            # Boost score for cells in time columns
            sheet_temporal = self.temporal_info.get(cell['sheet'], {})
            time_columns = [tc['column'] for tc in sheet_temporal.get('time_columns', [])]
            if cell['column_letter'] in time_columns:
                similarity *= 1.1
            
            if similarity > 0.3:
                results.append({
                    'cell': cell,
                    'score': min(similarity, 1.0),
                    'semantic_similarity': similarity
                })
        
        # Sort and format
        results.sort(key=lambda x: x['score'], reverse=True)
        return [self._format_result(r, query) for r in results[:top_k]]

    def _standard_search(
        self,
        query: str,
        structured_query: StructuredQuery,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform standard semantic search"""
        # Build expanded query
        search_terms = structured_query.metrics + structured_query.attributes
        if structured_query.intent:
            search_terms.append(structured_query.intent)
        expanded_query = " ".join(search_terms) if search_terms else query
        
        # Add traditional expansion
        expanded_query = self._expand_query(expanded_query)
        
        print(f"🔍 Expanded Query: {expanded_query}")
        
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
                cell, query, similarity, expanded_query, structured_query
            )
            
            if total_score > 0.1:  # Relevance threshold
                scored_results.append({
                    'cell': cell,
                    'score': total_score,
                    'semantic_similarity': similarity
                })
        
        # Sort by score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top K for reranking
        top_results = scored_results[:20]
        if not top_results:
            return []
        
        # Rerank with CrossEncoder
        rerank_pairs = []
        for result in top_results:
            cell = result['cell']
            cell_content = f"{cell.get('header', '')} {cell.get('value', '')}"
            rerank_pairs.append([query, cell_content.strip()])

        rerank_scores = self._reranker.predict(rerank_pairs)

        for result, score in zip(top_results, rerank_scores):
            result['score'] = score

        top_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results
        return [self._format_result(r, query) for r in top_results[:top_k]]

    def _find_cell_by_ref(self, full_ref: str) -> Optional[Dict[str, Any]]:
        """Find a cell by its full reference (Sheet!Cell)"""
        for cell in self.cells:
            if f"{cell['sheet']}!{cell['cell_ref']}" == full_ref:
                return cell
        return None

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
        expanded_query: str,
        structured_query: Optional[StructuredQuery] = None
    ) -> float:
        """Calculate comprehensive ranking score"""
        
        # Base: semantic similarity (30%)
        score = semantic_similarity * 0.3
        
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
        
        # Context Matching Bonus (10%)
        query_attrs = set(expanded_query.lower().split())
        cell_concepts = set(cell.get('concepts', []))
        
        context_matches = query_attrs.intersection(cell_concepts)
        if context_matches:
            score += min(len(context_matches) * 0.05, 0.1)
        
        # NEW: Dependency importance bonus (5%)
        dependency_importance = cell.get('dependency_importance', 0)
        score += dependency_importance * 0.05
        
        # NEW: Impact count bonus (5%) - cells that affect many others
        impacts_count = cell.get('impacts_count', 0)
        impact_score = min(impacts_count / 10, 1.0)
        score += impact_score * 0.05
        
        return min(score, 1.0)

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
        
        formatted = {
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
        
        # Add dependency info if available
        if cell.get('depends_on'):
            formatted['dependencies'] = {
                'depends_on_count': cell.get('depends_on_count', 0),
                'impacts_count': cell.get('impacts_count', 0)
            }
        
        return formatted

    def _determine_concept_name(self, cell: Dict[str, Any]) -> str:
        """Determine the business concept name for the cell"""
        if cell.get('header'):
            return cell['header']
        
        if cell.get('concepts'):
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
        
        # Check dependency importance
        impacts = cell.get('impacts_count', 0)
        if impacts > 3:
            explanations.append(f"Key cell: impacts {impacts} other calculations")
        
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

    def get_dependency_summary(self, cell_ref: str, sheet: str) -> Dict[str, Any]:
        """Get comprehensive dependency information for a cell"""
        if not self.dependency_graph:
            return {}
        
        full_ref = f"{sheet}!{cell_ref}"
        if full_ref not in self.dependency_graph:
            return {}
        
        G = self.dependency_graph
        
        predecessors = list(G.predecessors(full_ref))
        successors = list(G.successors(full_ref))
        
        try:
            # Get all upstream dependencies (transitive)
            ancestors = nx.ancestors(G, full_ref)
            # Get all downstream impacts (transitive)
            descendants = nx.descendants(G, full_ref)
        except:
            ancestors = set()
            descendants = set()
        
        return {
            'direct_dependencies': predecessors,
            'direct_impacts': successors,
            'total_dependencies': len(ancestors),
            'total_impacts': len(descendants),
            'dependency_depth': len(ancestors),
            'impact_breadth': len(descendants)
        }