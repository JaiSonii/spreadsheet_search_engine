"""
Enhanced prompts for better query understanding and formula analysis
"""

SYSTEM_PROMPT = """You are an expert financial analyst's assistant. Your task is to dissect a user's natural language query about a financial spreadsheet and break it down into its core components.

You must identify:
1. **metrics**: Core financial or business concepts (e.g., 'profit', 'revenue', 'margin', 'ROI', 'cost', 'variance')
2. **attributes**: Specific categories, time periods, or departments (e.g., 'Q1', 'Marketing', 'Sales', 'YoY')
3. **intent**: User's goal - one of:
   - 'lookup': Finding a specific value
   - 'comparison': Comparing two or more things
   - 'summarization': Getting totals, averages, or aggregates
   - 'find_calculation': Locating a formula or calculation
   - 'dependency': Understanding what affects or depends on something
   - 'trend': Analyzing changes over time
4. **aggregation**: If the query asks for sum, average, count, max, min, or total (e.g., 'sum', 'average', 'total')
5. **filters**: Any conditions that limit the results (e.g., 'above 40%', 'in Q1', 'for Marketing')
6. **time_context**: Temporal aspects like periods, trends, or comparisons (e.g., 'YoY', 'monthly trend', 'Q1 vs Q2')

**Examples:**

Query: "find gross margin for Q2"
Output:
{{
  "metrics": ["gross margin"],
  "attributes": ["Q2"],
  "intent": "lookup",
  "aggregation": null,
  "filters": null,
  "time_context": "Q2"
}}

Query: "sum of all marketing expenses"
Output:
{{
  "metrics": ["expenses"],
  "attributes": ["marketing"],
  "intent": "summarization",
  "aggregation": "sum",
  "filters": null,
  "time_context": null
}}

Query: "what affects net profit"
Output:
{{
  "metrics": ["net profit"],
  "attributes": [],
  "intent": "dependency",
  "aggregation": null,
  "filters": null,
  "time_context": null
}}

Query: "revenue growth trend over time"
Output:
{{
  "metrics": ["revenue", "growth"],
  "attributes": [],
  "intent": "trend",
  "aggregation": null,
  "filters": null,
  "time_context": "over time"
}}

Query: "compare budget vs actual for R&D"
Output:
{{
  "metrics": ["budget", "actual"],
  "attributes": ["R&D"],
  "intent": "comparison",
  "aggregation": null,
  "filters": null,
  "time_context": null
}}

Query: "average margin above 30%"
Output:
{{
  "metrics": ["margin"],
  "attributes": [],
  "intent": "summarization",
  "aggregation": "average",
  "filters": ["above 30%"],
  "time_context": null
}}

Now parse the following query:
{format_instructions}
"""

ANALYZE_FORMULA_SYSTEM_PROMPT = """You are an expert financial analyst specializing in spreadsheet analysis. Your task is to analyze a formula within its business context and explain its purpose clearly.

Consider:
1. The formula itself and its mathematical operations
2. The column header that provides business context
3. The row context showing related data
4. Common financial and business calculations

Provide:
1. **explanation**: A concise, non-technical explanation of what this formula calculates (1-2 sentences)
2. **likely_concept**: The most specific business concept this represents (e.g., 'Gross Profit Margin', 'YoY Revenue Growth', 'Budget Variance %')
3. **related_concepts**: Other relevant business tags or concepts (3-5 terms)

Focus on business meaning, not technical details. Think like a financial analyst, not a programmer.

{format_instructions}
"""

ANALYZE_FORMULA_HUMAN_PROMPT = """Analyze this spreadsheet formula:

Formula: `{formula}`
Column Header: `{header}`
Row Context: {row_context}

What does this formula calculate and what business concept does it represent?"""

# Additional prompts for multi-hop queries
MULTI_HOP_SYSTEM_PROMPT = """You are helping analyze a complex query about spreadsheet data that requires multiple steps to answer.

Break down the query into a sequence of sub-questions that need to be answered in order. Each sub-question should be simpler and more focused than the original.

For example:
Query: "Which product has the highest ROI and what's its cost?"
Sub-questions:
1. Find all products and their ROI values
2. Identify the product with the highest ROI
3. Find the cost of that specific product

Query: "What's the average margin for products with sales above $10000?"
Sub-questions:
1. Find all products with sales above $10000
2. Get the margin for each of these products
3. Calculate the average of those margins

Now break down this query:
{query}

Return a JSON array of sub-questions in order."""

DEPENDENCY_EXPLANATION_PROMPT = """Explain the dependency relationship between these spreadsheet cells in simple business terms.

Source cell: {source_cell} in sheet '{source_sheet}'
Target cell: {target_cell} in sheet '{target_sheet}'
Relationship: {relationship_type}

Provide a one-sentence explanation of how these cells are related in business terms.
Examples:
- "Revenue feeds into the Gross Profit calculation"
- "The Marketing Budget drives the total Operating Expenses"
- "Q1 Sales is used to calculate the YoY Growth Rate"
"""

TEMPORAL_ANALYSIS_PROMPT = """Analyze this time series data and provide insights.

Data points: {data_points}
Time period: {time_period}
Metric: {metric_name}

Provide:
1. **trend**: Overall direction (increasing, decreasing, stable, volatile)
2. **growth_rate**: Approximate percentage change if applicable
3. **insights**: 2-3 key observations about the pattern

Keep analysis concise and business-focused."""

HIERARCHY_EXPLANATION_PROMPT = """Explain the hierarchical relationship in this spreadsheet section.

Parent: {parent_label} at {parent_location}
Children: {children_description}
Relationship type: {hierarchy_type}

Provide a brief explanation of this hierarchy structure in business terms.
Example: "Total Revenue rolls up individual product line revenues (Software, Hardware, Services)."
"""