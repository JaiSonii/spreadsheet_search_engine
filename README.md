# Semantic Search Engine for Spreadsheets

A powerful semantic search system that understands spreadsheet content conceptually, allowing users to find what they're looking for using natural language queries.

### Design Document Link
``` bash
[https://docs.google.com/document/d/e/2PACX-1vRjTAL7iGW5gwI6lvocdUc3SZmbpw-M4GgS6uDVvGNTxmxaTaNUd_02P8b4j2bSWxrleAG75b0k0_Bj/pub][https://docs.google.com/document/d/e/2PACX-1vRjTAL7iGW5gwI6lvocdUc3SZmbpw-M4GgS6uDVvGNTxmxaTaNUd_02P8b4j2bSWxrleAG75b0k0_Bj/pub]
```

## Features

### 1. **Semantic Content Understanding**
- Recognizes business concepts (revenue, costs, margins, ratios, forecasts)
- Handles synonyms (sales = revenue, profit = earnings)
- Interprets context (distinguishes "Marketing Spend" from "Marketing ROI")
- Understands formula semantics (recognizes profitability calculations)

### 2. **Natural Language Query Processing**
- **Conceptual Queries**: "Find all profitability metrics" → gross margin, net profit, EBITDA
- **Functional Queries**: "Show percentage calculations" → all formulas with percentages
- **Comparative Queries**: "Budget vs actual analysis" → variance calculations

### 3. **Intelligent Ranking**
- Semantic relevance scoring
- Context importance weighting
- Formula complexity consideration
- Data recency awareness

### 4. **Rich Output Format**
- Human-readable summaries with context
- JSON format for programmatic use
- Explains why results match your query
- Provides business context for each result

## Installation

```bash
# Clone the repository
git clone https://github.com/JaiSonii/spreadsheet_search_engine
cd semantic-spreadsheet-search

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Search

```bash
# Index and search a spreadsheet
python main.py --file data.xlsx --search "find all profitability metrics"
```

### Index Only

```bash
# Just index without searching
python main.py --file data.xlsx --index
```

### Advanced Options

```bash
# Limit number of results
python main.py --file data.xlsx --search "show cost calculations" --top 3

# Get JSON output
python main.py --file data.xlsx --search "efficiency ratios" --format json

# Full help
python main.py --help
```

## Example Queries

### Conceptual Queries
- `"Find all revenue calculations"`
- `"Show me profitability metrics"`
- `"Where are my margin analyses?"`
- `"Find efficiency ratios"`

### Functional Queries
- `"Show percentage calculations"`
- `"Find average formulas"`
- `"What conditional calculations exist?"`
- `"Show lookup formulas"`

### Comparative Queries
- `"Budget vs actual analysis"`
- `"Time series data"`
- `"Variance calculations"`

## Architecture

```
main.py                 # CLI interface
spreadsheet_parser.py   # Extracts data and formulas from spreadsheets
semantic_analyzer.py    # Understands business concepts and context
search_engine.py        # Performs semantic search with ranking
```

### How It Works

1. **Parse Spreadsheet**: Extract cells, formulas, headers, and structure
2. **Semantic Analysis**: 
   - Recognize business concepts in content
   - Understand formula meanings
   - Generate embeddings for semantic search
   - Calculate importance scores
3. **Build Index**: Create searchable vector index
4. **Search & Rank**:
   - Expand query with related terms
   - Perform semantic similarity search
   - Apply intelligent ranking (semantic + keyword + importance + complexity)
   - Return top results with explanations

## Output Example

```
🔍 Searching for: 'find all profitability metrics'

Found 3 relevant results:

╭─ Result 1 ────────────────────────────────────────────╮
│ Concept: Gross Profit Margin                          │
│ Location: 'Revenue Analysis' sheet, cell C15          │
│ Value: 0.42                                            │
│ Formula: =(Revenue-COGS)/Revenue                       │
│ Relevance: 87.5%                                       │
│                                                        │
│ Matches concept: profitability, margin.                │
│ Contains ratio formula.                                │
│                                                        │
│ Context: This is a profitability metric, margin       │
│ calculation, percentage formula                        │
╰────────────────────────────────────────────────────────╯
```

## Advanced Features

### Query Expansion
Automatically expands queries with related terms:
- "profitability" → profit, margin, earnings, EBITDA
- "efficiency" → ROI, ROA, ROE, turnover, productivity

### Multi-Factor Ranking
Results are ranked using:
- **Semantic Similarity** (40%): Vector similarity to query
- **Keyword Matching** (20%): Direct text matches
- **Importance Score** (20%): Based on formula complexity and context
- **Formula Complexity** (10%): More sophisticated = more relevant
- **Concept Matching** (10%): Business concept alignment

### Concept Recognition
Recognizes 10+ business concept categories:
- Financial metrics (revenue, cost, profit, margin)
- Efficiency ratios (ROI, ROA, ROE, turnover)
- Growth metrics (YoY, QoQ, CAGR)
- Comparisons (budget, actual, variance)
- And more...

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Similarity Metric**: Cosine similarity
- **Supported Formats**: Excel (.xlsx, .xls), CSV (via pandas)
- **Formula Support**: Recognizes SUM, AVERAGE, IF, VLOOKUP, and custom formulas

## Limitations & Future Enhancements

### Current Limitations
- Single-sheet focus 
- English language only
- Limited to Excel/CSV formats

### Potential Enhancements
- Multi-sheet concept tracking
- Cross-sheet relationship mapping
- Support for Google Sheets API
- Custom business concept training
- Query history and learning
