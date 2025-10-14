"""
Semantic Search Engine for Spreadsheets
Main CLI Interface
"""

import argparse
import json
from pathlib import Path
from .spreadsheet_parser import SpreadsheetParser
from .semantic_analyzer import SemanticAnalyzer
from .search_engine import SearchEngine
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import numpy as np # Import numpy

console = Console()

# --- NEW: Add the Sigmoid function ---
def sigmoid(x):
    """Converts a raw score into a 0-1 probability."""
    return 1 / (1 + np.exp(-x))

class SemanticSpreadsheetSearch:
    def __init__(self):
        self.parser = SpreadsheetParser()
        self.analyzer = SemanticAnalyzer()
        self.search_engine = None
        self.indexed_file = None

    def index_spreadsheet(self, file_path: str):
        """Index a spreadsheet for semantic search"""
        console.print(f"\n[bold blue]📊 Indexing spreadsheet:[/bold blue] {file_path}")
        
        # Parse spreadsheet
        console.print("[yellow]→ Parsing spreadsheet...[/yellow]")
        parsed_data = self.parser.parse(file_path)
        
        # Analyze semantically
        console.print("[yellow]→ Analyzing content semantically...[/yellow]")
        semantic_data = self.analyzer.analyze(parsed_data)
        
        # Build search index
        console.print("[yellow]→ Building search index...[/yellow]")
        self.search_engine = SearchEngine()
        self.search_engine.build_index(semantic_data)
        self.indexed_file = file_path
        
        console.print(f"[bold green]✓ Successfully indexed {len(semantic_data['cells'])} cells[/bold green]\n")

    def search(self, query: str, top_k: int = 5, output_format: str = "human"):
        """Perform semantic search"""
        if not self.search_engine:
            console.print("[bold red]❌ No spreadsheet indexed! Please index a file first.[/bold red]")
            return
        
        console.print(f"\n[bold cyan]🔍 Searching for:[/bold cyan] '{query}'")
        
        results = self.search_engine.search(query, top_k=top_k)
        
        if output_format == "json":
            self._output_json(results)
        else:
            self._output_human(results, query)

    def _output_json(self, results):
        """Output results as JSON"""
        output = []
        for result in results:
            normalized_score = sigmoid(result["score"])
            output.append({
                "concept_name": result["concept"],
                "location": result["location"],
                "sheet": result["sheet"],
                "value": result["value"],
                "formula": result.get("formula"),
                "relevance_score": normalized_score, # Use normalized score
                "explanation": result["explanation"],
                "business_context": result["context"]
            })
        
        print(json.dumps(output, indent=2))

    def _output_human(self, results, query):
        """Output results in human-readable format"""
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        console.print(f"\n[bold green]Found {len(results)} relevant results:[/bold green]\n")
        
        for idx, result in enumerate(results, 1):
            # --- FIX: Normalize the score before displaying ---
            normalized_score = sigmoid(result['score'])

            # Create result panel
            content = []
            content.append(f"[bold cyan]Concept:[/bold cyan] {result['concept']}")
            content.append(f"[bold cyan]Location:[/bold cyan] {result['location']}")
            
            if result.get('value'):
                content.append(f"[bold cyan]Value:[/bold cyan] {result['value']}")
            
            if result.get('formula'):
                content.append(f"[bold cyan]Formula:[/bold cyan] {result['formula']}")
            
            # Use the normalized score for display
            content.append(f"[bold cyan]Relevance:[/bold cyan] {normalized_score:.2%}")
            content.append(f"\n[italic]{result['explanation']}[/italic]")
            
            if result.get('context'):
                content.append(f"\n[dim]Context: {result['context']}[/dim]")
            
            panel = Panel(
                "\n".join(content),
                title=f"Result {idx}",
                border_style="green" if idx == 1 else "blue",
                expand=False
            )
            console.print(panel)
            console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search Engine for Spreadsheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a spreadsheet
  python main.py --file data.xlsx --index

  # Search after indexing
  python main.py --file data.xlsx --search "find all profitability metrics"
  
  # Search with JSON output
  python main.py --file data.xlsx --search "show cost calculations" --format json
  
  # Limit results
  python main.py --file data.xlsx --search "efficiency ratios" --top 3
        """
    )
    
    parser.add_argument("--file", "-f", required=True, help="Path to spreadsheet file")
    parser.add_argument("--index", "-i", action="store_true", help="Index the spreadsheet")
    parser.add_argument("--search", "-s", type=str, help="Search query")
    parser.add_argument("--top", "-t", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--format", "-o", choices=["human", "json"], default="human", 
                       help="Output format (default: human)")
    
    args = parser.parse_args()
    
    # Initialize search system
    search_system = SemanticSpreadsheetSearch()
    
    # Always index first (in production, you'd save/load the index)
    if args.index or args.search:
        search_system.index_spreadsheet(args.file)
    
    # Perform search if requested
    if args.search:
        search_system.search(args.search, top_k=args.top, output_format=args.format)
    elif not args.index:
        console.print("[yellow]Please specify --index or --search[/yellow]")


if __name__ == "__main__":
    import os
    # It's better practice to handle environment variables in the main script
    # rather than the CLI module if they are needed for different parts of the app.
    # from dotenv import load_dotenv
    # load_dotenv()
    main()