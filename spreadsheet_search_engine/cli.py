"""
Enhanced Semantic Search Engine for Spreadsheets
Main CLI Interface with Advanced Features
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
from rich.tree import Tree
import numpy as np
from typing import Dict

console = Console()


def sigmoid(x):
    """Converts a raw score into a 0-1 probability."""
    return 1 / (1 + np.exp(-x))


class SemanticSpreadsheetSearch:
    def __init__(self):
        self.parser = SpreadsheetParser()
        self.analyzer = SemanticAnalyzer()
        self.search_engine = None
        self.indexed_file = None
        self.parsed_data = None

    def index_spreadsheet(self, file_path: str):
        """Index a spreadsheet for semantic search"""
        console.print(f"\n[bold blue]📊 Indexing spreadsheet:[/bold blue] {file_path}")
        
        # Parse spreadsheet
        console.print("[yellow]→ Parsing spreadsheet...[/yellow]")
        parsed_data = self.parser.parse(file_path)
        self.parsed_data = parsed_data
        
        # Show parsing stats
        self._show_parsing_stats(parsed_data)
        
        # Analyze semantically
        console.print("[yellow]→ Analyzing content semantically...[/yellow]")
        semantic_data = self.analyzer.analyze(parsed_data)
        
        # Build search index
        console.print("[yellow]→ Building search index...[/yellow]")
        self.search_engine = SearchEngine()
        self.search_engine.build_index(semantic_data)
        self.indexed_file = file_path
        
        console.print(f"[bold green]✓ Successfully indexed {len(semantic_data['cells'])} cells[/bold green]\n")

    def _show_parsing_stats(self, parsed_data: Dict):
        """Display parsing statistics"""
        stats = Table(title="📋 Spreadsheet Analysis", show_header=True, header_style="bold cyan")
        stats.add_column("Metric", style="cyan")
        stats.add_column("Count", justify="right", style="green")
        
        stats.add_row("Sheets", str(parsed_data['metadata']['sheet_count']))
        stats.add_row("Total Cells", str(len(parsed_data['cells'])))
        stats.add_row("Formulas", str(len(parsed_data['formulas'])))
        stats.add_row("Named Ranges", str(len(parsed_data.get('named_ranges', {}))))
        
        # Dependency stats
        if parsed_data.get('dependency_graph'):
            G = parsed_data['dependency_graph']
            stats.add_row("Dependency Edges", str(G.number_of_edges()))
            stats.add_row("Connected Cells", str(G.number_of_nodes()))
        
        # Temporal stats
        temporal_sheets = [s for s, info in parsed_data.get('temporal_info', {}).items() 
                          if info.get('has_time_series')]
        if temporal_sheets:
            stats.add_row("Time Series Sheets", str(len(temporal_sheets)))
        
        # Hierarchy stats
        hierarchy_sheets = list(parsed_data.get('hierarchies', {}).keys())
        if hierarchy_sheets:
            stats.add_row("Sheets with Hierarchies", str(len(hierarchy_sheets)))
        
        console.print(stats)
        console.print()

    def search(self, query: str, top_k: int = 5, output_format: str = "human", 
               show_dependencies: bool = False):
        """Perform semantic search"""
        if not self.search_engine:
            console.print("[bold red]❌ No spreadsheet indexed! Please index a file first.[/bold red]")
            return
        
        console.print(f"\n[bold cyan]🔎 Searching for:[/bold cyan] '{query}'")
        
        results = self.search_engine.search(query, top_k=top_k)
        
        if output_format == "json":
            self._output_json(results)
        else:
            self._output_human(results, query, show_dependencies)

    def show_dependencies(self, cell_ref: str, sheet: str):
        """Show dependency information for a specific cell"""
        if not self.search_engine or not self.search_engine.dependency_graph:
            console.print("[bold red]❌ No dependency information available[/bold red]")
            return
        
        summary = self.search_engine.get_dependency_summary(cell_ref, sheet)
        
        if not summary:
            console.print(f"[yellow]No dependency information found for {sheet}!{cell_ref}[/yellow]")
            return
        
        # Create a tree visualization
        tree = Tree(f"[bold cyan]{sheet}!{cell_ref}[/bold cyan]")
        
        # Add dependencies branch
        if summary['direct_dependencies']:
            dep_branch = tree.add("[yellow]📥 Direct Dependencies[/yellow]")
            for dep in summary['direct_dependencies'][:10]:
                dep_branch.add(f"[dim]{dep}[/dim]")
            if len(summary['direct_dependencies']) > 10:
                dep_branch.add(f"[dim]... and {len(summary['direct_dependencies']) - 10} more[/dim]")
        
        # Add impacts branch
        if summary['direct_impacts']:
            impact_branch = tree.add("[green]📤 Direct Impacts[/green]")
            for impact in summary['direct_impacts'][:10]:
                impact_branch.add(f"[dim]{impact}[/dim]")
            if len(summary['direct_impacts']) > 10:
                impact_branch.add(f"[dim]... and {len(summary['direct_impacts']) - 10} more[/dim]")
        
        console.print(tree)
        
        # Add summary table
        summary_table = Table(show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Dependencies (recursive)", str(summary['total_dependencies']))
        summary_table.add_row("Total Impacts (recursive)", str(summary['total_impacts']))
        summary_table.add_row("Dependency Depth", str(summary['dependency_depth']))
        summary_table.add_row("Impact Breadth", str(summary['impact_breadth']))
        
        console.print("\n")
        console.print(summary_table)

    def show_temporal_info(self):
        """Show temporal/time series information"""
        if not self.parsed_data or not self.parsed_data.get('temporal_info'):
            console.print("[yellow]No temporal information detected[/yellow]")
            return
        
        for sheet_name, info in self.parsed_data['temporal_info'].items():
            if info['has_time_series']:
                panel = Panel(
                    self._format_temporal_info(info),
                    title=f"⏰ Time Series: {sheet_name}",
                    border_style="cyan"
                )
                console.print(panel)

    def _format_temporal_info(self, info: Dict) -> str:
        """Format temporal information for display"""
        lines = []
        lines.append(f"[bold]Period Type:[/bold] {info.get('period_type', 'Unknown')}")
        lines.append(f"[bold]Time Columns:[/bold]")
        for col in info.get('time_columns', []):
            lines.append(f"  • {col['column']}: {col['header']} ({col['type']})")
        return "\n".join(lines)

    def show_hierarchies(self):
        """Show hierarchy information"""
        if not self.parsed_data or not self.parsed_data.get('hierarchies'):
            console.print("[yellow]No hierarchies detected[/yellow]")
            return
        
        for sheet_name, hierarchies in self.parsed_data['hierarchies'].items():
            tree = Tree(f"[bold cyan]📊 {sheet_name}[/bold cyan]")
            
            if hierarchies.get('total_rows'):
                total_branch = tree.add("[green]Totals[/green]")
                for total in hierarchies['total_rows']:
                    total_branch.add(f"Row {total['row']}: {total['label']}")
            
            if hierarchies.get('subtotal_rows'):
                subtotal_branch = tree.add("[yellow]Subtotals[/yellow]")
                for subtotal in hierarchies['subtotal_rows']:
                    subtotal_branch.add(f"Row {subtotal['row']}: {subtotal['label']}")
            
            if hierarchies.get('categories'):
                cat_branch = tree.add("[blue]Aggregations[/blue]")
                for cat in hierarchies['categories'][:5]:
                    cat_branch.add(f"{cat['parent_cell']}: SUM({cat['aggregates']})")
            
            console.print(tree)
            console.print()

    def _output_json(self, results):
        """Output results as JSON"""
        output = []
        for result in results:
            normalized_score = sigmoid(result["score"])
            output_item = {
                "concept_name": result["concept"],
                "location": result["location"],
                "sheet": result["sheet"],
                "value": result["value"],
                "formula": result.get("formula"),
                "relevance_score": normalized_score,
                "explanation": result["explanation"],
                "business_context": result["context"]
            }
            
            # Add dependency info if available
            if result.get('dependencies'):
                output_item['dependencies'] = result['dependencies']
            
            # Add relationship info if available
            if result.get('relationship'):
                output_item['relationship'] = result['relationship']
            
            # Add aggregation info if available
            if result.get('aggregation_type'):
                output_item['aggregation_type'] = result['aggregation_type']
            
            output.append(output_item)
        
        print(json.dumps(output, indent=2))

    def _output_human(self, results, query, show_dependencies=False):
        """Output results in human-readable format"""
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        console.print(f"\n[bold green]Found {len(results)} relevant results:[/bold green]\n")
        
        for idx, result in enumerate(results, 1):
            normalized_score = sigmoid(result['score'])

            # Create result panel
            content = []
            content.append(f"[bold cyan]Concept:[/bold cyan] {result['concept']}")
            content.append(f"[bold cyan]Location:[/bold cyan] {result['location']}")
            
            if result.get('value'):
                content.append(f"[bold cyan]Value:[/bold cyan] {result['value']}")
            
            if result.get('formula'):
                content.append(f"[bold cyan]Formula:[/bold cyan] {result['formula']}")
            
            content.append(f"[bold cyan]Relevance:[/bold cyan] {normalized_score:.2%}")
            
            # Add dependency info
            if result.get('dependencies'):
                deps = result['dependencies']
                content.append(f"[bold cyan]Dependencies:[/bold cyan] {deps['depends_on_count']} inputs, {deps['impacts_count']} outputs")
            
            # Add relationship info for dependency queries
            if result.get('relationship'):
                content.append(f"[bold cyan]Relationship:[/bold cyan] {result['relationship']}")
            
            # Add aggregation info
            if result.get('aggregation_type'):
                content.append(f"[bold cyan]Aggregation:[/bold cyan] {result['aggregation_type']}")
            
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
        description="Enhanced Semantic Search Engine for Spreadsheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python main.py --file data.xlsx --search "find all profitability metrics"
  
  # Aggregation query
  python main.py --file data.xlsx --search "sum of all revenue"
  
  # Dependency analysis
  python main.py --file data.xlsx --search "what affects net profit"
  python main.py --file data.xlsx --deps "A10" --sheet "Revenue"
  
  # Temporal analysis
  python main.py --file data.xlsx --search "revenue growth trend"
  python main.py --file data.xlsx --temporal
  
  # View structure
  python main.py --file data.xlsx --hierarchies
  
  # JSON output
  python main.py --file data.xlsx --search "efficiency ratios" --format json
        """
    )
    
    parser.add_argument("--file", "-f", required=True, help="Path to spreadsheet file")
    parser.add_argument("--index", "-i", action="store_true", help="Index the spreadsheet")
    parser.add_argument("--search", "-s", type=str, help="Search query")
    parser.add_argument("--top", "-t", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--format", "-o", choices=["human", "json"], default="human", 
                       help="Output format (default: human)")
    parser.add_argument("--deps", type=str, help="Show dependencies for a cell (e.g., A10)")
    parser.add_argument("--sheet", type=str, help="Sheet name (required with --deps)")
    parser.add_argument("--temporal", action="store_true", help="Show temporal/time series information")
    parser.add_argument("--hierarchies", action="store_true", help="Show hierarchy information")
    parser.add_argument("--show-deps", action="store_true", help="Show dependency info in search results")
    
    args = parser.parse_args()
    
    # Initialize search system
    search_system = SemanticSpreadsheetSearch()
    
    # Always index first
    if args.index or args.search or args.deps or args.temporal or args.hierarchies:
        search_system.index_spreadsheet(args.file)
    
    # Handle different commands
    if args.deps:
        if not args.sheet:
            console.print("[bold red]Error: --sheet is required with --deps[/bold red]")
            return
        search_system.show_dependencies(args.deps, args.sheet)
    
    elif args.temporal:
        search_system.show_temporal_info()
    
    elif args.hierarchies:
        search_system.show_hierarchies()
    
    elif args.search:
        search_system.search(
            args.search, 
            top_k=args.top, 
            output_format=args.format,
            show_dependencies=args.show_deps
        )
    
    elif not args.index:
        console.print("[yellow]Please specify --index, --search, --deps, --temporal, or --hierarchies[/yellow]")


if __name__ == "__main__":
    main()