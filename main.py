# Save this file as interactive_search.py

import os
from spreadsheet_search_engine.cli import SemanticSpreadsheetSearch
from rich.console import Console
from rich.panel import Panel

# Initialize the console for pretty printing
console = Console()

def interactive_mode():
    """Runs the semantic search engine in an interactive mode."""

    # 1. Print a welcome message
    console.print(Panel(
        "[bold cyan]Welcome to Interactive Spreadsheet Search! 🔎[/bold cyan]",
        subtitle="Type 'exit' or 'quit' to close the program.",
        border_style="blue"
    ))

    # 2. Get a valid spreadsheet file path from the user
    file_path = ""
    while True:
        prompt = "\n[bold]Enter the path to your spreadsheet file[/bold] (e.g., sample_financial_data.xlsx): "
        file_path = console.input(prompt)
        if os.path.exists(file_path):
            break
        else:
            console.print(f"\n[bold red]❌ Error: File not found at '[i]{file_path}[/i]'. Please try again.[/bold red]")

    # 3. Initialize and index the spreadsheet
    # This reuses the entire engine from your existing code.
    try:
        search_system = SemanticSpreadsheetSearch()
        search_system.index_spreadsheet(file_path)
    except Exception as e:
        console.print(f"[bold red]An error occurred during indexing: {e}[/bold red]")
        return

    console.print("\n[bold green]✅ Indexing complete! You can now start searching.[/bold green]")

    # 4. Start the interactive search loop
    while True:
        try:
            query = console.input("\n[bold cyan]Search > [/bold cyan]")

            # Check for exit command
            if query.lower().strip() in ["exit", "quit"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            # Perform the search if the query is not empty
            if query.strip():
                search_system.search(query=query, top_k=3) # Searching for top 3 results
            
        except KeyboardInterrupt: # Handle Ctrl+C gracefully
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break

if __name__ == "__main__":
    # The Hugging Face token is needed for the underlying sentence-transformer model.
    # Make sure this environment variable is set.
    from dotenv import load_dotenv
    load_dotenv()
         
    interactive_mode()