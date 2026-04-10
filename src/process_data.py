import argparse
import json
import os
import sys
from pathlib import Path

# Add the current directory to sys.path to allow imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking import ChunkingStrategyComparator

def process_file(file_path: Path, output_dir: Path, chunk_size: int):
    """Read a file, chunk it using all strategies, and save as JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            print(f"Skipping empty file: {file_path.name}")
            return

        comparator = ChunkingStrategyComparator()
        results = comparator.compare(text, chunk_size=chunk_size)
        
        # Save results to a JSON file named after the input file
        output_path = output_dir / f"{file_path.stem}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed: {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process raw text data into chunks using multiple strategies.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/raw_data", 
        help="Path to the directory containing raw text/markdown files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed_data", 
        help="Path to the directory where chunked JSON results will be saved."
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=500, 
        help="Target chunk size in characters (used by fixed and recursive strategies)."
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown and text files
    files = list(input_path.glob("*.md")) + list(input_path.glob("*.txt"))
    
    if not files:
        print(f"No .md or .txt files found in {args.input_dir}")
        return

    print(f"Starting processing of {len(files)} files...")
    for file_path in files:
        process_file(file_path, output_path, args.chunk_size)
    
    print(f"\nProcessing complete. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
