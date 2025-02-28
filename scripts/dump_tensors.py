import sys
from safetensors import safe_open
from pathlib import Path

def dump_tensor_names(file_path):
    """Dump all tensor names from a safetensors file."""
    try:
        with safe_open(file_path, framework="pt") as f:
            tensor_names = f.keys()
            
        print(f"\nTensors in {Path(file_path).name}:")
        print("-" * 80)
        for i, name in enumerate(sorted(tensor_names), 1):
            print(f"{i:3d}. {name}")
        print(f"\nTotal tensors: {len(tensor_names)}")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dump_tensors.py <path_to_safetensors_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    dump_tensor_names(file_path)
