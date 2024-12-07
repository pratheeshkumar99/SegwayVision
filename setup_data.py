import os
import subprocess
import sys
from pathlib import Path

def setup_dataset():
    """Download and setup the IDD dataset."""
    data_dir = Path('IDD_data')
    data_dir.mkdir(exist_ok=True)
    
    if not (data_dir / 'IDD').exists():
        print("Downloading IDD dataset...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/mohanrajmit/IDD.git",
            str(data_dir / "IDD")
        ])
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists!")
    
    required_paths = [
        data_dir / 'IDD/idd20k_lite/leftImg8bit',
        data_dir / 'IDD/idd20k_lite/gtFine'
    ]
    
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Missing required path: {path}")
    
    return str(data_dir / 'IDD/idd20k_lite')

if __name__ == "__main__":
    dataset_path = setup_dataset()
    print(f"Dataset ready at: {dataset_path}")
