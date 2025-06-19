import argparse
import os
from renumics import spotlight

def main():
    parser = argparse.ArgumentParser(description="Load and display a saved Spotlight HDF5 dataset")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the .h5 dataset file (e.g., visualization_datastore/docs_store.h5)")
    args = parser.parse_args()

    h5_file = args.h5_path
    if not os.path.isfile(h5_file):
        print(f"Error: File {h5_file} does not exist.")
        return

    print(f"Loading Spotlight dataset from {h5_file}...")
    spotlight.show(h5_file)

if __name__ == "__main__":
    main()
