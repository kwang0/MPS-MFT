#!/usr/bin/env python3
import pickle
import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert.py <old_pkl_file> <new_t_p>")
        sys.exit(1)

    old_file = sys.argv[1]
    new_tp = float(sys.argv[2])

    if not os.path.exists(old_file):
        print(f"Error: file {old_file} does not exist.")
        sys.exit(1)

    # Load old checkpoint
    with open(old_file, "rb") as f:
        data = pickle.load(f)

    # Update t_p and reset completed
    old_tp = data.get("t_p", None)
    U = data.get("U", None)
    data["t_p"] = new_tp
    data["completed"] = False

    # Build new filename
    new_file = f"results_U_{U}_t_p_{new_tp}.pkl"

    # Save checkpoint
    with open(new_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Converted {old_file} (t_p={old_tp}) → {new_file} (t_p={new_tp}, completed=False)")

if __name__ == "__main__":
    main()

# ex. python convert.py results_U_-4.0_t_p0.2.pkl 0.3