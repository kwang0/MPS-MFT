import pickle
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert.py <pkl_file>")
        sys.exit(1)

    file = sys.argv[1]

    with open(file, "rb") as f:
        data = pickle.load(f)
    
    print("order_param: {}, gap: {}, mu: {}".format(data['order_param'], data['gap'], data['mu']))

if __name__ == "__main__":
    main()
