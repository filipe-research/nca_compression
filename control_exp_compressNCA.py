import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='experiment name', required=True)
parser.add_argument('--conf', help='configuration file', required=True)
args = parser.parse_args()

print(f"o nome do experimento é {args.name}")
print(f"o nome do arquivo é {args.conf}")
