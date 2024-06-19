import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datanumber", help="number of the saving files name", type=int)
parser.add_argument("N", help="length of the array", type=int)
args = parser.parse_args()
arr=[random.randint(0, args.N) for _ in range(args.N)]
with open("array{}.txt".format(args.datanumber), "w") as file:
    for n in arr:
        file.write(str(n)+" ")
arr.sort()
with open("golden{}.txt".format(args.datanumber), "w") as file:
    for n in arr:
        file.write(str(n)+" ")
