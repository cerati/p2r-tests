import json
import re
import sys
import argparse
import numpy as np
from build import throughput,result_re

def main(options):
    result = []
    with open(options.input,'r') as f:
        for line in f:
            m = result_re.search(line)
            if m:
               #result.append( int(m.group("ntracks"))/float(m.group("time")))
               result.append( float(m.group("time"))/int(m.group("ntracks")))
    result = np.array(result)
    print(len(result))
    if len(result)>20:
        for i in [0,20,40,60,80]:
            print("result: ",i)
            print("    n               = %s"%len(result[i:i+20]))

            print("    arr = ")
            print(repr(result[i:i+20]))
            print("    mean time/track = %.3e"%np.mean(result[i:i+20]))
            print("    std time/track  = %.2e"%np.std(result[i:i+20]))
    else:
        print("result: ")
        print("    n               = %s"%len(result))
        print("    mean time/track = %.3e"%np.mean(result))
        print("    std time/track  = %.2e"%np.std(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and run")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Suffix of output JSON and log files. If the output JSON file exists, it will be updated (see also --overwrite) (default: '')")
    parser.add_argument("-i", "--input", type=str, default="slurm.out",
                        help="slurm output file")
    opts = parser.parse_args()
    sys.exit(main(opts))
