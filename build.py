#!/usr/bin/env python3

import re
import os
import sys
import glob
import json
import argparse
import itertools
import subprocess
import collections

prefix = "propagate-tor-test"
#prefix = "propagate-toz-test"


technologies = {
    "tbb": {
       "threads":["icc","gcc"],
    },
    "cuda_v2":{
        "cuda":['nvcc']
    },
    "cuda_v3":{
        "cuda":['nvcc']
    },
    "cuda_v4":{
        "cuda":['nvcc']
    },
    "cuda":{
        "cuda":['nvcc']
    },
    "acc.v3":{
        "cuda":['nvc++'],
        "cpu":['nvc++']
     },
    "pstl":{
        "cpu":['gcc'], # add other compilers
        'cuda': ['nvc++','nvc++_x86']
    },
    "hip":{
        "hip":['hipcc'],
    }
    #"kokkos": {
    #  "serial": ["icc", "gcc"],
    #  "threads": ["icc", "gcc"],
    #  "cuda": ["nvcc"],
    #  "hip": ["hipcc"]
    #}
}
cmds ={
    #"tbb":{"threads":["srun","-n","1",'-c','40',"numactl", "--cpunodebind=1"]},
    "tbb":{"threads":[]},
    #"cuda":{"cuda":["srun","-n","1","-c","80","--exclusive","numactl","--cpunodebind=0"]}
    "cuda":{"cuda":["srun","-n","1"]},
    #"cuda_v2":{"cuda":["srun","-n","1","-c","80"]}
    "cuda_v2":{"cuda":["srun","-n","1"]},
    "cuda_v3":{"cuda":["srun","-n","1"]},
    "cuda_v4":{"cuda":["srun","-n","1"]},
    "acc.v3":{"cuda":["srun","-n","1"],
             "cpu":["srun","-n","1"]},
    "pstl":{"cuda":["srun","-n","1"],
            "cpu":["srun","-n","1"]},
    "cuda_v4":{"cuda":["srun","-n","1"]},
    "hip":{"hip":[]},
}
# with default values
scanParameters = [
    ("ntrks", 8192),
    ("nevts", 100),
    ("NITER", 5),
    ("bsize", 32),
    ("nlayer", 20),
    ("nthreads", 1),
    ("num_streams", 1),
    #("threadsperblock", 1000),
    ("threadsperblockx", 32),
    #("threadsperblocky", 16),
    #("blockspergrid", 40)
]
ScanPoint = collections.namedtuple("ScanPoint", [x[0] for x in scanParameters])

result_re = re.compile("done ntracks=(?P<ntracks>\d+) tot time=(?P<time>\S+) ")

def compilationCommand(compiler, technology, backend, target, source, scanPoint,opts):
    cmd = []
    if compiler == "gcc":
        if technology=="pstl":
            cmd.extend(["g++", "-Wall", "-Isrc", "-O3", "-fopenmp", "-march=native", "-std=c++17","-mavx512f",'-lm',"-lgomp","-ltbb"])
        else:
            cmd.extend(["g++", "-Wall", "-Isrc", "-O3", "-fopenmp", "-march=native", "-std=c++17","-ffast-math"])

    if compiler == "icc":
        cmd.extend(["icc", "-Wall", "-Isrc", "-O3", "-fopenmp", "-march=native",'-xHost','-qopt-zmm-usage=high'])

    if compiler == "nvcc":
        cmd.extend(["nvcc",'-arch=sm_70',"-Iinclude","-std=c++17",'-maxrregcount=64','-g','-lineinfo'])
    if compiler == "nvc++":
        cmd.extend(["nvc++","-Iinclude","-O2","-std=c++17","-stdpar=gpu","-gpu=cc70","-gpu=managed","-gpu=fma","-gpu=fastmath","-gpu=autocollapse","-gpu=loadcache:L1","-gpu=unroll"])
    if compiler == "nvc++_x86":
        cmd.extend(["nvc++","-Iinclude","-O2","-std=c++17","-stdpar=multicore"])

    if technology=="acc.v3":
        if backend=="cuda":
            cmd.extend(["nvc++","-Iinclude","-std=c++17",'-acc','-fast','-gpu=fastmath','-gpu=cc70',"-DNUM_WORKERS=8","-Msafeptr"])
        elif backend=="cpu":
            cmd.extend(["nvc++","-Iinclude","-std=c++17",'-acc=multicore'
                        ,'-fast','-Mfprelaxed',"-Msafeptr",'-Mvect=simd:256',"-DNUM_WORKERS=1"])
    if compiler == "hipcc":
        cmd.extend(["hipcc","-Iinclude","-std=c++17"])

    cmd.extend(["-o", target, source])
        
    if technology == "tbb" :
        cmd.append("-ltbb")

    cmd.extend(["-D{}={}".format(name, getattr(scanPoint, name)) for name in ScanPoint._fields])
    if opts.noH2D:
        cmd.extend(["-DEXCLUDE_H2D_TRANSFER"])
    if opts.noD2H:
        cmd.extend(["-DEXCLUDE_D2H_TRANSFER"])
        
    return cmd

######
class ExeException(Exception):
    def __init__(self, code):
        super(ExeException, self).__init__()
        self._code = code

    def errorCode(self):
        return self._code

def scanProduct(opts):
    return itertools.product(*[getattr(opts, x[0]) for x in scanParameters])


def execute(command, verbose):    
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    try:
        (out, err) = p.communicate()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        (out, err) = p.communicate()
    if p.returncode != 0:
        if not verbose:
            print(" ".join(command))
        print(out)
        raise ExeException(p.returncode)
    return out

def build(opts, src, compiler, technology, backend, scanPoint):
    target = "{}_{}_{}_{}".format(prefix, technology,backend, compiler)
    print("Building {} for {} with {} for {} as {}".format(src, technology, compiler, scanPoint, target))
    cmd = compilationCommand(compiler, technology,backend, target, src, scanPoint, opts)
    if opts.verbose or opts.dryRun:
        print(" ".join(cmd))

    if not opts.dryRun:
        execute(cmd, opts.verbose)
    return target

def throughput(log):
    for line in log.split("\n"):
        m = result_re.search(line)
        if m:
            return int(m.group("ntracks"))/float(m.group("time"))
    raise Exception("No result in output")

def sysinfo(opts):
    cmd = ["perl","/global/homes/k/kkwok/PPS/PPSwork/misc/sysinfo.pl"]
    out = execute(cmd,opts.verbose)
    out = out.replace('"','').strip().split("\n")
    return out 

def run(opts, exe, tech, backend, scanPoint):
    print("Running {} for {}".format(exe, scanPoint))
    #cmd = ["./"+exe]
    cmd_prefix = cmds[tech][backend]
    cmd = cmd_prefix+["./"+exe] if len(cmd_prefix)>0 else ["./"+exe]
    if opts.verbose or opts.dryRun:
        print(" ".join(cmd))
        if opts.dryRun:
            return
  
    result = {} 
    for name in scanPoint._fields:
        result.update({name:float(getattr(scanPoint,name))})
    out = execute(cmd, opts.verbose)
    if opts.verbose:
        for line in out.split("\n"): print(line)
    try:
        result['throughput']=throughput(out)
        return result 
    except Exception as e:
        print("Caught exception, printout of the program", " ".join(cmd))
        print(out)
        raise

def main(opts):
    fname_re = re.compile(prefix+"_(?P<tech>.*)\.(cpp|cu)")

    sources = sorted(glob.glob("src/*.cu")+glob.glob("src/*.cpp"))

    for source in sources:
        m = fname_re.search(source)
        if not m:
#            raise Exception("Source file name {} does not follow the expected pattern".format(source))
            continue
        tech = m.group("tech")

        print(source)
        if len(opts.technologies) > 0 and tech not in opts.technologies:
            print("Skipping", tech)
            continue
        backends = technologies[tech]
        for backend,compilers in backends.items():
            if len(opts.backends) > 0 and backend not in opts.backends:
                print("Skipping", backend)
                continue

            for comp in compilers:
                if len(opts.compilers) > 0 and comp not in opts.compilers:
                    print("Skipping", comp)
                    continue

                data = dict(
                    backend=backend,
                    compiler=comp,
                    results=[]
                )
                data["sysinfo"] = sysinfo(opts)
                print(comp,backend)
                outputJson = "result_{}.json".format("_".join(filter(None,[tech,backend,comp,opts.output])))
                alreadyExists = set()
                if not opts.overwrite and os.path.exists(outputJson):
                    with open(outputJson) as inp:
                        data = json.load(inp)
                if not opts.append:
                    for res in data["results"]:
                        alreadyExists.add( tuple([res[k] for k in sorted(ScanPoint._fields) ]) )
  
                for p in scanProduct(opts):
                    scanPoint = ScanPoint(*p)
                    scanPoint_tuple = tuple([getattr(scanPoint,name) for name in sorted(ScanPoint._fields)])
                    if scanPoint_tuple in alreadyExists and not opts.dryRun:
                        print('Alread found this point in result, skipping:', scanPoint_tuple)
                        continue 
                    try:
                        exe = build(opts, source, comp, tech,backend, scanPoint)
                    except ExeException as e:
                        return e.errorCode()

                    if opts.build:
                        continue

                    try:
                        result = run(opts, exe, tech, backend, scanPoint)
                        data["results"].append(result)
                    except ExeException as e:
                        return e.errorCode()
                    if opts.dryRun:
                        continue
                    print("Throughput {} tracks/second".format(result['throughput']))
                if not opts.dryRun:
                    with open(outputJson, "w") as out:
                        json.dump(data, out, indent=2)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and run")

    parser.add_argument("--dryRun", action="store_true",
                        help="Print out commands, don't actually run anything")
    parser.add_argument("-b", "--build", action="store_true",
                        help="Build only (default is to build and run)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out compilation commands and compiler outputs")

    parser.add_argument("-c", "--compilers", type=str, default="",
                        help="Comma separated list of compilers, default is all compilers for each technology")
    parser.add_argument("--backends", type=str, default="",
                        help="Comma separated list of backends, default is all backends for each technology")
    parser.add_argument("-t", "--technologies", type=str, default="",
                        help="Comma separated list of technologies, default is all ({})".format(",".join(sorted(technologies.keys()))))
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Suffix of output JSON and log files. If the output JSON file exists, it will be updated (see also --overwrite) (default: '')")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output JSON instead of updating it")
    parser.add_argument("--append", action="store_true",
                        help="Append new (stream, threads) results insteads of ignoring already existing point")
    parser.add_argument("--noH2D", action="store_true",
                        help="exclude H2D transfer in measured time")
    parser.add_argument("--noD2H", action="store_true",
                        help="exclude D2H transfer in measured time")


    for par, default in scanParameters:
        parser.add_argument("--"+par, type=str, default=str(default),
                            help="Comma separated list of {} values (default {})".format(par, str(default)))

    opts = parser.parse_args()
    if opts.compilers == "":
        opts.compilers = []
    else:
        opts.compilers = opts.compilers.split(",")
    if opts.backends == "":
        opts.backends = []
    else:
        opts.backends = opts.backends.split(",")

    if opts.technologies == "":
        opts.technologies = []
    else:
        opts.technologies = opts.technologies.split(",")

    for par, default in scanParameters:
        setattr(opts, par, getattr(opts, par).split(","))

    sys.exit(main(opts))
