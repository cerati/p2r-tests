#!/usr/bin/env python3

import re
import sys
import glob
import argparse
import itertools
import subprocess
import collections

prefix = "propagate-tor-test"

compilers = [
    "gcc",
    "icc"
]
technologies = {
    "tbb": ["icc","gcc"]
}

# with default values
scanParameters = [
    ("ntrks", 9600),
    ("nevts", 100),
    ("NITER", 5),
    ("bsize", 128),
    ("nlayer", 20),
    ("nthreads", 1)
]
ScanPoint = collections.namedtuple("ScanPoint", [x[0] for x in scanParameters])

result_re = re.compile("done ntracks=(?P<ntracks>\d+) tot time=(?P<time>\S+) ")

def compilationCommand(compiler, technology, target, source, scanPoint):
    cmd = []
    if compiler == "gcc":
        cmd.extend(["g++", "-Wall", "-Isrc", "-O3", "-fopenmp", "-march=native"])

    if compiler == "icc":
        cmd.extend(["icc", "-Wall", "-Isrc", "-O3", "-fopenmp", "-march=native",'-xHost','-qopt-zmm-usage=high'])

    cmd.extend(["-o", target, source])
        
    if technology == "tbb":
        cmd.append("-ltbb")

    cmd.extend(["-D{}={}".format(name, getattr(scanPoint, name)) for name in ScanPoint._fields])
        
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

def build(opts, src, compiler, technology, scanPoint):
    target = "{}_{}_{}".format(prefix, technology, compiler)
    print("Building {} for {} with {} for {} as {}".format(src, technology, compiler, scanPoint, target))
    cmd = compilationCommand(compiler, technology, target, src, scanPoint)
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

def run(opts, exe, scanPoint):
    print("Running {} for {}".format(exe, scanPoint))
    cmd = ["./"+exe]
    if opts.verbose or opts.dryRun:
        print(" ".join(cmd))
        if opts.dryRun:
            return
        
    out = execute(cmd, opts.verbose)
    if opts.verbose:
        for line in out.split("\n"): print(line)
    try:
        return throughput(out)
    except Exception as e:
        print("Caught exception, printout of the program", " ".join(cmd))
        print(out)
        raise

def main(opts):
    fname_re = re.compile(prefix+"_(?P<tech>.*)\.cpp")

    sources = sorted(glob.glob("src/*.cpp"))

    for source in sources:
        m = fname_re.search(source)
        if not m:
            raise Exception("Source file name {} does not follow the expected pattern".format(source))
        tech = m.group("tech")

        if len(opts.technologies) > 0 and tech not in opts.technologies:
            print("Skipping", tech)
            continue
        compilers = technologies[tech]
        for comp in compilers:
            if len(opts.compilers) > 0 and comp not in opts.compilers:
                print("Skipping", comp)
                continue

            for p in scanProduct(opts):
                scanPoint = ScanPoint(*p)
                print()
                try:
                    exe = build(opts, source, comp, tech, scanPoint)
                except ExeException as e:
                    return e.errorCode()

                if opts.build:
                    continue

                try:
                    throughput = run(opts, exe, scanPoint)
                except ExeException as e:
                    return e.errorCode()
                if opts.dryRun:
                    continue
                print("Throughput {} tracks/second".format(throughput))

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and run")

    parser.add_argument("--dryRun", action="store_true",
                        help="Print out commands, don't actually run anything")
    parser.add_argument("-b", "--build", action="store_true",
                        help="Build only (default is to build and run)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print out compilation commands and compiler outputs")

    parser.add_argument("-c", "--compilers", type=str, default="gcc",
                        help="Comma separated list of compilers, default 'gcc' ({})".format(",".join(sorted(compilers))))
    parser.add_argument("-t", "--technologies", type=str, default="",
                        help="Comma separated list of technologies, default is all ({})".format(",".join(sorted(technologies.keys()))))

    for par, default in scanParameters:
        parser.add_argument("--"+par, type=str, default=str(default),
                            help="Comma separated list of {} values (default {})".format(par, str(default)))

    opts = parser.parse_args()
    if opts.compilers == "":
        opts.compilers = []
    else:
        opts.compilers = opts.compilers.split(",")
    if opts.technologies == "":
        opts.technologies = []
    else:
        opts.technologies = opts.technologies.split(",")

    for par, default in scanParameters:
        setattr(opts, par, getattr(opts, par).split(","))

    sys.exit(main(opts))
