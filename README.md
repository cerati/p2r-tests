# p2r-tests

## instructions to compile and run p2r on apollo@cs.uoregon.edu

```
module load intel

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/packages/intel/20/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/

icc -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high propagate-tor-test_tbb.cpp -I/packages/intel/20/compilers_and_libraries/linux/tbb/include/ -L/packages/intel/20/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/ -Wl,-rpath,/lib -ltbb -o propagate-tor-test.exe
```

add the following to create detailed optimization report: `-qopt-report=5`

```./propagate-tor-test.exe```


## instructions on cori
```
module load intel
module load tbb
python build.py -t tbb -c icc -v
```
