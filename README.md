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
#Build and run once with icc as compiler
python build.py -t tbb -c icc -v
```
Example commands:
```
#print out compile command
python build.py -t tbb -c icc -v --dryRun
#build and scan with multiple threads
python build.py -t tbb -c icc -v --nthreads 1,2,3,4,5
#Scan for two compilers with multiple threads
python build.py -t tbb -c icc,gcc -v --nthreads 1,2,3,4,5
#Append results to the same result json (Default is to skip existing scan points)
python build.py -t tbb -c icc -v --nthreads 1,2,3,4,5 --append
```
