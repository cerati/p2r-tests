export PSTL_USAGE_WARNINGS=1
export ONEDPL_USE_DPCPP_BACKEND=1

#clang++ -std=c++17 -fsycl -O2 src/propagate-tor-test_sycl.cpp -o test-sycl.exe -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20

clang++ -std=c++17 -fsycl -O2 src/propagate-tor-test_sycl_esimd.cpp -o test-sycl-esimd.exe -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=16 -Dnlayer=20

