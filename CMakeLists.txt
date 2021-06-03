cmake_minimum_required(VERSION 3.12)
project(p2r_kokkos CXX) # C++ needed to build my project 
#find_package(Kokkos REQUIRED) # fail if Kokkos not found

add_subdirectory(kokkos)

# build my executable from the specified source code
add_executable(p2r_kokkos "./src/propagate-tor-test_kokkos.cpp")

# declare dependency on Kokkos
target_link_libraries(p2r_kokkos PRIVATE Kokkos::kokkos)
