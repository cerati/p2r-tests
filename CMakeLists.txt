cmake_minimum_required(VERSION 3.12)
project(p2r_kokkos CXX) # C++ needed to build my project 
#find_package(Kokkos REQUIRED) # fail if Kokkos not found

add_subdirectory(kokkos)

# build my executable from the specified source code
#add_executable(p2r_kokkos "./src/propagate-tor-test_kokkos.cpp")
add_executable(p2r_kokkos "./src/propagate-tor-test_kokkos_local.cpp")

set(CMAKE_BUILD_TYPE Debug)

# declare dependency on Kokkos, build p2r together with Kokkos, inheriting all the kokkos flags
#target_link_libraries(p2r_kokkos PRIVATE Kokkos::kokkos)    ## standalone build
target_link_libraries(p2r_kokkos Kokkos::kokkos)             ## JLSE build
