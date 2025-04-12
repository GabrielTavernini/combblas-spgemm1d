project(CombBLAS_Benchmark LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# include(mkl/CMakeLists.txt)
include(ctf/CMakeLists.txt)
include(petsc/CMakeLists.txt)