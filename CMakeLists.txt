cmake_minimum_required(VERSION 3.21)
project(CombBLAS VERSION 2.0.1 LANGUAGES C CXX)
# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Main CombBLAS library
add_library(CombBLAS src/CommGrid.cpp src/mmio.c src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)

# fast matrix market support for sequential read/write
add_subdirectory(fast_matrix_market_yuxi)
target_link_libraries(CombBLAS PUBLIC fast_matrix_market::fast_matrix_market)

# set include directories
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PRIVATE include/CombBLAS)

# metis
add_subdirectory(parmetis-4.0.3)
target_link_libraries(CombBLAS PUBLIC parmetis)
# MPI and OpenMP dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
  target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
endif()

if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(usort)
target_link_libraries(CombBLAS PUBLIC Usortlib)

add_subdirectory(graph500-1.2/generator)
target_link_libraries(CombBLAS PUBLIC GraphGenlib)

########### Some Options For Debug And Analysis ################
option(RDMA_DEBUG "debug RDMA, print some stats info" OFF)
if(RDMA_DEBUG)
target_compile_definitions(CombBLAS PUBLIC RDMA_DEBUG)
endif()
option(COMM_ANA "Communication [symbolic] analysis, disable computation" OFF)
if(COMM_ANA)
target_compile_definitions(CombBLAS PUBLIC COMM_ANA)
endif()
option(SKIP_COMP "in Comm ana, SKIP COMP for lage dataset" OFF)
if(SKIP_COMP AND NOT COMM_ANA)
message(FATAL_ERROR "SKIP COMP is only valid when activating COMM_ANA!!")
endif()
if(COMM_ANA AND SKIP_COMP)
target_compile_definitions(CombBLAS PUBLIC SKIP_COMP)
endif()
option(SKIP_COMM "in Comm ana, SKIP COMP for lage dataset" OFF)
if(SKIP_COMM AND NOT COMM_ANA)
message(FATAL_ERROR "SKIP COMM is only valid when activating COMM_ANA!!")
endif()
if(COMM_ANA AND SKIP_COMM)
target_compile_definitions(CombBLAS PUBLIC SKIP_COMM)
endif()
# MAT1D_DEBUG_LEVEL
target_compile_definitions(CombBLAS PUBLIC MAT1D_DEBUG_LEVEL=1)






# add_subdirectory(ReleaseTests)
add_subdirectory(Applications)
# add_subdirectory(Applications/Ordering)
# add_subdirectory(Applications/BipartiteMatchings)
# add_subdirectory(Applications/SpMSpV-IPDPS2017)
add_subdirectory(3DSpGEMM)

add_subdirectory(SSMCL)
# add_subdirectory(Benchmarks)

# target_link_libraries(CombBLAS PUBLIC tbb)
# Set up exported configuration
# This allows CombBLAS to be installed in two ways:
#   1. In /usr/local (or whatever prefix is specified)
#   2. Exporting the current build directory. This allows a user to make
#      modifications to CombBLAS and have the changes automatically recompiled for
#      dependent projects.
# Either way, we need to create a CombBLASConfig.cmake.
# set(ConfigPackageLocation lib/cmake/CombBLAS)

# Generate version number header
# include(GenerateExportHeader)
# generate_export_header(CombBLAS)
# set_target_properties(CombBLAS PROPERTIES VERSION ${CombBLAS_VERSION})

# # installation
# install(DIRECTORY include/ DESTINATION include)
# install(DIRECTORY psort-1.0/include/ DESTINATION include)
# install(TARGETS CombBLAS EXPORT CombBLASTargets
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         RUNTIME DESTINATION bin
#         INCLUDES DESTINATION include
#         )

# include(CMakePackageConfigHelpers)

# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
#     VERSION ${CombBLAS_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )

# # The following commands allow for option 2.
# export(EXPORT CombBLASTargets
#        FILE "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASTargets.cmake"
#        NAMESPACE CombBLAS::
#        )
# configure_file(cmake/CombBLASConfig.cmake
#                "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake"
#                COPYONLY
#                )

# # Allow for option 2
# install(EXPORT CombBLASTargets
#         FILE
#         CombBLASTargets.cmake
#         NAMESPACE
#         CombBLAS::
#         DESTINATION
#         ${ConfigPackageLocation}
#         )
# export(PACKAGE CombBLAS)
# install(
#     FILES
#     cmake/CombBLASConfig.cmake
#     "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
#     DESTINATION
#     ${ConfigPackageLocation}
#     COMPONENT
#     Devel
# )

 # Testing
# enable_testing()
# include(CTest)
