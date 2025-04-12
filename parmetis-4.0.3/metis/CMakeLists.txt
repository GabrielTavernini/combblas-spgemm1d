cmake_minimum_required(VERSION 3.20)
project(METIS LANGUAGES C)

add_subdirectory(GKlib)

# set(GKLIB_PATH "GKlib" CACHE PATH "path to GKlib")
# set(SHARED FALSE CACHE BOOL "build a shared library")

if(MSVC)
  set(METIS_INSTALL FALSE)
else()
  set(METIS_INSTALL TRUE)
endif()

# Configure libmetis library.
if(SHARED)
  set(METIS_LIBRARY_TYPE SHARED)
else()
  set(METIS_LIBRARY_TYPE STATIC)
endif(SHARED)

# include(${GKLIB_PATH}/GKlibSystem.cmake)
# Add include directories.
# include_directories(${GKLIB_PATH})

# Recursively look for CMakeLists.txt in subdirs.
add_subdirectory("include")
add_subdirectory("libmetis")
target_include_directories(metis PUBLIC include)
add_subdirectory("programs")
