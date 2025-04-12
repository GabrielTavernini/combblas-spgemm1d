include_directories(.)
# Create programs.
find_package(OpenMP REQUIRED)
# if(TARGET OpenMP::OpenMP_C) # Use target if available (CMake >= 3.9)
#   target_compile_definitions(CombBLAS PUBLIC THREADED)
#   target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
# endif()

add_executable(ptest ptest.c io.c adaptgraph.c)
add_executable(mtest mtest.c io.c)
add_executable(parmetis_prog parmetis.c io.c adaptgraph.c)
set_target_properties(parmetis_prog PROPERTIES OUTPUT_NAME parmetis)
add_executable(pometis pometis.c io.c)
# Link to libparmetis.
foreach(prog ptest mtest parmetis_prog pometis)
  target_link_libraries(${prog} parmetis OpenMP::OpenMP_C)
endforeach(prog)

install(TARGETS ptest mtest parmetis_prog pometis
  RUNTIME DESTINATION bin)
