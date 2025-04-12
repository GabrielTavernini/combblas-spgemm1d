# Include directories for library code.

# Find sources.
file(GLOB parmetis_sources *.c)
# Create libparmetis
add_library(parmetis ${ParMETIS_LIBRARY_TYPE} ${parmetis_sources})
# Link with metis and MPI libraries.
find_package(MPI REQUIRED)
target_link_libraries(parmetis PUBLIC metis MPI::MPI_CXX)
set_target_properties(parmetis PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
target_include_directories(parmetis PUBLIC ${CMAKE_CURRENT_LIST_DIR})
install(TARGETS parmetis
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib)
