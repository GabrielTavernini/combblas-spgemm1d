cmake_minimum_required(VERSION 3.20)
project(GKlib LANGUAGES C)

get_filename_component(abs "." ABSOLUTE)
message(STATUS ${abs})
set(GKLIB_PATH ${abs})

include(GKlibSystem.cmake)

# include_directories(".")
add_library(GKlib STATIC ${GKlib_sources})
message(STATUS ${abs})
target_include_directories(GKlib PUBLIC ${abs})
unset(abs)
if(UNIX)
  target_link_libraries(GKlib PUBLIC m)
endif(UNIX)

# include_directories("test")
# add_subdirectory("test")

install(TARGETS GKlib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(FILES ${GKlib_includes} DESTINATION include)
