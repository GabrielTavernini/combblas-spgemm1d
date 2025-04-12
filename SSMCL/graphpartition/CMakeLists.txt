SET(TFOLDER ${CMAKE_CURRENT_SOURCE_DIR}/graphpartition)
SET(BINARYS
parmetisGP
metisGP
    # gpanalyzer
    mtx2gpinput
    symmgraph
    mtx2graph
    checkvectoreq
    checkperm2d
    getinfo2d
    spgemm2d
    gp2combblas
)

SET(CPPFILES
parmetisGP
metisGP
    # gpanalyzer
    mtx2gpinput
    symmgraph
    mtx2graph
    checkvectoreq
    checkperm2d
    getinfo2d
    spgemm2d
    gp2combblas
)
find_package(ZLIB REQUIRED)
foreach(bin cppf IN ZIP_LISTS BINARYS CPPFILES)
    ADD_EXECUTABLE( ${bin} ${TFOLDER}/${cppf}.cpp)
    TARGET_LINK_LIBRARIES( ${bin} PUBLIC CombBLAS ${ZLIB_LIBRARIES})
    target_include_directories(${bin} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${bin} PUBLIC ${CMAKE_SOURCE_DIR}/Applications)
endforeach()
