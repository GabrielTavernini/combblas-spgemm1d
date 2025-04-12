SET(TFOLDER ${CMAKE_CURRENT_SOURCE_DIR}/spgemm1d)
SET(BINARYS
## bench
benchspgemm1d

## test
testfmm
testmpioneside
testmpiwindow
testspgemm1d
printhardware
printdataset
printdataset2d
## App
comminfo
testeveryoneprint
mtxequal
)

SET(CPPFILES
## bench
benchspgemm1d

## test
testfmm
testmpioneside
testmpiwindow
testspgemm1d
printhardware
printdataset
printdataset2d
## App
comminfo
testeveryoneprint
mtxequal
)

foreach(bin cppf IN ZIP_LISTS BINARYS CPPFILES)
    ADD_EXECUTABLE( ${bin} ${TFOLDER}/${cppf}.cpp)
    TARGET_LINK_LIBRARIES( ${bin} PUBLIC CombBLAS)
    target_include_directories(${bin} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(${bin} PUBLIC ${CMAKE_SOURCE_DIR}/Applications)
endforeach()



# ADD_TEST(NAME VirusCscTest COMMAND $<TARGET_FILE:csctestvirus> ../vir_vs_vir_30_50length_propermm.mtx)