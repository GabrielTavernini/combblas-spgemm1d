set(CTFROOT /pscratch/sd/y/yuxihong/graphclustering/ctf)
find_package(MKL REQUIRED)
set(BIN 
algebra_multigrid
matmul-benchmark)

foreach(bin ${BIN})
    add_executable(${bin} ${CMAKE_CURRENT_SOURCE_DIR}/ctf/${bin}.cxx)
    # message(STATUS ${bin})
    target_include_directories(${bin} PRIVATE ${CTFROOT}/include)
    target_link_libraries(${bin} PRIVATE ${CTFROOT}/build/lib/libctf.a) 
    target_link_libraries(${bin} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX MKL::MKL) 
endforeach(bin ${BIN})

