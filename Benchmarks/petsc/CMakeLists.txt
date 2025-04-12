find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
message(STATUS ${PETSc_INCLUDE_DIRS} ${PETSc_FOUND})
set(BIN 
spgemm-petsc)

foreach(bin ${BIN})
    add_executable(${bin} ${CMAKE_CURRENT_SOURCE_DIR}/petsc/${bin}.c)
    target_include_directories(${bin} PRIVATE ${PETSc_INCLUDE_DIRS})
    target_link_libraries(${bin} PRIVATE ${PETSc_LINK_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX) 
endforeach(bin ${BIN})
