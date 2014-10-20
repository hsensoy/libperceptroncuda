
cmake_minimum_required(VERSION 2.8.4)

SET( PROJ_NAME      "perceptroncuda" )
SET( PROJ_PATH      ${CMAKE_SOURCE_DIR} )
SET( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR} )
SET( PROJ_SOURCES   "src/perceptroncuda.c" "src/kernelperceptroncuda.c" "src/simpleperceptroncuda.c" "src/epblas/epblas_cuda.c" "src/epblas/epblas_common.c" "src/debug.c" "src/eputil.c")
SET( PROJ_HEADERS   "perceptron.h" "kernelperceptron.h"  "simpleperceptron.h" "perceptron_common.h")
SET( PROJ_LIBRARIES "-lcurand" )

SET( PROJ_INCLUDES  "include" )

#SET( CMAKE_SHARED_LINKER_FLAGS "-O3 -L/Users/husnusensoy/NetBeansProjects/ai-parse/epblas/build -lepblasmkl" CACHE STRING "" FORCE )


SET (${PROJ_NAME}_VERSION_MAJOR 0)
SET (${PROJ_NAME}_VERSION_MINOR 1)


PROJECT(${PROJ_NAME})

FIND_PACKAGE(CUDA REQUIRED)
SET( CUDA_NVCC_FLAGS "-O3 -arch=sm_20" CACHE STRING "nvcc flags" FORCE )
SET( CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE )


INCLUDE_DIRECTORIES( ${PROJ_INCLUDES} )

CUDA_ADD_LIBRARY( ${PROJ_NAME} SHARED ${PROJ_SOURCES} )

CUDA_ADD_CUBLAS_TO_TARGET( ${PROJ_NAME} )

#CUDA_ADD_CURAND_TO_TARGET( ${PROJ_NAME} )
#LINK_DIRECTORIES( "/Users/husnusensoy/NetBeansProjects/ai-parse/epblas/build" )

#TARGET_LINK_LIBRARIES( ${PROJ_NAME} ${PROJ_LIBRARIES} )


enable_testing()


SET ( PROJ_TESTS
"t1"
"t2"
"tMatrixVectorProductDimMismatch"
"tMatrixVectorProductwithTranspose"
"tRectangularMatrixMatrixProduct"
"tRectangularMatrixMatrixProductTranspose"
"tRectangularMatrixMatrixProductTransposeWithSizingError"
"tSquareMatrixMatrixProduct"
"tdotprod"
"tmatrixvectorprod" )

foreach( t ${PROJ_TESTS} )
    add_executable( test_${PROJ_NAME}_${t} tests/${t}.c)
    target_link_libraries( test_${PROJ_NAME}_${t} "cunit" "perceptroncuda")
    add_test( test_${PROJ_NAME}_${t}  ./test_${PROJ_NAME}_${t} )
endforeach(t)


set (CMAKE_INSTALL_PREFIX /Users/husnusensoy/ai-lab)
install(TARGETS ${PROJ_NAME} DESTINATION lib)
install(FILES include/perceptron_common.h include/perceptron.h DESTINATION include)
