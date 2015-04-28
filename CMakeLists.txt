
cmake_minimum_required(VERSION 2.8.4)

SET( PROJ_NAME      "perceptroncuda" )
SET( PROJ_PATH      ${CMAKE_SOURCE_DIR} )
SET( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR} )
FILE( GLOB_RECURSE PROJ_SOURCES "include/*.h" "src/*.c" "src/*.cu") 

#SET( PROJ_HEADERS   "perceptron.h" "kernelperceptron.h"  "simpleperceptron.h" "perceptron_common.h" "epcudakernel.h")
SET( PROJ_LIBRARIES "m" )

SET( PROJ_INCLUDES  "include" "include/epblas" "/usr/local/include/shared" )

SET( CMAKE_SHARED_LINKER_FLAGS " -O3 /usr/local/lib/shared/libshared.a" CACHE STRING "" FORCE )
SET( CMAKE_C_FLAGS "-std=gnu99 -Wall -O3 -DNDEBUG  -g")


SET (${PROJ_NAME}_VERSION_MAJOR 0)
SET (${PROJ_NAME}_VERSION_MINOR 1)


PROJECT(${PROJ_NAME})

FIND_PACKAGE(CUDA REQUIRED)
SET( CUDA_NVCC_FLAGS "-O3 -lcurand   -DNDEBUG " CACHE STRING "nvcc flags" FORCE )
SET( CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE )


INCLUDE_DIRECTORIES( ${PROJ_INCLUDES} )

CUDA_ADD_LIBRARY( ${PROJ_NAME} SHARED ${PROJ_SOURCES} )

CUDA_ADD_CUBLAS_TO_TARGET( ${PROJ_NAME} )

#CUDA_ADD_CURAND_TO_TARGET( ${PROJ_NAME} )
#LINK_DIRECTORIES( "/Users/husnusensoy/NetBeansProjects/ai-parse/epblas/build" )

#TARGET_LINK_LIBRARIES( ${PROJ_NAME} ${PROJ_LIBRARIES} )


enable_testing()

set(DART_TESTING_TIMEOUT "10")


SET ( PROJ_TESTS
"t1"
"t2"
# "t2large"	# Long running
"t3"
"tMatrixVectorProductDimMismatch"
"tMatrixVectorProduct"
"tMatrixVectorProductwithTranspose"
"tRectangularMatrixMatrixProduct"
"tRectangularMatrixMatrixProductTranspose"
"tRectangularMatrixMatrixProductTransposeWithSizingError"
"tSquareMatrixMatrixProduct"
"tdotprod"
"tmatrixvectorprod"
"tSimple"
     "tSimplePerceptronwithRBFSampler"
     "tPerceptronUpdate"
     "tPerceptronUpdatewithRBFSampler"
     "tPerceptronBatchScorewithRBFSampler"
     "tPerceptronBatchScorewithRBFSamplerToyProblem"
    "tCosSinFast")

foreach( t ${PROJ_TESTS} )
    add_executable( test_${PROJ_NAME}_${t} tests/${t}.c)
    target_link_libraries( test_${PROJ_NAME}_${t} "perceptroncuda")
    target_link_libraries( test_${PROJ_NAME}_${t} "/usr/local/lib/shared/libshared.a")
    target_link_libraries( test_${PROJ_NAME}_${t} "/usr/local/cuda/lib64/libcurand.so")
    add_test( test_${PROJ_NAME}_${t}  ./test_${PROJ_NAME}_${t} )
endforeach(t)


#set (CMAKE_INSTALL_PREFIX /home/hsensoy/ai-lab)
install(TARGETS ${PROJ_NAME} DESTINATION lib)
install(FILES include/perceptron_common.h include/perceptron.h include/featuretransform.h DESTINATION include/perceptron)
install(FILES include/epblas/eputil.h include/epblas/epblas.h DESTINATION include/perceptron/epblas)
