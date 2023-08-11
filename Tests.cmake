include(cmake_modules/FindCatch2.cmake)

add_executable(tests ${src_bladebit}
    cuda/harvesting/CudaThresherDummy.cpp
    tests/TestUtil.h
    tests/TestDiskQueue.cpp
)

target_compile_definitions(tests PRIVATE
    BB_TEST_MODE=1
)
target_link_libraries(tests PRIVATE bladebit_config bladebit_core Catch2::Catch2WithMain)

set_target_properties(tests PROPERTIES 
    EXCLUDE_FROM_ALL ON
)