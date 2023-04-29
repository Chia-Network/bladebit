add_executable(tests ${src_bladebit})
target_compile_definitions(tests PRIVATE
    BB_TEST_MODE=1
)
target_link_libraries(tests PRIVATE bladebit_config Catch2::Catch2WithMain)

