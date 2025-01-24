# Read the version from the file
file(READ "${CMAKE_SOURCE_DIR}/VERSION" version_file_content)
string(STRIP "${version_file_content}" version_file_content)
string(REPLACE "\n" ";" version_file_lines "${version_file_content}")

# Parse major, minor, and revision numbers
list(GET version_file_lines 0 version_str)
string(REPLACE "." ";" version_numbers "${version_str}")
list(GET version_numbers 0 bb_ver_maj)
list(GET version_numbers 1 bb_ver_min)
list(GET version_numbers 2 bb_ver_rev)

# Parse the optional suffix
list(LENGTH version_file_lines version_file_lines_length)
if(${version_file_lines_length} GREATER 1)
    list(GET version_file_lines 1 bb_ver_suffix)
else()
    set(bb_ver_suffix "")
endif()

# Determine if we are in a CI environment
if(DEFINED ENV{CI})
    # CI build; use the suffix from the VERSION file
    if(bb_ver_suffix STREQUAL "")
        set(bb_ver_suffix_final "")
    else()
        set(bb_ver_suffix_final "-${bb_ver_suffix}")
    endif()
else()
    # Local build; use "-dev" as the suffix
    set(bb_ver_suffix_final "-dev")
endif()

# Get the Git commit hash
execute_process(COMMAND git rev-parse HEAD
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE bb_ver_commit
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set compile definitions
add_compile_definitions(BLADEBIT_VERSION_MAJ=${bb_ver_maj})
add_compile_definitions(BLADEBIT_VERSION_MIN=${bb_ver_min})
add_compile_definitions(BLADEBIT_VERSION_REV=${bb_ver_rev})
add_compile_definitions(BLADEBIT_VERSION_SUFFIX="${bb_ver_suffix_final}")
add_compile_definitions(BLADEBIT_GIT_COMMIT="${bb_ver_commit}")
