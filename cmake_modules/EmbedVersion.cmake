
if((NOT DEFINED ENV{CI}) AND (NOT DEFINED CACHE{bb_version_embedded}))
    message("Embedding local build version")

    set(cmd_shell bash)
    set(cmd_ext sh)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

        find_program(bash_path NAMES bash.exe NO_CACHE)

        if(${bash_path} MATCHES "-NOTFOUND")
            set(cmd_shell powershell)
            set(cmd_ext ps1)
        else()
            set(cmd_shell "${bash_path}")
        endif()
    endif()

    execute_process(COMMAND ${cmd_shell} ${CMAKE_SOURCE_DIR}/extract-version.${cmd_ext} major    OUTPUT_VARIABLE bb_ver_maj    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_shell} ${CMAKE_SOURCE_DIR}/extract-version.${cmd_ext} minor    OUTPUT_VARIABLE bb_ver_min    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_shell} ${CMAKE_SOURCE_DIR}/extract-version.${cmd_ext} revision OUTPUT_VARIABLE bb_ver_rev    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_shell} ${CMAKE_SOURCE_DIR}/extract-version.${cmd_ext} suffix   OUTPUT_VARIABLE bb_ver_suffix WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${cmd_shell} ${CMAKE_SOURCE_DIR}/extract-version.${cmd_ext} commit   OUTPUT_VARIABLE bb_ver_commit WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY)

    # Remove trailing whitespace incurred in windows gitbash
    string(STRIP "${bb_ver_maj}"    bb_ver_maj)
    string(STRIP "${bb_ver_min}"    bb_ver_min)
    string(STRIP "${bb_ver_rev}"    bb_ver_rev)
    string(STRIP "${bb_ver_suffix}" bb_ver_suffix)
    string(STRIP "${bb_ver_commit}" bb_ver_commit)

    set(bb_ver_suffix ${bb_ver_suffix}-dev)

    # This is slow on windows, so let's cache them
    set(bb_ver_maj    ${bb_ver_maj}    CACHE STRING "")
    set(bb_ver_min    ${bb_ver_min}    CACHE STRING "")
    set(bb_ver_rev    ${bb_ver_rev}    CACHE STRING "")
    set(bb_ver_suffix ${bb_ver_suffix} CACHE STRING "")
    set(bb_ver_commit ${bb_ver_commit} CACHE STRING "")
endif()

if(NOT DEFINED ENV{CI})
    add_compile_definitions(BLADEBIT_VERSION_MAJ=${bb_ver_maj})
    add_compile_definitions(BLADEBIT_VERSION_MIN=${bb_ver_min})
    add_compile_definitions(BLADEBIT_VERSION_REV=${bb_ver_rev})
    add_compile_definitions(BLADEBIT_VERSION_SUFFIX="${bb_ver_suffix}")
    add_compile_definitions(BLADEBIT_GIT_COMMIT="${bb_ver_commit}")
endif()

set(bb_version_embedded on CACHE BOOL "Version embedding has already happened.")