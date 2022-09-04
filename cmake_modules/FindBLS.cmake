# Find libnuma
#
# NUMA_FOUND       : True if libnuma was found
# NUMA_INCLUDE_DIR : Where to find numa.h & numaif.h
# NUMA_LIBRARY     : Library to link

include(FindPackageHandleStandardArgs)

find_path(BLS_INCLUDE_DIR
  NAMES bls.hpp chiabls/bls.hpp
  HINTS ${INCLUDE_INSTALL_DIR})

find_library(BLS_LIBRARY
  NAMES libbls.a
  HINTS ${LIB_INSTALL_DIR})

if(BLS_INCLUDE_DIR)
    message("Found bls includes: ${BLS_INCLUDE_DIR}")
endif()

if(BLS_LIBRARY)
    message("Found bls library: ${BLS_LIBRARY}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(bls
  REQUIRED_VARS BLS_INCLUDE_DIR BLS_LIBRARY)

mark_as_advanced(
  BLS_INCLUDE_DIR
  BLS_LIBRARY)