# Find libnuma
#
# NUMA_FOUND       : True if libnuma was found
# NUMA_INCLUDE_DIR : Where to find numa.h & numaif.h
# NUMA_LIBRARY     : Library to link

include(FindPackageHandleStandardArgs)

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h numaif.h
  HINTS ${INCLUDE_INSTALL_DIR})

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${LIB_INSTALL_DIR})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA 
  REQUIRED_VARS NUMA_INCLUDE_DIR NUMA_LIBRARY)

mark_as_advanced(
  NUMA_INCLUDE_DIR
  NUMA_LIBRARY)