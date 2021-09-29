# Find libnuma
#
# NUMA_FOUND       : True if libnuma was found
# NUMA_INCLUDE_DIR : Where to find numa.h & numaif.h
# NUMA_LIBRARY     : Library to link

include(FindPackageHandleStandardArgs)

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h numaif.h
  HINTS ${NUMA_ROOT_DIR}/include)

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${NUMA_ROOT_DIR}/lib)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA 
  REQUIRED_VARS NUMA_INCLUDE_DIR NUMA_LIBRARY)

mark_as_advanced(
  NUMA_INCLUDE_DIR
  NUMA_LIBRARY)