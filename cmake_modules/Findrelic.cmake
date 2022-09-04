# Find libnuma
#
# NUMA_FOUND       : True if libnuma was found
# NUMA_INCLUDE_DIR : Where to find numa.h & numaif.h
# NUMA_LIBRARY     : Library to link

include(FindPackageHandleStandardArgs)

find_path(RELIC_INCLUDE_DIR
  NAMES relic_conf.h relic/relic_conf.h
  HINTS ${INCLUDE_INSTALL_DIR})

find_library(RELIC_LIBRARY
  NAMES librelic_s.a
  HINTS ${LIB_INSTALL_DIR})

set(RELIC_INCLUDE_DIR ${RELIC_INCLUDE_DIR}/relic)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RELIC
  REQUIRED_VARS RELIC_INCLUDE_DIR RELIC_LIBRARY)

mark_as_advanced(
  RELIC_INCLUDE_DIR
  RELIC_LIBRARY)