set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# Programs (cmake tools) run on build host, not ARM target
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Libraries and headers must come from ARM sysroot / SDK
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# Allow find_package to search both host and sysroot (needed for header-only libs like Eigen3)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
