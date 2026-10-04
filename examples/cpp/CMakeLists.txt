# Build the C++ quickstart against a prebuilt SLD-RNC package.  From the unzipped package's root:
#
#   cmake -S examples/cpp -B build-example
#   cmake --build build-example --config Release
#   build-example/quickstart          (Windows: build-example\Release\quickstart.exe)
#
# The package two levels up is found automatically; elsewhere pass -DCMAKE_PREFIX_PATH=/path/to/the/package.
cmake_minimum_required(VERSION 3.16)
project(sldrnc_quickstart CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")
find_package(sldrnc 1.0 REQUIRED)

add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE sldrnc::sldrnc)

if(WIN32)
  # the DLL must sit next to the executable (or on PATH)
  add_custom_command(TARGET quickstart POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SLDRNC_LIBRARY_DIR}/sldrnc.dll" "$<TARGET_FILE_DIR:quickstart>")
else()
  # find the shared library where it is, without installing it
  set_target_properties(quickstart PROPERTIES BUILD_RPATH "${SLDRNC_LIBRARY_DIR}")
endif()
