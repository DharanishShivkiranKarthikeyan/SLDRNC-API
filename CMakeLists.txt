cmake_minimum_required(VERSION 3.16)
project(sldrnc VERSION 1.0.0 LANGUAGES CXX)

# SLD-RNC (Self-correcting Learned Dynamics for Real-time Neural Control) shared library:
# Windows (MSVC / clang-cl), Linux (GCC / Clang), macOS (AppleClang), x86-64 and ARM64.
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
#   ctest --test-dir build -C Release

option(SLDRNC_BUILD_TESTS "Build the test program" ON)
option(SLDRNC_BUILD_EXAMPLES "Build the C++ example" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)          # export only the C interface
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC
  src/capi.cpp src/correction.cpp src/features.cpp src/kernels_generic.cpp src/model_io.cpp
  src/onnx_import.cpp src/runtime.cpp src/session.cpp src/train.cpp src/fleet.cpp src/structure.cpp)

set(SLDRNC_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
  set(SLDRNC_X86 ON)
  list(APPEND SRC src/kernels_avx2.cpp src/kernels_table_avx2.cpp)
endif()

add_library(sldrnc SHARED ${SRC})
target_include_directories(sldrnc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_definitions(sldrnc PRIVATE SLDRNC_BUILD)
set_target_properties(sldrnc PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

find_package(Threads REQUIRED)
target_link_libraries(sldrnc PRIVATE Threads::Threads)

if(MSVC)
  # carry the C++ runtime inside the DLL (no Visual C++ Redistributable needed; only C types cross the boundary)
  option(SLDRNC_STATIC_RUNTIME "Link the C++ runtime into sldrnc.dll" ON)
  if(SLDRNC_STATIC_RUNTIME)
    set_property(TARGET sldrnc PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  target_compile_options(sldrnc PRIVATE /O2 /W3 /EHsc /permissive-)
  set_source_files_properties(src/kernels_generic.cpp PROPERTIES COMPILE_OPTIONS "/fp:contract")
  if(SLDRNC_X86)
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:contract")
    # tables: the vector and scalar lookups must round identically, so no contraction there
    set_source_files_properties(src/kernels_table_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
else()
  target_compile_options(sldrnc PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
  set_source_files_properties(src/kernels_generic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast")
  if(SLDRNC_X86)
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=fast")
    set_source_files_properties(src/kernels_table_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
  endif()
  set_source_files_properties(src/structure.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  if(APPLE)
    target_link_options(sldrnc PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/exports_macos.txt")
  else()
    # export only the C interface; carry the C++ runtime inside (only C types cross the boundary)
    target_link_options(sldrnc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map" -static-libstdc++ -static-libgcc)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      target_link_options(sldrnc PRIVATE -s)      # strip symbols from release builds
    endif()
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS sldrnc EXPORT sldrncTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/sldrnc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT sldrncTargets NAMESPACE sldrnc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sldrnc)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/sldrncConfig.cmake "include(\${CMAKE_CURRENT_LIST_DIR}/sldrncTargets.cmake)\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sldrncConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sldrnc)

if(SLDRNC_BUILD_TESTS)
  enable_testing()
  add_executable(sldrnc_test tests/test_sldrnc.cpp)
  target_link_libraries(sldrnc_test PRIVATE sldrnc)
  add_test(NAME sldrnc_test COMMAND sldrnc_test)
endif()
if(SLDRNC_BUILD_EXAMPLES)
  add_executable(sldrnc_example examples/cpp/quickstart.cpp)
  target_link_libraries(sldrnc_example PRIVATE sldrnc)
  add_executable(sldrnc_latency benchmarks/latency.cpp)
  target_link_libraries(sldrnc_latency PRIVATE sldrnc)
endif()
