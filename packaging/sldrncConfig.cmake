# sldrncConfig.cmake -- find_package(sldrnc) for a prebuilt SLD-RNC package (the unzipped sldrnc-<version>-<platform>).
#
#   find_package(sldrnc 1.0 REQUIRED)          # with -DCMAKE_PREFIX_PATH=/path/to/sldrnc-<version>-<platform>
#   target_link_libraries(my_app PRIVATE sldrnc::sldrnc)
#
# Provides the imported target sldrnc::sldrnc (include path + shared library) and SLDRNC_LIBRARY_DIR, the directory of
# the runtime library: copy sldrnc.dll from there next to your executable on Windows, or use it as an rpath elsewhere.
# The C header needs C99; the C++ header needs C++17.
if(TARGET sldrnc::sldrnc)
  return()
endif()

get_filename_component(_sldrnc_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(sldrnc::sldrnc SHARED IMPORTED)
set_target_properties(sldrnc::sldrnc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_sldrnc_root}/include")

if(WIN32)
  set(SLDRNC_LIBRARY_DIR "${_sldrnc_root}/bin")
  set(_sldrnc_lib "${_sldrnc_root}/bin/sldrnc.dll")
  set_target_properties(sldrnc::sldrnc PROPERTIES IMPORTED_LOCATION "${_sldrnc_lib}" IMPORTED_IMPLIB "${_sldrnc_root}/lib/sldrnc.lib")
elseif(APPLE)
  set(SLDRNC_LIBRARY_DIR "${_sldrnc_root}/lib")
  set(_sldrnc_lib "${_sldrnc_root}/lib/libsldrnc.dylib")
  set_target_properties(sldrnc::sldrnc PROPERTIES IMPORTED_LOCATION "${_sldrnc_lib}" IMPORTED_SONAME "@rpath/libsldrnc.dylib")
else()
  set(SLDRNC_LIBRARY_DIR "${_sldrnc_root}/lib")
  set(_sldrnc_lib "${_sldrnc_root}/lib/libsldrnc.so")
  set_target_properties(sldrnc::sldrnc PROPERTIES IMPORTED_LOCATION "${_sldrnc_lib}" IMPORTED_SONAME "libsldrnc.so")
endif()

if(NOT EXISTS "${_sldrnc_lib}")
  set(sldrnc_FOUND FALSE)
  set(sldrnc_NOT_FOUND_MESSAGE "this SLD-RNC package has no library for this platform (expected ${_sldrnc_lib}); use the package built for your platform")
else()
  set(sldrnc_FOUND TRUE)
endif()
unset(_sldrnc_lib)
unset(_sldrnc_root)
