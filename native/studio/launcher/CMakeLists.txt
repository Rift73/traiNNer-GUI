cmake_minimum_required(VERSION 3.21)
project(trainner_gui_launcher LANGUAGES CXX)
if(NOT WIN32)
  message(FATAL_ERROR "On Linux use ./trainner-gui directly. The desktop launcher uses Windows and WSL.")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "Configure with LLVM clang-cl for Windows.")
endif()

set(TRAINNER_GUI_PAYLOAD_SHA256 "" CACHE STRING "SHA-256 of the accompanying source archive")
string(LENGTH "${TRAINNER_GUI_PAYLOAD_SHA256}" payload_hash_length)
if(TRAINNER_GUI_PAYLOAD_SHA256 AND
   (NOT payload_hash_length EQUAL 64 OR TRAINNER_GUI_PAYLOAD_SHA256 MATCHES "[^0-9a-f]"))
  message(FATAL_ERROR "Provide the lowercase SHA-256 of the accompanying source archive.")
endif()

set(TRAINNER_GUI_RELEASE_TAG "" CACHE STRING "GitHub release tag for the packaged Windows launcher")
set(release_number "(0|[1-9][0-9]*)")
if(TRAINNER_GUI_RELEASE_TAG)
  if(NOT TRAINNER_GUI_RELEASE_TAG MATCHES "^v${release_number}\\.${release_number}\\.${release_number}(-preview\\.${release_number})?$")
    message(FATAL_ERROR "Use a GitHub release tag of the form vX.Y.Z or vX.Y.Z-preview.N.")
  endif()
  string(REGEX MATCHALL "[0-9]+" release_numbers "${TRAINNER_GUI_RELEASE_TAG}")
  foreach(number IN LISTS release_numbers)
    if(number GREATER 4294967295)
      message(FATAL_ERROR "Each release tag component must fit in an unsigned 32-bit integer.")
    endif()
  endforeach()
endif()

add_executable(traiNNer-GUI WIN32 launcher.cpp updates.cpp windows_host.cpp)
target_compile_features(traiNNer-GUI PRIVATE cxx_std_20)
target_compile_definitions(traiNNer-GUI PRIVATE UNICODE _UNICODE
  TRAINNER_GUI_PAYLOAD_SHA256="${TRAINNER_GUI_PAYLOAD_SHA256}"
  TRAINNER_GUI_RELEASE_TAG="${TRAINNER_GUI_RELEASE_TAG}")
target_link_libraries(traiNNer-GUI PRIVATE advapi32 bcrypt comctl32 ole32 oleaut32
  runtimeobject shell32 uuid wbemuuid winhttp wintrust)
if(MSVC)
  set_property(TARGET traiNNer-GUI PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
  target_compile_options(traiNNer-GUI PRIVATE /W4 /utf-8)
  target_link_options(traiNNer-GUI PRIVATE
    "/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'")
endif()
