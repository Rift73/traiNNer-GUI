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

add_executable(traiNNer-GUI WIN32 launcher.cpp windows_host.cpp)
target_compile_features(traiNNer-GUI PRIVATE cxx_std_17)
target_compile_definitions(traiNNer-GUI PRIVATE UNICODE _UNICODE
  TRAINNER_GUI_PAYLOAD_SHA256="${TRAINNER_GUI_PAYLOAD_SHA256}")
target_link_libraries(traiNNer-GUI PRIVATE advapi32 bcrypt comctl32 ole32 oleaut32 shell32 uuid wbemuuid)
if(MSVC)
  set_property(TARGET traiNNer-GUI PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
  target_compile_options(traiNNer-GUI PRIVATE /W4 /utf-8)
  target_link_options(traiNNer-GUI PRIVATE
    "/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'")
endif()
