include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/avble/llama.cpp.git
    GIT_TAG        1955993e
)

FetchContent_Declare(
    av_connect
    GIT_REPOSITORY https://github.com/avble/av_connect.git
    GIT_TAG        dev-01
)

FetchContent_Declare(
  curl
  GIT_REPOSITORY https://github.com/curl/curl.git
  GIT_TAG        curl-8_14_0
)

FetchContent_GetProperties(llama_cpp)
if(NOT llama_cpp_POPULATED)
  FetchContent_Populate(llama_cpp)
  set(GGML_AMX OFF)
  set(GGML_BLAS OFF)
  option(LLAMA_CURL "llama libcur" OFF)
  option(LLAMA_BUILD_SERVER "llama server" OFF)
  option(LLAMA_BUILD_EXAMPLES "llama example" OFF)
  option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)
  option(LLAMA_BUILD_TESTS "llama: build llama tests" OFF)
  add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR})
endif()

FetchContent_GetProperties(av_connect)
if(NOT av_connect_POPULATED)
  FetchContent_Populate(av_connect)
  option(AV_CONNECT_BUILD_EXAMPLES "av_connect: Build examples" ON)
  add_subdirectory(${av_connect_SOURCE_DIR} ${av_connect_BINARY_DIR})
endif()

FetchContent_Declare(
  mbedtls
  GIT_REPOSITORY https://github.com/Mbed-TLS/mbedtls.git
  GIT_TAG v3.6.3
)

set(ENABLE_PROGRAMS OFF CACHE BOOL "Disable mbedTLS example programs")
set(ENABLE_TESTING OFF CACHE BOOL "Disable mbedTLS tests")
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "Avoid compiler warnings as errors")
set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "Build static mbedTLS")

FetchContent_MakeAvailable(mbedtls)
set(MBEDTLS_INCLUDE_DIRS "${mbedtls_SOURCE_DIR}/include" "${mbedtls_BINARY_DIR}")
set(MBEDTLS_INCLUDE_DIR "${mbedtls_SOURCE_DIR}/include" CACHE PATH "" FORCE)

set(MBEDTLS_LIBRARY "${mbedtls_BINARY_DIR}/library/Debug/mbedtls.lib" CACHE FILEPATH "" FORCE)
set(MBEDX509_LIBRARY "${mbedtls_BINARY_DIR}/library/Debug/mbedx509.lib" CACHE FILEPATH "" FORCE)
set(MBEDCRYPTO_LIBRARY "${mbedtls_BINARY_DIR}/library/Debug/mbedcrypto.lib" CACHE FILEPATH "" FORCE)

message("DEBUG " ${MBEDTLS_LIBRARY} )

FetchContent_GetProperties(curl)
if(NOT curl_POPULATED)
  FetchContent_Populate(curl)
  option(CURL_DISABLE_ALTSVC "curl disable CURL_DISABLE_ALTSVC" ON)
  option(CURL_DISABLE_AWS "curl disable CURL_DISABLE_ALTSVC" ON)
  option(CURL_DISABLE_DOH "curl disable CURL_DISABLE_DOH" ON)
  option(CURL_DISABLE_FTP "curl disable CURL_DISABLE_FTP" ON)
  option(CURL_DISABLE_GOPHER "curl disable CURL_DISABLE_GOPHER" ON)
  option(CURL_DISABLE_HSTS "curl disable CURL_DISABLE_HSTS" ON)
  option(CURL_DISABLE_IMAP "curl disable CURL_DISABLE_IMAP" ON)
  option(CURL_DISABLE_INSTALL "curl disable CURL_DISABLE_INSTALL" ON)
  option(CURL_DISABLE_IPFS "curl disable CURL_DISABLE_IPFS" ON)
  option(CURL_DISABLE_KERBEROS_AUTH "curl disable CURL_DISABLE_KERBEROS_AUTH" ON)
  option(CURL_DISABLE_LDAP "curl disable CURL_DISABLE_LDAP" ON)
  option(CURL_DISABLE_LDAPS "curl disable CURL_DISABLE_LDAPS" ON)
  option(CURL_DISABLE_MQTT "curl disable CURL_DISABLE_MQTT" ON)
  option(CURL_DISABLE_NEGOTIATE_AUTH "curl disable CURL_DISABLE_NEGOTIATE_AUTH" ON)
  option(CURL_DISABLE_NTLM "curl disable CURL_DISABLE_NTLM" ON)
  option(CURL_DISABLE_POP3 "curl disable CURL_DISABLE_POP3" ON)
  option(CURL_DISABLE_SMB "curl disable CURL_DISABLE_SMB" ON)
  option(CURL_DISABLE_TELNET "curl disable CURL_DISABLE_TELNET" ON)
  option(CURL_DISABLE_TFTP "curl disable CURL_DISABLE_TFTP" ON)
  option(CURL_DISABLE_WEBSOCKETS "curl disable CURL_DISABLE_WEBSOCKETS" ON)
  option(CURL_USE_LIBPSL "curl disable CURL_USE_LIBPSL" OFF) 
  # option(BUILD_STATIC_CURL "curl build static lib" ON)  
  set(CURL_USE_MBEDTLS TRUE CACHE BOOL "" FORCE)
  # option(CURL_USE_LIBSSH2 "curl CURL_USE_LIBSSH2" OFF)
  execute_process(COMMAND git submodule update --init --recursive
      WORKING_DIRECTORY ${curl_SOURCE_DIR}
  )
  add_subdirectory(${curl_SOURCE_DIR} ${curl_BINARY_DIR})
endif()

