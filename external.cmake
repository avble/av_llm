include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
	if (POLICY CMP0169)
		cmake_policy(SET CMP0169 OLD)
	endif()
endif()

FetchContent_Declare(
	llama_cpp
	GIT_REPOSITORY https://github.com/avble/llama.cpp.git
	GIT_TAG        main
)

FetchContent_GetProperties(llama_cpp)
if(NOT llama_cpp_POPULATED)
	FetchContent_Populate(llama_cpp)
	option(LLAMA_CURL "llama libcur" OFF)
	option(LLAMA_BUILD_SERVER "llama server" ON)
	option(LLAMA_BUILD_EXAMPLES "llama example" OFF)
	option(LLAMA_BUILD_TOOLS    "llama: build tools" ON)
	option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)
	option(LLAMA_BUILD_TESTS "llama: build llama tests" OFF)
	add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR})
endif()


FetchContent_Declare(
	mbedtls
	GIT_REPOSITORY https://github.com/Mbed-TLS/mbedtls.git
	GIT_TAG v3.6.3
)
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "Avoid compiler warnings as errors")
# set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "Build static mbedTLS")

FetchContent_GetProperties(mbedtls)
if (NOT mbedtls_POPULATED)
	message("the mbedtls has not populated " ${mbedtls_SOURCE_DIR})
	FetchContent_Populate(mbedtls)
	execute_process(COMMAND git submodule update --init --recursive
		WORKING_DIRECTORY ${mbedtls_SOURCE_DIR}
		RESULT_VARIABLE mbedtls_cmd_result
		OUTPUT_VARIABLE mbedtls_cmd_ouput
		ERROR_VARIABLE mbedtls_cmd_error
	)
	if (NOT mbedtls_cmd_result EQUAL 0)
		message(FATAL_ERROR "Comand error with ${mbedtls_cmd_result} msg: ${mbedtls_cmd_error}")
	else()
		message(STATUS " mbedtls-git success: " ${mbedtls_cmd_result})
	endif()

	execute_process(
		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/cmake/patches/mbedtls_CMakeLists.txt ${mbedtls_SOURCE_DIR}/CMakeLists.txt
		WORKING_DIRECTORY ${mbedtls_SOURCE_DIR}
		RESULT_VARIABLE mbedtls_cmd_result
		OUTPUT_VARIABLE mbedtls_cmd_ouput
		ERROR_VARIABLE mbedtls_cmd_error
	)

	if (NOT mbedtls_cmd_result EQUAL 0)
		message(FATAL_ERROR "Comand error with ${mbedtls_cmd_result} msg: ${mbedtls_cmd_error}")
	else()
		message(STATUS " mbedtls-copy success: " ${mbedtls_cmd_result})
	endif()

	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "Build static mbedTLS" FORCE)
	add_subdirectory(${mbedtls_SOURCE_DIR} ${mbedtls_BINARY_DIR})

	# Determine build type (for Ninja and single-config generators)
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	endif()

	# Detect platform and library extensions
	if (WIN32)
		set(LIB_EXT ".lib")
	else()
		set(LIB_EXT ".a")
	endif()

	set(LIB_DIR "${mbedtls_BINARY_DIR}/library")
	set(MBEDTLS_LIBRARY     "${LIB_DIR}/mbedtls${LIB_EXT}"     CACHE FILEPATH "mbedtls library" FORCE)
	set(MBEDCRYPTO_LIBRARY  "${LIB_DIR}/mbedcrypto${LIB_EXT}"  CACHE FILEPATH "mbedcrypto library" FORCE)
	set(MBEDX509_LIBRARY    "${LIB_DIR}/mbedx509${LIB_EXT}"    CACHE FILEPATH "mbedx509 library" FORCE)
	set(MBEDTLS_INCLUDE_DIRS "${mbedtls_SOURCE_DIR}/include" "${mbedtls_BINARY_DIR}")
	set(MBEDTLS_INCLUDE_DIR  "${mbedtls_SOURCE_DIR}/include" CACHE PATH "mbedtls include" FORCE)
	set(MBEDTLS_LIBRARY_DIRS "${LIB_DIR}")
	set(MBEDTLS_LIBRARIES mbedtls mbedx509 mbedcrypto)
	set(MBEDTLS_FOUND TRUE CACHE BOOL "MBEDTLS is found" FORCE)


endif()
FetchContent_Declare(
	curl
	GIT_REPOSITORY https://github.com/curl/curl.git
	GIT_TAG        curl-8_14_0
)
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


FetchContent_Declare(
	av_connect
	GIT_REPOSITORY https://github.com/avble/av_connect.git
	GIT_TAG        dev-01
)

FetchContent_GetProperties(av_connect)
if(NOT av_connect_POPULATED)
	FetchContent_Populate(av_connect)
	option(AV_CONNECT_BUILD_EXAMPLES "av_connect: Build examples" OFF)
	add_subdirectory(${av_connect_SOURCE_DIR} ${av_connect_BINARY_DIR})
endif()


include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

