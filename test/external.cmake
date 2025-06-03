include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
endif()

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.10
)

FetchContent_MakeAvailable(catch2)


# FetchContent_Declare(
#   curl
#   GIT_REPOSITORY https://github.com/curl/curl.git
#   GIT_TAG        curl-8_14_0
# )

# FetchContent_GetProperties(curl)
# if(NOT curl_POPULATED)
#   FetchContent_Populate(curl)
#   option(CURL_DISABLE_ALTSVC "curl disable CURL_DISABLE_ALTSVC" ON)
#   option(CURL_DISABLE_AWS "curl disable CURL_DISABLE_ALTSVC" ON)
#   option(CURL_DISABLE_DOH "curl disable CURL_DISABLE_DOH" ON)
#   option(CURL_DISABLE_FTP "curl disable CURL_DISABLE_FTP" ON)
#   option(CURL_DISABLE_GOPHER "curl disable CURL_DISABLE_GOPHER" ON)
#   option(CURL_DISABLE_HSTS "curl disable CURL_DISABLE_HSTS" ON)
#   option(CURL_DISABLE_IMAP "curl disable CURL_DISABLE_IMAP" ON)
#   option(CURL_DISABLE_INSTALL "curl disable CURL_DISABLE_INSTALL" ON)
#   option(CURL_DISABLE_IPFS "curl disable CURL_DISABLE_IPFS" ON)
#   option(CURL_DISABLE_KERBEROS_AUTH "curl disable CURL_DISABLE_KERBEROS_AUTH" ON)
#   option(CURL_DISABLE_LDAP "curl disable CURL_DISABLE_LDAP" ON)
#   option(CURL_DISABLE_LDAPS "curl disable CURL_DISABLE_LDAPS" ON)
#   option(CURL_DISABLE_MQTT "curl disable CURL_DISABLE_MQTT" ON)
#   option(CURL_DISABLE_NEGOTIATE_AUTH "curl disable CURL_DISABLE_NEGOTIATE_AUTH" ON)
#   option(CURL_DISABLE_NTLM "curl disable CURL_DISABLE_NTLM" ON)
#   option(CURL_DISABLE_POP3 "curl disable CURL_DISABLE_POP3" ON)
#   option(CURL_DISABLE_SMB "curl disable CURL_DISABLE_SMB" ON)
#   option(CURL_DISABLE_TELNET "curl disable CURL_DISABLE_TELNET" ON)
#   option(CURL_DISABLE_TFTP "curl disable CURL_DISABLE_TFTP" ON)
#   option(CURL_DISABLE_WEBSOCKETS "curl disable CURL_DISABLE_WEBSOCKETS" ON)
#   option(CURL_USE_LIBPSL "curl disable CURL_USE_LIBPSL" OFF) 
#   option(BUILD_STATIC_CURL "curl build static lib" ON)
#   add_subdirectory(${curl_SOURCE_DIR} ${curl_BINARY_DIR})
# endif()
