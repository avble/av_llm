cmake_minimum_required(VERSION 3.12)

# Set the project name and version
project(MyApp VERSION 1.0.0 LANGUAGES CXX)

# Set the installation directories
set(CMAKE_INSTALL_PREFIX /usr/local)

# Specify the executable and its source files
add_executable(myapp src/main.cpp)

# Install the executable
install(TARGETS myapp DESTINATION bin)

# Optional: Install additional files (like configuration or documentation)
install(FILES README.md DESTINATION share/myapp)

# Specify the packaging details for CPack
set(CPACK_PACKAGE_NAME "myapp")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "A simple C++ application")
set(CPACK_PACKAGE_VENDOR "Your Company")
set(CPACK_PACKAGE_CONTACT "your.email@example.com")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_URL "http://example.com/myapp")
set(CPACK_PACKAGE_MAINTAINER "Your Name <your.email@example.com>")

# Specify CPack generator (in this case, Debian package)
if(WIN32)
    set(CPACK_GENERATOR "NSIS")  # Use NSIS for Windows (installer)
    set(CPACK_PACKAGE_FILE_NAME "myapp-${CPACK_PACKAGE_VERSION}-win64")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")  # Use DragNDrop for macOS (DMG)
    set(CPACK_PACKAGE_FILE_NAME "myapp-${CPACK_PACKAGE_VERSION}-macOS")
else()
    set(CPACK_GENERATOR "TGZ")  # Use TGZ for Linux (tarball)
    set(CPACK_PACKAGE_FILE_NAME "myapp-${CPACK_PACKAGE_VERSION}-linux")
endif()

# Specify additional post-installation steps (optional)
# set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICTNESS "experimental")

# Include CPack module
include(CPack)

