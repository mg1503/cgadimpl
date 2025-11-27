# Dialect headers will go here
add_subdirectory(Dialect)
add_subdirectory(Translation)

# Install all header files
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Compiler"
    FILES_MATCHING 
    PATTERN "*.h"
    PATTERN "*.td"
    PATTERN "CMakeLists.txt" EXCLUDE
)
add_subdirectory(Transforms)
add_subdirectory(Pipeline)
