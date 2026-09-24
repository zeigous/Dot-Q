cmake_minimum_required(VERSION 3.12)
project(main LANGUAGES CXX)

set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "CXX compiler" FORCE)

set(CMAKE_CXX_STANDARD 23)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -mtune=native -Wno-bit-int-extension)
    
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-flto=thin)
        add_link_options(-flto=thin -fuse-ld=lld)
    else()
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(EXECUTABLE_NAME "main")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(TBB REQUIRED)

add_executable(${EXECUTABLE_NAME} ${SOURCES})

target_link_libraries(main PRIVATE TBB::tbb)

target_include_directories(${EXECUTABLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_custom_target(symlink_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json"
)
