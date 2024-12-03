cmake_minimum_required(VERSION 3.20)
project(llama.hlsl)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(llama.hlsl
    src/main.cpp
    src/d3dx12.h
)
add_definitions(-DUNICODE -D_UNICODE)

target_link_libraries(llama.hlsl PRIVATE dxcore dxgi d3d12 dxcompiler)

add_custom_command(TARGET llama.hlsl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/external/dxil.dll"
    $<TARGET_FILE_DIR:llama.hlsl>)