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
