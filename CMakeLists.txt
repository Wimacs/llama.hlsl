cmake_minimum_required(VERSION 3.12)
project(llm.hlsl)

# 设置C++版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(UNICODE _UNICODE)

set(IMGUI_SRC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/backends/imgui_impl_dx12.cpp
    third_party/imgui/backends/imgui_impl_win32.cpp
)

add_library(ImGui STATIC ${IMGUI_SRC})
add_subdirectory(DXSample)
target_include_directories(ImGui PUBLIC third_party/imgui third_party/imgui/backends)


file(GLOB SOURCES src/*.cpp src/*.h)

add_executable(llm.hlsl ${SOURCES})
set_target_properties(llm.hlsl PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS")

target_link_libraries(llm.hlsl PRIVATE d3d12 dxgi dxguid D3DCompiler DXSample ImGui)
