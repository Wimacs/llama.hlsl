file(GLOB DXSample_SOURCES "*.cpp" "*.h")

add_library(DXSample STATIC ${DXSample_SOURCES})

target_include_directories(DXSample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})