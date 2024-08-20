cmake_minimum_required(VERSION 3.10)
project(llama)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Create the executable
add_executable(llama main.cpp)

# Link Vulkan libraries
target_link_libraries(llama Vulkan::Vulkan)

# Set C++ standard
set_target_properties(llama PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
