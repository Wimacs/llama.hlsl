# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1409/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1409/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wimaxs/Desktop/llama.hlsl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wimaxs/Desktop/llama.hlsl

# Include any dependencies generated for this target.
include CMakeFiles/VulkanExample.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/VulkanExample.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/VulkanExample.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VulkanExample.dir/flags.make

CMakeFiles/VulkanExample.dir/main.cpp.o: CMakeFiles/VulkanExample.dir/flags.make
CMakeFiles/VulkanExample.dir/main.cpp.o: main.cpp
CMakeFiles/VulkanExample.dir/main.cpp.o: CMakeFiles/VulkanExample.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wimaxs/Desktop/llama.hlsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VulkanExample.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/VulkanExample.dir/main.cpp.o -MF CMakeFiles/VulkanExample.dir/main.cpp.o.d -o CMakeFiles/VulkanExample.dir/main.cpp.o -c /home/wimaxs/Desktop/llama.hlsl/main.cpp

CMakeFiles/VulkanExample.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/VulkanExample.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wimaxs/Desktop/llama.hlsl/main.cpp > CMakeFiles/VulkanExample.dir/main.cpp.i

CMakeFiles/VulkanExample.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/VulkanExample.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wimaxs/Desktop/llama.hlsl/main.cpp -o CMakeFiles/VulkanExample.dir/main.cpp.s

# Object files for target VulkanExample
VulkanExample_OBJECTS = \
"CMakeFiles/VulkanExample.dir/main.cpp.o"

# External object files for target VulkanExample
VulkanExample_EXTERNAL_OBJECTS =

VulkanExample: CMakeFiles/VulkanExample.dir/main.cpp.o
VulkanExample: CMakeFiles/VulkanExample.dir/build.make
VulkanExample: /usr/lib/x86_64-linux-gnu/libvulkan.so
VulkanExample: CMakeFiles/VulkanExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/wimaxs/Desktop/llama.hlsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VulkanExample"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VulkanExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VulkanExample.dir/build: VulkanExample
.PHONY : CMakeFiles/VulkanExample.dir/build

CMakeFiles/VulkanExample.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VulkanExample.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VulkanExample.dir/clean

CMakeFiles/VulkanExample.dir/depend:
	cd /home/wimaxs/Desktop/llama.hlsl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl/CMakeFiles/VulkanExample.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/VulkanExample.dir/depend

