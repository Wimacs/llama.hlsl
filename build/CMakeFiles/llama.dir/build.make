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
CMAKE_BINARY_DIR = /home/wimaxs/Desktop/llama.hlsl/build

# Include any dependencies generated for this target.
include CMakeFiles/llama.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/llama.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/llama.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/llama.dir/flags.make

CMakeFiles/llama.dir/main.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/main.cpp.o: /home/wimaxs/Desktop/llama.hlsl/main.cpp
CMakeFiles/llama.dir/main.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wimaxs/Desktop/llama.hlsl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/llama.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/main.cpp.o -MF CMakeFiles/llama.dir/main.cpp.o.d -o CMakeFiles/llama.dir/main.cpp.o -c /home/wimaxs/Desktop/llama.hlsl/main.cpp

CMakeFiles/llama.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/llama.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wimaxs/Desktop/llama.hlsl/main.cpp > CMakeFiles/llama.dir/main.cpp.i

CMakeFiles/llama.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/llama.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wimaxs/Desktop/llama.hlsl/main.cpp -o CMakeFiles/llama.dir/main.cpp.s

# Object files for target llama
llama_OBJECTS = \
"CMakeFiles/llama.dir/main.cpp.o"

# External object files for target llama
llama_EXTERNAL_OBJECTS =

llama: CMakeFiles/llama.dir/main.cpp.o
llama: CMakeFiles/llama.dir/build.make
llama: /usr/lib/x86_64-linux-gnu/libvulkan.so
llama: CMakeFiles/llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/wimaxs/Desktop/llama.hlsl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable llama"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/llama.dir/build: llama
.PHONY : CMakeFiles/llama.dir/build

CMakeFiles/llama.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/llama.dir/cmake_clean.cmake
.PHONY : CMakeFiles/llama.dir/clean

CMakeFiles/llama.dir/depend:
	cd /home/wimaxs/Desktop/llama.hlsl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl /home/wimaxs/Desktop/llama.hlsl/build /home/wimaxs/Desktop/llama.hlsl/build /home/wimaxs/Desktop/llama.hlsl/build/CMakeFiles/llama.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/llama.dir/depend

