# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build

# Include any dependencies generated for this target.
include CMakeFiles/animals.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/animals.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/animals.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/animals.dir/flags.make

CMakeFiles/animals.dir/Animal.cpp.o: CMakeFiles/animals.dir/flags.make
CMakeFiles/animals.dir/Animal.cpp.o: ../Animal.cpp
CMakeFiles/animals.dir/Animal.cpp.o: CMakeFiles/animals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/animals.dir/Animal.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/animals.dir/Animal.cpp.o -MF CMakeFiles/animals.dir/Animal.cpp.o.d -o CMakeFiles/animals.dir/Animal.cpp.o -c /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Animal.cpp

CMakeFiles/animals.dir/Animal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/animals.dir/Animal.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Animal.cpp > CMakeFiles/animals.dir/Animal.cpp.i

CMakeFiles/animals.dir/Animal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/animals.dir/Animal.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Animal.cpp -o CMakeFiles/animals.dir/Animal.cpp.s

CMakeFiles/animals.dir/Cat.cpp.o: CMakeFiles/animals.dir/flags.make
CMakeFiles/animals.dir/Cat.cpp.o: ../Cat.cpp
CMakeFiles/animals.dir/Cat.cpp.o: CMakeFiles/animals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/animals.dir/Cat.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/animals.dir/Cat.cpp.o -MF CMakeFiles/animals.dir/Cat.cpp.o.d -o CMakeFiles/animals.dir/Cat.cpp.o -c /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Cat.cpp

CMakeFiles/animals.dir/Cat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/animals.dir/Cat.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Cat.cpp > CMakeFiles/animals.dir/Cat.cpp.i

CMakeFiles/animals.dir/Cat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/animals.dir/Cat.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Cat.cpp -o CMakeFiles/animals.dir/Cat.cpp.s

CMakeFiles/animals.dir/Dog.cpp.o: CMakeFiles/animals.dir/flags.make
CMakeFiles/animals.dir/Dog.cpp.o: ../Dog.cpp
CMakeFiles/animals.dir/Dog.cpp.o: CMakeFiles/animals.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/animals.dir/Dog.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/animals.dir/Dog.cpp.o -MF CMakeFiles/animals.dir/Dog.cpp.o.d -o CMakeFiles/animals.dir/Dog.cpp.o -c /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Dog.cpp

CMakeFiles/animals.dir/Dog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/animals.dir/Dog.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Dog.cpp > CMakeFiles/animals.dir/Dog.cpp.i

CMakeFiles/animals.dir/Dog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/animals.dir/Dog.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/Dog.cpp -o CMakeFiles/animals.dir/Dog.cpp.s

# Object files for target animals
animals_OBJECTS = \
"CMakeFiles/animals.dir/Animal.cpp.o" \
"CMakeFiles/animals.dir/Cat.cpp.o" \
"CMakeFiles/animals.dir/Dog.cpp.o"

# External object files for target animals
animals_EXTERNAL_OBJECTS =

libanimals.dylib: CMakeFiles/animals.dir/Animal.cpp.o
libanimals.dylib: CMakeFiles/animals.dir/Cat.cpp.o
libanimals.dylib: CMakeFiles/animals.dir/Dog.cpp.o
libanimals.dylib: CMakeFiles/animals.dir/build.make
libanimals.dylib: CMakeFiles/animals.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libanimals.dylib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/animals.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/animals.dir/build: libanimals.dylib
.PHONY : CMakeFiles/animals.dir/build

CMakeFiles/animals.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/animals.dir/cmake_clean.cmake
.PHONY : CMakeFiles/animals.dir/clean

CMakeFiles/animals.dir/depend:
	cd /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build /Users/wenshuiluo/coding/CPP/CMakeTest/cmake-cookbook/chapter-01/recipe-09/cxx-example/build/CMakeFiles/animals.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/animals.dir/depend

