cmake_minimum_required(VERSION 3.12)
project(BLIP2)

# Add the ggml submodule to your project
add_subdirectory(ggml)

# Add your main.cpp to the project
add_executable(blip2 blip2.cpp)

# Add the sanitizer flag to your C++ compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

# Link your executable with ggml
target_link_libraries(blip2 ggml)
