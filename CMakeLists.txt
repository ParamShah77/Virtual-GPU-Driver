cmake_minimum_required(VERSION 3.16)
project(vgpu C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# -DSANITIZE=address,undefined  or  -DSANITIZE=thread
set(SANITIZE "" CACHE STRING "Sanitizers to enable")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
if(SANITIZE)
  add_compile_options(-fsanitize=${SANITIZE} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${SANITIZE})
endif()

find_package(Threads REQUIRED)

# OS-agnostic resource-manager core (also compiled into the kernel module).
add_library(vrm_core STATIC
  core/vrm_mem.c core/vrm_exec.c core/vrm_misc.c)
target_include_directories(vrm_core PUBLIC include core)

# User-mode runtime: kernel backend + in-process simulator backend.
add_library(vrm STATIC user/vrm.cpp user/sim.cpp)
target_include_directories(vrm PUBLIC include core user)
target_link_libraries(vrm PUBLIC vrm_core Threads::Threads)

add_executable(vgpu_tests tests/vgpu_tests.cpp)
target_link_libraries(vgpu_tests PRIVATE vrm)

add_executable(vgpu_trace tools/vgpu_trace.cpp)
target_link_libraries(vgpu_trace PRIVATE vrm)

add_executable(vgpu_bench tools/vgpu_bench.cpp)
target_link_libraries(vgpu_bench PRIVATE vrm)

enable_testing()
add_test(NAME sim COMMAND vgpu_tests --backend sim)
