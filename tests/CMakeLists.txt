include(FetchContent)

# Fetch GoogleTest (header + source) at configure time
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2    # or any recent stable
)

# This adds gtest, gtest_main, etc.
FetchContent_MakeAvailable(googletest)

# -----------------------------------------
# test_unique_ptrs
# -----------------------------------------

add_executable(test_unique_ptrs
    test_unique_ptrs.cpp
)

target_include_directories(test_unique_ptrs
  PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
)

# Inherit warnings, C++20, CUDA flags, cudart, and common helpers
target_link_libraries(test_unique_ptrs
  PRIVATE
    # gpu_lab_common
    gpu_lab_options
    gtest_main
)

add_test(
    NAME unique_ptrs_tests
    COMMAND test_unique_ptrs
)

# -----------------------------------------
# test_buffer
# -----------------------------------------

add_executable(test_buffer
    test_buffer.cpp
)

target_include_directories(test_buffer
  PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
)

# Inherit warnings, C++20, CUDA flags, cudart, and common helpers
target_link_libraries(test_buffer
  PRIVATE
    # gpu_lab_common
    gpu_lab_options
    gtest_main
)

add_test(
    NAME buffer_tests
    COMMAND test_buffer
)
