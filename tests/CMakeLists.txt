include(FetchContent)

# ---------------------------
# GoogleTest via FetchContent
# ---------------------------
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)

FetchContent_MakeAvailable(googletest)

# ------------------------------------
# Helper to define a gpu_lab test
# ------------------------------------
function(add_gpu_lab_test name)
  add_executable(${name} ${ARGN})

  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/common/include
  )

  target_link_libraries(${name}
    PRIVATE
      gpu_lab_options
      GTest::gtest_main
  )

  add_test(
    NAME ${name}
    COMMAND ${name}
  )
endfunction()

# ------------------------------------
# Actual tests
# ------------------------------------

add_gpu_lab_test(test_unique_ptrs
  test_unique_ptrs.cpp
)

add_gpu_lab_test(test_buffer
  test_buffer.cpp
)
