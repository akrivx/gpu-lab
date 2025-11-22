#include <stdexcept>

#include <gtest/gtest.h>

#include "pinned_ptr.hpp"
#include "device_ptr.hpp"

static constexpr int N = 1000;

TEST(PinnedPtr, Allocate)
{
  using namespace gpu_lab;

  UniquePinnedPtr<int[]> ptr;
  EXPECT_EQ(ptr, nullptr);

  ptr = make_unique_pinned_ptr<int[]>(N);

  EXPECT_NE(ptr, nullptr);
}

TEST(PinnedPtr, Initialise)
{
  using namespace gpu_lab;

  auto ptr = make_unique_pinned_ptr<int[]>(N);
  for (int i = 0; i < N; ++i) {
    ptr[i] = i;
  }
  
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(ptr[i], i);
  }
}

TEST(DevicePtr, Allocate)
{
  using namespace gpu_lab;

  UniqueDevicePtr<int[]> ptr;
  EXPECT_EQ(ptr, nullptr);

  ptr = make_unique_device_ptr<int[]>(N);

  EXPECT_NE(ptr, nullptr);
}
