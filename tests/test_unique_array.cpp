#include <stdexcept>

#include <gtest/gtest.h>

#include "unique_array.hpp"

static constexpr int N = 1000;

TEST(HostPinnedArray, Allocate)
{
  using namespace gpu_lab;

  UniqueHostPinnedArray<int> a;
  EXPECT_EQ(a, nullptr);

  a = make_unique_host_pinned_array<int>(N);

  EXPECT_NE(a, nullptr);
}

TEST(HostPinnedArray, Initialise)
{
  using namespace gpu_lab;

  auto a = make_unique_host_pinned_array<int>(N);
  for (int i = 0; i < N; ++i) {
    a[i] = i;
  }
  
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(a[i], i);
  }
}

TEST(DeviceArray, Allocate)
{
  using namespace gpu_lab;

  UniqueDeviceArray<int> a;
  EXPECT_EQ(a, nullptr);

  a = make_unique_device_array<int>(N);

  EXPECT_NE(a, nullptr);
}
