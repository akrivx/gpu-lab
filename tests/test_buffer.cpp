#include <cstdint>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>

#include "buffer.hpp"
#include "memory_location.hpp"

using namespace gpu_lab;

// ----------------------------------- Compile-time checks -----------------------------------

static_assert(HostPageableBufferView<int>::MEMORY_LOCATION == MemoryLocation::HOST_PAGEABLE);
static_assert(HostPinnedBufferView<int>::MEMORY_LOCATION == MemoryLocation::HOST_PINNED);
static_assert(DeviceBufferView<int>::MEMORY_LOCATION == MemoryLocation::DEVICE);

static_assert(std::is_same_v<HostPageableBufferView<int>::element_type, int>);
static_assert(std::is_same_v<HostPageableBufferView<const int>::element_type, const int>);
static_assert(std::is_same_v<HostPageableBufferView<int>::value_type, int>);
static_assert(std::is_same_v<HostPageableBufferView<const int>::value_type, int>);

static_assert(std::is_same_v<HostPinnedBufferView<int>::element_type, int>);
static_assert(std::is_same_v<HostPinnedBufferView<const int>::element_type, const int>);
static_assert(std::is_same_v<HostPinnedBufferView<int>::value_type, int>);
static_assert(std::is_same_v<HostPinnedBufferView<const int>::value_type, int>);

static_assert(std::is_same_v<DeviceBufferView<int>::element_type, int>);
static_assert(std::is_same_v<DeviceBufferView<const int>::element_type, const int>);
static_assert(std::is_same_v<DeviceBufferView<int>::value_type, int>);
static_assert(std::is_same_v<DeviceBufferView<const int>::value_type, int>);

// ------------ BufferView tests ------------

TEST(BufferView, BasicPropertiesAndIndexing)
{
  int data[4] = {1, 2, 3, 4};
  HostPageableBufferView v{data, 4};

  EXPECT_EQ(v.data(), data);
  EXPECT_EQ(v.size(), 4u);
  EXPECT_FALSE(v.empty());

  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
  EXPECT_EQ(v[3], 4);

  v[2] = 42;
  EXPECT_EQ(data[2], 42);
}

TEST(BufferView, EmptyView)
{
  HostPageableBufferView<int> v{nullptr, 0};
  EXPECT_EQ(v.data(), nullptr);
  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(BufferView, SubspanTakeDrop)
{
  int data[5] = {10, 20, 30, 40, 50};
  HostPageableBufferView v{data, 5};

  auto mid = v.subspan(1, 3);
  EXPECT_EQ(mid.size(), 3u);
  EXPECT_EQ(mid.data(), &data[1]);
  EXPECT_EQ(mid[0], 20);
  EXPECT_EQ(mid[1], 30);
  EXPECT_EQ(mid[2], 40);

  auto first_two = v.take(2);
  EXPECT_EQ(first_two.size(), 2u);
  EXPECT_EQ(first_two.data(), &data[0]);
  EXPECT_EQ(first_two[0], 10);
  EXPECT_EQ(first_two[1], 20);

  auto last_three = v.drop(2);
  EXPECT_EQ(last_three.size(), 3u);
  EXPECT_EQ(last_three.data(), &data[2]);
  EXPECT_EQ(last_three[0], 30);
  EXPECT_EQ(last_three[1], 40);
  EXPECT_EQ(last_three[2], 50);
}

TEST(BufferView, AsConst)
{
  int data[3] = {7, 8, 9};
  HostPageableBufferView v{data, 3};

  auto cv = v.as_const();

  static_assert(std::is_same_v<
    decltype(cv),
    HostPageableBufferView<const int>
  >);

  EXPECT_EQ(cv.size(), 3u);
  EXPECT_EQ(cv.data(), data);
  EXPECT_EQ(cv[0], 7);
}

TEST(BufferView, RebindViewToSmallerElementType)
{
  uint32_t data[4] = {0x11223344u, 0, 0, 0};
  HostPageableBufferView v{data, 4};

  // Rebind to bytes: 4 * 4 = 16 bytes
  auto bytes = v.as<std::uint8_t>();

  static_assert(std::is_same_v<
    decltype(bytes),
    HostPageableBufferView<std::uint8_t>
  >);

  EXPECT_EQ(bytes.size(), 16u);
  EXPECT_EQ(bytes.data(), reinterpret_cast<std::uint8_t*>(data));
  EXPECT_EQ(bytes[0], 0x44);
  EXPECT_EQ(bytes[1], 0x33);
  EXPECT_EQ(bytes[2], 0x22);
  EXPECT_EQ(bytes[3], 0x11);
  for (size_t i = 4; i < bytes.size(); ++i) {
    EXPECT_EQ(bytes[i], 0);
  }
}

// ---------- Buffer (host-pageable) basic tests ----------

TEST(Buffer, DefaultConstructedIsEmpty)
{
  HostPageableBuffer<int> buf;
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.empty());

  auto v = buf.view();
  EXPECT_EQ(v.data(), nullptr);
  EXPECT_EQ(v.size(), 0u);
}

TEST(Buffer, SizedConstructAllocates)
{
  constexpr size_t N = 16;
  HostPageableBuffer<int> buf{N};

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), N);
  EXPECT_FALSE(buf.empty());

  auto v = buf.view();
  EXPECT_EQ(v.data(), buf.data());
  EXPECT_EQ(v.size(), buf.size());
}

TEST(Buffer, MoveConstructorTransfersOwnership)
{
  constexpr size_t N = 8;

  HostPageableBuffer<int> a{N};
  int* a_ptr = a.data();
  for (size_t i = 0; i < N; ++i) {
    a.data()[i] = static_cast<int>(i);
  }

  HostPageableBuffer<int> b{std::move(a)};
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), 0u);

  EXPECT_EQ(b.data(), a_ptr);
  EXPECT_EQ(b.size(), N);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(b.data()[i], static_cast<int>(i));
  }
}

TEST(Buffer, MoveAssignmentTransfersOwnership)
{
  constexpr size_t N = 8;
  constexpr size_t M = 4;

  HostPageableBuffer<int> a{N};
  for (size_t i = 0; i < N; ++i) {
    a.data()[i] = static_cast<int>(i);
  }

  HostPageableBuffer<int> b{M};
  EXPECT_NE(b.data(), nullptr);

  b = std::move(a);

  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), 0u);

  EXPECT_EQ(b.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(b.data()[i], static_cast<int>(i));
  }
}

TEST(Buffer, ReleaseResetsBufferAndReturnsHandle)
{
  constexpr size_t N = 10;
  HostPageableBuffer<int> buf{N};
  int* raw_before = buf.data();

  auto handle = buf.release();

  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.empty());

  EXPECT_NE(handle.get(), nullptr);
  EXPECT_EQ(handle.get(), raw_before);
}

// ---------- Buffer view/cview/view_as tests ----------

TEST(Buffer, ViewAndCViewTypes)
{
  HostPageableBuffer<int> buf{4};

  auto v = buf.view();
  static_assert(std::is_same_v<
    decltype(v),
    HostPageableBufferView<int>
  >);

  const auto& cbuf = buf;
  auto cv = cbuf.view();
  static_assert(std::is_same_v<
    decltype(cv),
    HostPageableBufferView<const int>
  >);

  auto cv2 = buf.cview();
  static_assert(std::is_same_v<
    decltype(cv2),
    HostPageableBufferView<const int>
  >);
}

TEST(Buffer, ViewAsRebindsElementType)
{
  HostPageableBuffer<std::uint32_t> buf{4};
  auto bv = buf.view_as<std::uint8_t>();

  static_assert(std::is_same_v<
    decltype(bv),
    HostPageableBufferView<std::uint8_t>
  >);

  EXPECT_EQ(bv.size(), buf.size() * sizeof(std::uint32_t));
  EXPECT_EQ(bv.data(), reinterpret_cast<std::uint8_t*>(buf.data()));
}

// ---------- Copy & clone tests ----------

TEST(BufferCopy, HostToHostCopy)
{
  constexpr size_t N = 32;
  HostPageableBuffer<int> src{N};
  HostPageableBuffer<int> dst{N};

  for (size_t i = 0; i < N; ++i) {
    src.data()[i] = static_cast<int>(i * 2);
    dst.data()[i] = -1;
  }

  copy(src.view(), dst.view());

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(dst.data()[i], src.data()[i]);
  }
}

TEST(BufferCopy, SizeMismatchThrows)
{
  constexpr size_t N = 16;
  constexpr size_t M = 8;

  HostPageableBuffer<int> small{M};
  HostPageableBuffer<int> big{N};

  EXPECT_THROW(copy(small.cview(), big.view()), std::runtime_error);
  EXPECT_THROW(copy(big.cview(), small.view()), std::runtime_error);
}

TEST(BufferCopy, HostToDeviceAndBack)
{
  constexpr size_t N = 64;

  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(i + 100);
  }

  // Host -> Device
  auto d_buf = to_device_buffer(h_src.view());

  // Device -> Host
  auto h_dst = to_host_pageable_buffer(d_buf.view());

  ASSERT_EQ(h_dst.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}

TEST(BufferCopyAsync, HostToDeviceAndBackDefaultStream)
{
  constexpr size_t N = 64;

  HostPinnedBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(42 + i);
  }

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto d_buf = to_device_buffer_async(h_src.view(), stream);
  auto h_dst = to_host_pinned_buffer_async(d_buf.view(), stream);

  // ensure async copies are finished
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(h_dst.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}
