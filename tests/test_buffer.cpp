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

TEST(BufferView, Copy)
{
  const int src_data[5] = {10, 20, 30, 40, 50};
  int dst_data[5] = {-1, -1, -1, -1, -1};

  HostPageableBufferView src{src_data, 5};
  HostPageableBufferView dst{dst_data, 5};

  copy(src, dst);

  for (size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }
}

// ---------- Buffer (host-pageable) basic tests ----------

TEST(Buffer, DefaultConstructedIsEmpty)
{
  HostPageableBuffer<int> buf;
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.empty());

  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0u);
}

TEST(Buffer, SizedConstructAllocates)
{
  constexpr size_t N = 16;
  HostPageableBuffer<int> buf{N};

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), N);
  EXPECT_FALSE(buf.empty());

  EXPECT_EQ(buf.data(), buf.data());
  EXPECT_EQ(buf.size(), buf.size());
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

TEST(BufferCopy, SizeMismatchThrows)
{
  constexpr size_t N = 16;
  constexpr size_t M = 8;

  HostPageableBuffer<int> small{M};
  HostPageableBuffer<int> big{N};

  EXPECT_THROW(copy(small, big), std::runtime_error);
  EXPECT_THROW(copy(big, small), std::runtime_error);
}

TEST(BufferCopy, HostToHostCopy)
{
  constexpr size_t N = 32;
  HostPageableBuffer<int> src{N};
  HostPageableBuffer<int> dst{N};

  for (size_t i = 0; i < N; ++i) {
    src.data()[i] = static_cast<int>(i * 2);
    dst.data()[i] = -1;
  }

  copy(src, dst);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(dst.data()[i], src.data()[i]);
  }
}

TEST(BufferCopy, HostToDeviceAndBack)
{
  constexpr size_t N = 64;

  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(i + 100);
  }

  // Host -> Device
  auto d_buf = to_device_buffer(h_src);

  // Device -> Host
  auto h_dst = to_host_pageable_buffer(d_buf);

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

  auto d_buf = to_device_buffer_async(h_src, stream);
  auto h_dst = to_host_pinned_buffer_async(d_buf, stream);

  // ensure async copies are finished
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(h_dst.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}

TEST(BufferCopy, DeviceToHostCopy)
{
  constexpr size_t N = 64;

  // Initialise host source
  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(i * 5);
  }

  // Copy to device first
  auto d_src = to_device_buffer(h_src);

  // Prepare host destination
  HostPageableBuffer<int> h_dst{N};
  for (size_t i = 0; i < N; ++i) {
    h_dst.data()[i] = -1;
  }

  // Now the thing we actually care about: device -> host
  copy(d_src, h_dst);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}

TEST(BufferCopy, DeviceToDeviceCopy)
{
  constexpr size_t N = 128;

  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(100 + i);
  }

  // Fill d_src from host
  auto d_src = to_device_buffer(h_src);

  // d_dst initially different
  DeviceBuffer<int> d_dst{N};
  {
    HostPageableBuffer<int> h_tmp{N};
    for (size_t i = 0; i < N; ++i) {
      h_tmp.data()[i] = -1;
    }
    copy(h_tmp, d_dst);
  }

  // Device -> Device
  copy(d_src, d_dst);

  // Copy d_dst back and compare
  auto h_dst = to_host_pageable_buffer(d_dst);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}

TEST(BufferCopyAsync, DeviceToHost)
{
  constexpr size_t N = 64;

  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(i + 1000);
  }

  HostPageableBuffer<int> h_dst{N};
  for (size_t i = 0; i < N; ++i) {
    h_dst.data()[i] = -1;
  }

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // Host -> Device
  auto d_buf = to_device_buffer_async(h_src, stream);
  // Device -> Host
  copy_async(d_buf, h_dst, stream);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}

TEST(BufferCopyAsync, DeviceToDevice)
{
  constexpr size_t N = 128;

  HostPageableBuffer<int> h_src{N};
  for (size_t i = 0; i < N; ++i) {
    h_src.data()[i] = static_cast<int>(42 + i);
  }

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // Host -> d_src
  auto d_src = to_device_buffer_async(h_src, stream);

  // d_src -> d_dst
  auto d_dst = to_device_buffer_async(d_src, stream);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  // Bring d_dst back to host and compare
  auto h_dst = to_host_pageable_buffer(d_dst);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(h_dst.data()[i], h_src.data()[i]);
  }
}
