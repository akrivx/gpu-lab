#include <cstddef>
// #include <vector>
// #include <type_traits>

#include <gtest/gtest.h>

#include "image_view.hpp"
#include "memory_location.hpp"

using namespace gpu_lab;

// ----------------------------------- Compile-time checks -----------------------------------

static_assert(HostImageView<int>::accessor_type::location() == MemoryLocation::Host);
static_assert(HostPinnedImageView<int>::accessor_type::location() == MemoryLocation::HostPinned);
static_assert(DeviceImageView<int>::accessor_type::location() == MemoryLocation::Device);


// ------------ ImageView tests ------------

struct TestRawImage {
  static constexpr std::size_t N = 8;
  static constexpr std::size_t STRIDE = 8;
  static constexpr std::size_t STRIDE_BYTES = STRIDE * sizeof(int);

  int data[N][N];

  TestRawImage() {
    for (std::size_t y = 0; y < N; ++y) {
      for (std::size_t x = 0; x < N; ++x) {
        data[y][x] = y * STRIDE + x;
      }
    }
  }

  auto view_5x4() {
    return image_view<MemoryLocation::Host>(&data[0][0], 4, 5, STRIDE_BYTES);
  }
};

TEST(ImageView, BasicPropertiesAndIndexing)
{
  TestRawImage raw;

  auto img_5x4 = raw.view_5x4();

  EXPECT_EQ(img_5x4.extent(0), 5);
  EXPECT_EQ(img_5x4.extent(1), 4);
  EXPECT_EQ(img_5x4.stride(0), TestRawImage::STRIDE_BYTES);
  EXPECT_EQ(img_5x4.stride(1), sizeof(int));

  for (std::size_t y = 0; y < img_5x4.extent(0); ++y) {
    for (std::size_t x = 0; x < img_5x4.extent(1); ++x) {
      const auto value = raw.data[y][x];
      EXPECT_EQ(img_5x4(y, x), value);
      img_5x4(y, x) *= 10;
      EXPECT_EQ(raw.data[y][x], 10 * value);
    }
  }
}

TEST(ImageView, Subcols)
{
  TestRawImage raw;

  auto img_5x4 = raw.view_5x4();
  
  auto left_5x2 = subcols(img_5x4, {0, 2});
  for (std::size_t y = 0; y < left_5x2.extent(0); ++y) {
    for (std::size_t x = 0; x < left_5x2.extent(1); ++x) {
      EXPECT_EQ(left_5x2(y, x), img_5x4(y, x));
    }
  }

  auto right_5x2 = subcols(img_5x4, {2, 4});
  for (std::size_t y = 0; y < right_5x2.extent(0); ++y) {
    for (std::size_t x = 0; x < right_5x2.extent(1); ++x) {
      EXPECT_EQ(right_5x2(y, x), img_5x4(y, x + 2));
    }
  }

  auto mid_5x2 = subcols(img_5x4, {1, 3});
  for (std::size_t y = 0; y < mid_5x2.extent(0); ++y) {
    for (std::size_t x = 0; x < mid_5x2.extent(1); ++x) {
      EXPECT_EQ(mid_5x2(y, x), img_5x4(y, x + 1));
    }
  }
}

TEST(ImageView, Subrows)
{
  TestRawImage raw;

  auto img_5x4 = raw.view_5x4();
  
  auto top_2x4 = subrows(img_5x4, {0, 2});
  for (std::size_t y = 0; y < top_2x4.extent(0); ++y) {
    for (std::size_t x = 0; x < top_2x4.extent(1); ++x) {
      EXPECT_EQ(top_2x4(y, x), img_5x4(y, x));
    }
  }

  auto bot_2x4 = subrows(img_5x4, {3, 5});
  for (std::size_t y = 0; y < bot_2x4.extent(0); ++y) {
    for (std::size_t x = 0; x < bot_2x4.extent(1); ++x) {
      EXPECT_EQ(bot_2x4(y, x), img_5x4(y + 3, x));
    }
  }

  auto mid_3x4 = subrows(img_5x4, {1, 4});
  for (std::size_t y = 0; y < mid_3x4.extent(0); ++y) {
    for (std::size_t x = 0; x < mid_3x4.extent(1); ++x) {
      EXPECT_EQ(mid_3x4(y, x), img_5x4(y + 1, x));
    }
  }
}

TEST(ImageView, Subviews)
{
  TestRawImage raw;

  auto img_5x4 = raw.view_5x4();
  
  // Top-left 3x3
  auto tl_3x3 = subview(img_5x4, {0, 3}, {0, 3});
  for (std::size_t y = 0; y < tl_3x3.extent(0); ++y) {
    for (std::size_t x = 0; x < tl_3x3.extent(1); ++x) {
      EXPECT_EQ(tl_3x3(y, x), img_5x4(y, x));
    }
  }

  // Top-right 3x3
  auto tr_3x3 = subview(img_5x4, {0, 3}, {1, 4});
  for (std::size_t y = 0; y < tr_3x3.extent(0); ++y) {
    for (std::size_t x = 0; x < tr_3x3.extent(1); ++x) {
      EXPECT_EQ(tr_3x3(y, x), img_5x4(y, x + 1));
    }
  }

  // Bottom-left 3x3
  auto bl_3x3 = subview(img_5x4, {2, 5}, {0, 3});
  for (std::size_t y = 0; y < bl_3x3.extent(0); ++y) {
    for (std::size_t x = 0; x < bl_3x3.extent(1); ++x) {
      EXPECT_EQ(bl_3x3(y, x), img_5x4(y + 2, x));
    }
  }

  // Bottom-right 3x3
  auto br_3x3 = subview(img_5x4, {2, 5}, {1, 4});
  for (std::size_t y = 0; y < br_3x3.extent(0); ++y) {
    for (std::size_t x = 0; x < br_3x3.extent(1); ++x) {
      EXPECT_EQ(br_3x3(y, x), img_5x4(y + 2, x + 1));
    }
  }

  // Mid 3x2
  auto mid_4x3 = subview(img_5x4, {1, 4}, {1, 3});
  for (std::size_t y = 0; y < mid_4x3.extent(0); ++y) {
    for (std::size_t x = 0; x < mid_4x3.extent(1); ++x) {
      EXPECT_EQ(mid_4x3(y, x), img_5x4(y + 1, x + 1));
    }
  }
}

TEST(ImageView, AsConst)
{
  TestRawImage raw;

  auto img_5x4 = raw.view_5x4();
  auto const_img_5x4 = as_const(img_5x4);

  static_assert(std::is_same_v<
    decltype(const_img_5x4),
    HostImageView<const int>
  >);

  EXPECT_EQ(img_5x4.data_handle(), const_img_5x4.data_handle());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(img_5x4.extent(i), const_img_5x4.extent(i));
    EXPECT_EQ(img_5x4.stride(i), const_img_5x4.stride(i));
  }
}
