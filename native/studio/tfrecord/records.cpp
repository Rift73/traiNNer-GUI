// Verify complete TensorBoard TFRecord frames in a bounded memory buffer.
// The caller decodes Event messages with the installed protobuf implementation.
#ifndef __clang__
#error "Build the native training tools with LLVM Clang."
#endif
#include <array>
#include <cstddef>
#include <cstdint>

namespace {
constexpr auto crc_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < table.size(); ++i) {
        auto value = i;
        for (unsigned bit = 0; bit < 8; ++bit)
            value = (value >> 1) ^ ((value & 1) ? 0x82f63b78u : 0u);
        table[i] = value;
    }
    return table;
}
constexpr auto table = crc_table();
std::uint32_t masked_crc(const unsigned char* p, std::size_t size) {
    std::uint32_t crc = ~0u;
    for (std::size_t i = 0; i < size; ++i)
        crc = table[(crc ^ p[i]) & 255u] ^ (crc >> 8);
    crc = ~crc;
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
}
template<typename T> T little(const unsigned char* p) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
        value |= static_cast<T>(p[i]) << (8 * i);
    return value;
}
}

// Each pair of adjacent offsets delimits a complete, CRC-checked record,
// including its 12-byte length header and its final 4-byte data checksum.
// The last verified offset also identifies the failing or incomplete record.
extern "C" int index_records(const unsigned char* p, std::size_t size,
                             std::size_t* offsets, std::size_t capacity,
                             std::size_t* count, std::uint64_t* needed) {
    if (!p || !offsets || !capacity || !count || !needed) return 3;
    *needed = 12;
    *count = 0;
    offsets[0] = 0;
    std::size_t pos = 0;
    while (size - pos >= 12 && *count < capacity - 1) {
        const auto* record = p + pos;
        if (masked_crc(record, 8) != little<std::uint32_t>(record + 8)) return 1;
        const auto length = little<std::uint64_t>(record);
        if (length > UINT64_MAX - 16) return 3;
        *needed = length + 16;
        if (*needed > size - pos) break;
        const auto n = static_cast<std::size_t>(length);
        if (masked_crc(record + 12, n) != little<std::uint32_t>(record + 12 + n)) return 2;
        pos += n + 16;
        offsets[++*count] = pos;
        *needed = 12;
    }
    return 0;
}
