// Stat the exact ordered paths supplied by PairedImageDataset in bounded batches.
#ifndef __clang__
#error LLVM Clang is required.
#endif

#include <sys/stat.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
void write_u64(std::uint8_t*& destination, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        *destination++ = static_cast<std::uint8_t>(value >> shift);
    }
}
}

// Each path ends with NUL. Output is seven little-endian uint64 fields per
// path, in input order. Positive results are errno, -1 means non-regular,
// and -2 means malformed input. An error returns the failing byte offset.
extern "C" int studio_paired_stat_v1(
    const char* paths, std::size_t size, std::size_t count,
    std::uint8_t* records, std::size_t* failed
) noexcept {
    const auto* next = paths;
    const auto* end = paths + size;
    for (std::size_t index = 0; index < count; ++index) {
        *failed = static_cast<std::size_t>(next - paths);
        const auto* terminator = static_cast<const char*>(
            std::memchr(next, '\0', static_cast<std::size_t>(end - next))
        );
        if (!terminator || terminator == next) {
            return -2;
        }

        struct stat info{};
        if (stat(next, &info) != 0) {
            return errno;
        }
        if (!S_ISREG(info.st_mode)) {
            return -1;
        }
        write_u64(records, info.st_dev);
        write_u64(records, info.st_ino);
        write_u64(records, info.st_size);
        write_u64(records, info.st_mtim.tv_sec);
        write_u64(records, info.st_mtim.tv_nsec);
        write_u64(records, info.st_ctim.tv_sec);
        write_u64(records, info.st_ctim.tv_nsec);
        next = terminator + 1;
    }
    return next == end ? 0 : -2;
}
