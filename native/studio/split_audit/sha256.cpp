// Hash only the selected file's encoded bytes, using the installed GLib SHA-256.
#ifndef __clang__
#error LLVM Clang is required.
#endif

#include <glib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <unistd.h>

namespace {
ssize_t read_retry(int fd, void* data, std::size_t size) {
    ssize_t count;
    do {
        count = read(fd, data, size);
    } while (count < 0 && errno == EINTR);
    return count;
}
}

// fd is positioned at the start. Return 1 for a size change, 2 for
// cancellation, and -errno for a read error. The caller checks file identity.
extern "C" int studio_split_sha256_v1(
    int fd, std::uint64_t size, unsigned char* digest, int (*cancelled)()
) noexcept {
    std::unique_ptr<GChecksum, decltype(&g_checksum_free)> sha(
        g_checksum_new(G_CHECKSUM_SHA256), g_checksum_free
    );
    if (!sha) {
        return -ENOMEM;
    }
    std::array<unsigned char, 1 << 20> bytes;
    while (size != 0) {
        if (cancelled && cancelled()) {
            return 2;
        }
        const auto got = read_retry(fd, bytes.data(), std::min<std::uint64_t>(size, bytes.size()));
        if (got < 0) {
            return -errno;
        }
        if (got == 0) {
            return 1;
        }
        g_checksum_update(sha.get(), bytes.data(), got);
        size -= static_cast<std::uint64_t>(got);
    }
    const auto extra = read_retry(fd, bytes.data(), 1);
    if (extra < 0) {
        return -errno;
    }
    if (extra != 0) {
        return 1;
    }
    gsize length = 32;
    g_checksum_get_digest(sha.get(), digest, &length);
    return 0;
}
