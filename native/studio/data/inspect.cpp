#include <gdk-pixbuf/gdk-pixbuf.h>
#include <tiffio.h>
#include <webp/decode.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t sample_limit = 100;
constexpr std::size_t issue_limit = 200;

struct Dimensions {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    bool known() const { return width > 0 && height > 0; }
};

struct Image {
    fs::path path;
    std::string relative;
    Dimensions size{};
};

struct DirectoryScan {
    std::size_t count = 0;
    bool complete = true;
};

struct Issue {
    std::size_t root_id;
    fs::path path;
    std::string message;
};

struct Report {
    std::vector<Issue> errors;
    std::vector<Issue> warnings;
    std::size_t issue_count = 0;
    std::size_t errors_count = 0;
    std::size_t root_id = 1;

    void issue(bool error, const fs::path& path, std::string message) {
        ++issue_count;
        errors_count += error;
        auto& entries = error ? errors : warnings;
        if (entries.size() < issue_limit) {
            entries.push_back({root_id, path, std::move(message)});
        }
    }
};

std::string_view image_codec(const fs::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return std::tolower(character); });

    static const std::map<std::string, std::string_view> codecs = {
        {".apng", "png"}, {".avif", "unsupported"}, {".avifs", "unsupported"},
        {".bmp", "bmp"},  {".dds", "unsupported"}, {".dng", "unsupported"},
        {".exr", "unsupported"}, {".gif", "gif"}, {".hdr", "unsupported"},
        {".heic", "unsupported"}, {".heif", "unsupported"}, {".ico", "ico"},
        {".j2k", "unsupported"}, {".jfif", "jpeg"}, {".jp2", "unsupported"},
        {".jpe", "jpeg"}, {".jpeg", "jpeg"}, {".jpg", "jpeg"},
        {".jxl", "unsupported"}, {".jxr", "unsupported"}, {".pbm", "pnm"},
        {".pgm", "pnm"}, {".png", "png"}, {".pnm", "pnm"}, {".ppm", "pnm"},
        {".psd", "unsupported"}, {".qoi", "unsupported"}, {".tga", "unsupported"},
        {".tif", "tiff"}, {".tiff", "tiff"}, {".webp", "webp"},
    };
    const auto codec = codecs.find(extension);
    return codec == codecs.end() ? std::string_view{} : codec->second;
}

void record_pixbuf_size(GdkPixbufLoader* loader, int width, int height, gpointer data) {
    auto& size = *static_cast<Dimensions*>(data);
    if (!size.known() && width > 0 && height > 0) {
        size = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
    }
    // The loader can process pixel data in the same chunk as the header.
    gdk_pixbuf_loader_set_size(loader, 1, 1);
}

Dimensions pixbuf_size(std::FILE* input, std::string_view codec) {
    Dimensions size;
    const auto unref = [](GdkPixbufLoader* loader) { g_object_unref(loader); };
    std::unique_ptr<GdkPixbufLoader, decltype(unref)> loader(
        gdk_pixbuf_loader_new_with_type(codec.data(), nullptr), unref);
    if (!loader) {
        return {};
    }

    g_signal_connect(loader.get(), "size-prepared", G_CALLBACK(record_pixbuf_size), &size);
    std::array<unsigned char, 4096> buffer;
    constexpr int max_chunks = 4096;
    bool valid = true;
    for (int chunk = 0; chunk < max_chunks && !size.known(); ++chunk) {
        const auto bytes = std::fread(buffer.data(), 1, buffer.size(), input);
        if (bytes == 0) {
            break;
        }
        if (!gdk_pixbuf_loader_write(loader.get(), buffer.data(), bytes, nullptr)) {
            valid = false;
            break;
        }
    }
    gdk_pixbuf_loader_close(loader.get(), nullptr);
    return valid ? size : Dimensions{};
}

Dimensions webp_size(std::FILE* input) {
    std::array<unsigned char, 64> header;
    const auto bytes = std::fread(header.data(), 1, header.size(), input);
    WebPBitstreamFeatures features;
    if (bytes < 12 || std::memcmp(header.data(), "RIFF", 4) != 0 ||
        std::memcmp(header.data() + 8, "WEBP", 4) != 0 ||
        WebPGetFeatures(header.data(), bytes, &features) != VP8_STATUS_OK) {
        return {};
    }
    return {static_cast<std::uint32_t>(features.width),
            static_cast<std::uint32_t>(features.height)};
}

Dimensions tiff_size(int fd, const fs::path& path) {
    // TIFF owns the descriptor after a successful open.
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(
        TIFFFdOpen(fd, path.c_str(), "rm"), TIFFClose);
    if (!tiff) {
        close(fd);
        return {};
    }
    Dimensions size;
    if (!TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &size.width) ||
        !TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &size.height)) {
        return {};
    }
    return size;
}

std::string_view detected_codec(int fd, const fs::path& path) {
    std::array<unsigned char, 16> bytes;
    const auto read = pread(fd, bytes.data(), bytes.size(), 0);
    if (read >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
        std::memcmp(bytes.data() + 8, "WEBP", 4) == 0) {
        return "webp";
    }
    if (read >= 8 && std::memcmp(bytes.data(), "\x89PNG\r\n\x1a\n", 8) == 0) {
        return "png";
    }
    if (read >= 4 && (std::memcmp(bytes.data(), "II\x2a\0", 4) == 0 ||
                      std::memcmp(bytes.data(), "MM\0\x2a", 4) == 0 ||
                      std::memcmp(bytes.data(), "II\x2b\0", 4) == 0 ||
                      std::memcmp(bytes.data(), "MM\0\x2b", 4) == 0)) {
        return "tiff";
    }
    if (read >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff) {
        return "jpeg";
    }
    if (read >= 6 && (std::memcmp(bytes.data(), "GIF87a", 6) == 0 ||
                     std::memcmp(bytes.data(), "GIF89a", 6) == 0)) {
        return "gif";
    }
    if (read >= 2 && bytes[0] == 'B' && bytes[1] == 'M') {
        return "bmp";
    }
    if (read >= 12 && std::memcmp(bytes.data() + 4, "ftyp", 4) == 0 &&
        (std::memcmp(bytes.data() + 8, "avif", 4) == 0 ||
         std::memcmp(bytes.data() + 8, "avis", 4) == 0 ||
         std::memcmp(bytes.data() + 8, "heic", 4) == 0 ||
         std::memcmp(bytes.data() + 8, "mif1", 4) == 0)) {
        return "unsupported";
    }
    return image_codec(path);
}

void inspect(Image& image, Report& report, bool selected) {
    const int fd = open(image.path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    struct stat file_status {};
    if (fd < 0 || fstat(fd, &file_status) != 0 || !S_ISREG(file_status.st_mode)) {
        if (fd >= 0) {
            close(fd);
        }
        report.issue(selected, image.path, "Could not open the image as a regular file.");
        return;
    }

    const auto codec = detected_codec(fd, image.path);
    if (codec == "unsupported" || codec.empty()) {
        close(fd);
        report.issue(selected && codec.empty(), image.path,
                     "Native header dimensions unavailable for this trainer-visible file.");
        return;
    }

    if (codec == "tiff") {
        image.size = tiff_size(fd, image.path);
    } else {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> input(fdopen(fd, "rb"), std::fclose);
        if (!input) {
            close(fd);
            report.issue(selected, image.path, "Could not read the image file.");
            return;
        }
        image.size = codec == "webp" ? webp_size(input.get())
                                          : pixbuf_size(input.get(), codec);
    }

    if (!image.size.known()) {
        report.issue(selected, image.path, "Unreadable image header or file type differs from its extension.");
    }
}

using FileIdentity = std::pair<dev_t, ino_t>;
using Groups = std::map<std::string, std::vector<Image>>;

void walk(const fs::path& directory, const fs::path& root, Groups& groups,
          std::set<FileIdentity>& ancestors, Report& report, DirectoryScan& result) {
    struct stat status {};
    if (stat(directory.c_str(), &status) != 0) {
        report.issue(true, directory, "Could not read directory (" +
                     std::string(std::strerror(errno)) + ").");
        result.complete = false;
        return;
    }
    const FileIdentity identity{status.st_dev, status.st_ino};
    if (!ancestors.insert(identity).second) {
        report.issue(false, directory, "Skipped a directory link that leads to an ancestor.");
        return;
    }

    std::error_code error;
    std::vector<fs::directory_entry> entries;
    const fs::directory_iterator end;
    for (fs::directory_iterator entry(directory, error); entry != end && !error;
         entry.increment(error)) {
        entries.push_back(*entry);
    }
    if (error) {
        report.issue(true, directory, "Could not read directory (" + error.message() + ").");
        result.complete = false;
    }

    std::sort(entries.begin(), entries.end());
    for (const auto& entry : entries) {
        const auto& path = entry.path();
        if (path.filename().native().front() == '.') {
            continue;
        }
        if (entry.is_regular_file(error)) {
            auto relative = path.lexically_relative(root);
            auto stem = fs::path(relative).replace_extension().native();
            groups[stem].push_back({path, relative.native()});
            ++result.count;
        } else if (error == std::errc::no_such_file_or_directory) {
            // The trainer's DirEntry predicates skip dangling symlinks.
            continue;
        } else if (!error && entry.is_directory(error)) {
            walk(path, root, groups, ancestors, report, result);
        }
        if (error) {
            report.issue(true, path,
                         "Could not inspect filesystem entry (" + error.message() + ").");
            result.complete = false;
        }
    }
    ancestors.erase(identity);
}

DirectoryScan discover(const fs::path& root, Groups& groups, Report& report) {
    DirectoryScan result;
    std::set<FileIdentity> ancestors;
    walk(root, root, groups, ancestors, report, result);
    return result;
}

void sort_groups(Groups& groups, std::string_view side, Report& report, bool paired) {
    for (auto& [stem, files] : groups) {
        std::sort(files.begin(), files.end(), [](const Image& lhs, const Image& rhs) {
            return lhs.relative < rhs.relative;
        });
        if (paired && files.size() > 1) {
            report.issue(true, files.front().path,
                         "Duplicate " + std::string(side) + " relative stem '" + stem +
                         "' (" + std::to_string(files.size()) + " files).");
        }
    }
}

struct Selection {
    Image* gt;
    Image* lq;
};

using TemplateLookup = std::map<std::pair<std::size_t, std::string>, std::string>;

struct Naming {
    bool strict = true;
    std::vector<std::string> literals;
    const TemplateLookup* formatted = nullptr;

    std::optional<std::string> expected(const std::string& stem, std::size_t root_id,
                                        const Image& gt) const {
        if (strict) {
            return stem;
        }
        if (formatted != nullptr) {
            const auto found = formatted->find({root_id, gt.relative});
            if (found == formatted->end()) {
                return std::nullopt;
            }
            return found->second;
        }
        const fs::path relative(stem);
        const auto base = relative.filename().native();
        std::string expanded = literals.front();
        for (std::size_t index = 1; index < literals.size(); ++index) {
            expanded += base;
            expanded += literals[index];
        }
        // Python join preserves a trailing separator on an empty expansion.
        if (expanded.empty() && !relative.parent_path().empty()) {
            return relative.parent_path().native() + '/';
        }
        return (relative.parent_path() / expanded).native();
    }
};

std::vector<Selection> select_paths(
    Groups& gt, Groups& lq, bool has_lq, bool gt_complete, bool lq_complete,
    const Naming& naming, std::size_t root_id, Report& report) {
    std::vector<Selection> rows;
    std::set<std::string> expected_lq;
    std::map<const Image*, std::size_t> claims;
    for (auto& [stem, files] : gt) {
        for (auto& file : files) {
            Image* partner = nullptr;
            if (has_lq) {
                const auto expected = naming.expected(stem, root_id, file);
                if (!expected) {
                    report.issue(true, file.path, "GT contents changed while applying the filename template.");
                } else {
                    expected_lq.insert(*expected);
                    const auto found = lq.find(*expected);
                    if (found != lq.end() && files.size() == 1 && found->second.size() == 1) {
                        partner = &found->second.front();
                        ++claims[partner];
                    } else if (found == lq.end() && lq_complete) {
                        report.issue(naming.strict, file.path, "Missing LQ image for expected relative stem '" +
                                     *expected + "'.");
                    }
                }
            }
            rows.push_back({&file, partner});
        }
    }
    for (auto& row : rows) {
        if (row.lq != nullptr && claims[row.lq] > 1) {
            report.issue(true, row.gt->path, "Several GT files map to the same LQ file '" +
                         row.lq->relative + "'.");
            row.lq = nullptr;
        }
    }
    if (gt_complete) {
        for (const auto& [stem, files] : lq) {
            if (expected_lq.count(stem) == 0) {
                report.issue(naming.strict, files.front().path,
                             "No selected GT file expects LQ relative stem '" + stem + "'.");
            }
        }
    }
    if (!naming.strict || !has_lq) {
        std::sort(rows.begin(), rows.end(), [](const Selection& lhs, const Selection& rhs) {
            return lhs.gt->relative < rhs.gt->relative;
        });
    }
    return rows;
}

std::string describe_size(const Dimensions& size) {
    return std::to_string(size.width) + " x " + std::to_string(size.height);
}

struct Geometry {
    std::uint64_t scale = 0;
    std::uint64_t patch = 0;
    bool train = true;
};

std::string_view geometry_status(const Selection& row, const Geometry& geometry, bool has_lq,
                                 Report& report) {
    const auto& gt = *row.gt;
    bool too_small = false;
    if (geometry.train && geometry.patch && gt.size.known() &&
        std::min(gt.size.width, gt.size.height) < geometry.patch) {
        report.issue(has_lq && row.lq != nullptr, gt.path, "GT is smaller than the requested patch (" +
                     std::to_string(geometry.patch) + ").");
        too_small = true;
    }
    if (geometry.train && geometry.patch && geometry.scale && row.lq != nullptr &&
        row.lq->size.known() &&
        std::min(row.lq->size.width, row.lq->size.height) < geometry.patch / geometry.scale) {
        report.issue(true, row.lq->path, "LQ is smaller than the required patch (" +
                     std::to_string(geometry.patch / geometry.scale) + ").");
        too_small = true;
    }
    if (row.lq == nullptr) {
        return "";
    }
    if (!geometry.scale || !gt.size.known() || !row.lq->size.known()) {
        return too_small ? "mismatch" : "unknown";
    }

    const auto& lq = *row.lq;
    const auto& a = gt.size;
    const auto& b = lq.size;
    const auto scale = geometry.scale;
    const bool large_enough = a.width / b.width >= scale && a.height / b.height >= scale;
    const bool exact = large_enough && a.width % b.width == 0 && a.height % b.height == 0 &&
                       a.width / b.width == scale && a.height / b.height == scale;
    if (!exact) {
        const bool will_crop = !geometry.train && large_enough;
        report.issue(!will_crop, gt.path,
                     std::string(will_crop ? "Validation crops excess GT at the top-left. " :
                                             "Scale mismatch. ") +
                     "GT is " + describe_size(a) + ", LQ is " + describe_size(b) +
                     ", and the requested scale is " + std::to_string(scale) + ".");
        return will_crop ? "crop" : "mismatch";
    }
    return too_small ? "mismatch" : "exact";
}

// NUL separators preserve filesystem names with whitespace or control characters.
template <typename... Fields>
void record(const Fields&... fields) {
    ((std::cout << fields << '\0'), ...);
}

using ValidationFiles = std::map<FileIdentity, fs::path>;

void check_shared_file(const Image& image, const ValidationFiles& validation, Report& report) {
    if (validation.empty()) {
        return;
    }
    struct stat status {};
    if (stat(image.path.c_str(), &status) == 0) {
        const auto match = validation.find({status.st_dev, status.st_ino});
        if (match != validation.end()) {
            report.issue(false, image.path, "Same physical file as validation path '" +
                         match->second.native() + "'.");
        }
    }
}

int run_folders(char** begin, char** end, const Naming& naming, const Geometry& geometry,
                const ValidationFiles& validation, Report& report, bool training_paths) {
    std::size_t total_gt = 0, total_lq = 0, total_pairs = 0, samples_written = 0;
    bool expects_pairs = false;
    for (auto roots = begin; roots != end; roots += 2) {
        const fs::path gt_root = roots[0];
        const fs::path lq_root = roots[1];
        report.root_id = static_cast<std::size_t>((roots - begin) / 2 + 1);
        const bool has_lq = !lq_root.empty();
        expects_pairs = expects_pairs || has_lq;
        Groups gt, lq;
        const auto gt_count = discover(gt_root, gt, report);
        const auto lq_count = has_lq ? discover(lq_root, lq, report) : DirectoryScan{};
        if (!training_paths && gt_count.complete && gt_count.count == 0) {
            report.issue(false, gt_root, "No trainer-visible GT files found in this root.");
        }
        if (!training_paths && has_lq && lq_count.complete && lq_count.count == 0) {
            report.issue(false, lq_root, "No trainer-visible LQ files found in this root.");
        }
        std::error_code equivalent_error;
        if (!training_paths && has_lq && fs::equivalent(gt_root, lq_root, equivalent_error)) {
            report.issue(false, gt_root, "GT and LQ refer to the same directory.");
        }
        sort_groups(gt, "GT", report, has_lq);
        sort_groups(lq, "LQ", report, true);
        const auto rows = select_paths(gt, lq, has_lq, gt_count.complete, lq_count.complete,
                                       naming, report.root_id, report);
        if (training_paths && report.errors_count != 0) {
            return 3;
        }

        std::size_t paired = 0;
        for (const auto& row : rows) {
            auto& image = *row.gt;
            auto* peer = row.lq;
            paired += peer != nullptr;
            if (training_paths) {
                if (peer != nullptr) {
                    record("pair", report.root_id, image.relative, peer->relative);
                }
                continue;
            }

            if (peer != nullptr || !has_lq) {
                inspect(image, report, true);
                check_shared_file(image, validation, report);
                if (peer != nullptr) {
                    inspect(*peer, report, true);
                    check_shared_file(*peer, validation, report);
                }
            } else if (samples_written < sample_limit) {
                inspect(image, report, false);
            }
            const auto alignment = geometry_status(row, geometry, has_lq, report);
            if (samples_written < sample_limit) {
                record("sample", report.root_id, image.relative, image.path.native(),
                       peer ? peer->path.native() : "", image.size.width, image.size.height,
                       peer ? peer->size.width : 0, peer ? peer->size.height : 0, alignment);
                ++samples_written;
            }
        }
        if (!training_paths) {
            if (has_lq && !naming.strict && paired == 0 && gt_count.complete &&
                lq_count.complete && gt_count.count && lq_count.count) {
                report.issue(false, gt_root, "The filename template selected no pairs in this root group.");
            }
            record("root", report.root_id, gt_root.native(), lq_root.native(),
                   gt_count.count, lq_count.count, paired);
            total_gt += gt_count.count;
            total_lq += lq_count.count;
            total_pairs += paired;
        }
    }

    if (!training_paths) {
        if (total_gt == 0 || (expects_pairs && total_pairs == 0)) {
            report.root_id = 0;
            report.issue(true, begin[0], expects_pairs ? "No unambiguous pairs were selected."
                                                       : "No trainer-visible GT files were selected.");
        }
        std::size_t issues_written = 0;
        for (const auto& error : report.errors) {
            record("issue", error.root_id, "error", error.path.native(), error.message);
            ++issues_written;
        }
        for (const auto& warning : report.warnings) {
            if (issues_written == issue_limit) {
                break;
            }
            record("issue", warning.root_id, "warning", warning.path.native(), warning.message);
            ++issues_written;
        }
        record("counts", total_gt, total_lq, total_pairs, report.issue_count,
               report.errors_count, sample_limit, issue_limit);
    }
    return std::cout ? 0 : 1;
}

int list_gt(char** begin, char** end) {
    Report report;
    for (auto argument = begin; argument != end; ++argument) {
        Groups files;
        if (!discover(*argument, files, report).complete) {
            std::cerr << report.errors.front().path << ' ' << report.errors.front().message << '\n';
            return 1;
        }
        const auto id = static_cast<std::size_t>(argument - begin + 1);
        for (const auto& [stem, images] : files) {
            for (const auto& image : images) {
                record("file", id, image.relative);
            }
        }
    }
    return std::cout ? 0 : 1;
}

std::uint64_t read_unsigned(std::string_view argument) {
    std::uint64_t value = 0;
    const auto* end = argument.data() + argument.size();
    const auto parsed = std::from_chars(argument.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("Expected a nonnegative integer.");
    }
    return value;
}

TemplateLookup read_mapping() {
    TemplateLookup result;
    std::string id, path, expected;
    while (std::getline(std::cin, id, '\0')) {
        if (std::cin.eof() || !std::getline(std::cin, path, '\0') || std::cin.eof() ||
            !std::getline(std::cin, expected, '\0') || std::cin.eof()) {
            throw std::invalid_argument("Incomplete filename template mapping.");
        }
        result.emplace(std::make_pair(read_unsigned(id), path), expected);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string_view(argv[1]) == "--list-gt") {
            return list_gt(argv + 2, argv + argc);
        }
        Report report;
        if (argc >= 4 && argc % 2 == 0 && std::string_view(argv[1]) == "--train-pairs") {
            return run_folders(argv + 2, argv + argc, Naming{}, Geometry{}, {}, report, true);
        }
        if (argc < 10 || std::string_view(argv[1]) != "--scan") {
            throw std::invalid_argument("Expected --scan, --list-gt, or --train-pairs.");
        }
        TIFFSetErrorHandler(nullptr);
        TIFFSetWarningHandler(nullptr);
        const Geometry geometry{read_unsigned(argv[2]), read_unsigned(argv[3]),
                                std::string_view(argv[4]) == "train"};
        const std::string_view naming_mode(argv[5]);
        const auto parts = read_unsigned(argv[6]);
        const auto validation_roots = read_unsigned(argv[7]);
        const auto extra = static_cast<std::uint64_t>(argc - 10);
        if (parts > extra || validation_roots > extra - parts ||
            (argc - 8 - parts - validation_roots) % 2 != 0) {
            throw std::invalid_argument("Invalid filename template or root arguments.");
        }
        Naming naming;
        TemplateLookup mapping;
        if (naming_mode == "format" && parts > 0) {
            naming.strict = false;
            naming.literals.assign(argv + 8, argv + 8 + parts);
        } else if (naming_mode == "mapped" && parts == 0) {
            naming.strict = false;
            mapping = read_mapping();
            naming.formatted = &mapping;
        } else if (naming_mode != "stem" || parts != 0) {
            throw std::invalid_argument("Invalid filename template mode.");
        }
        ValidationFiles validation;
        report.root_id = 0;
        auto validation_start = argv + 8 + parts;
        auto dataset_start = validation_start + validation_roots;
        for (auto argument = validation_start; argument != dataset_start; ++argument) {
            Groups files;
            discover(*argument, files, report);
            for (const auto& [stem, entries] : files) {
                for (const auto& entry : entries) {
                    struct stat status {};
                    if (stat(entry.path.c_str(), &status) == 0) {
                        validation.try_emplace({status.st_dev, status.st_ino}, entry.path);
                    }
                }
            }
        }
        return run_folders(dataset_start, argv + argc, naming, geometry, validation,
                           report, false);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
