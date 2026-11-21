#ifndef __clang__
#error "Build traiNNer-GUI native tools with LLVM Clang."
#endif
#include <charconv>
#include <filesystem>
#include <iostream>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Item = std::pair<unsigned,std::string>;

int main(int argc, char** argv) {
    if (argc != 8) return 2;
    const fs::path root(argv[1]);
    const std::string mode(argv[2]), filter(argv[3]);
    const bool hidden = std::string_view(argv[4]) == "1";
    const Item after{std::string_view(argv[5]) == "file" ? 1u : 0u, argv[6]};
    unsigned limit{};
    auto last = std::string_view(argv[7]);
    auto parse = std::from_chars(last.data(), last.data()+last.size(),limit);
    if (parse.ec != std::errc{} || parse.ptr != last.data()+last.size() || limit == 0 ||
        limit > 1024 || (mode != "all" && mode != "folders" && mode != "files")) return 2;
    try {
        std::priority_queue<Item> page;
        std::size_t after_count = 0;
        for (const auto& entry : fs::directory_iterator(root)) {
            auto name = entry.path().filename().string();
            if ((!hidden && !name.empty() && name.front() == '.') ||
                (!filter.empty() && name.find(filter) == std::string::npos)) continue;
            std::error_code e;
            bool directory = entry.is_directory(e);
            if (e || (directory && mode == "files") || (!directory && mode == "folders")) continue;
            if (!directory && !entry.is_regular_file(e)) continue;
            Item key{directory ? 0u : 1u,std::move(name)};
            if (!after.second.empty() && key <= after) continue;
            ++after_count;
            if (page.size() < limit) page.push(std::move(key));
            else if (key < page.top()) { page.pop(); page.push(std::move(key)); }
        }
        std::vector<Item> ordered;
        while (!page.empty()) { ordered.push_back(page.top()); page.pop(); }
        for (auto it=ordered.rbegin();it!=ordered.rend();++it) {
            std::cout << (it->first ? 'f' : 'd') << it->second << '\0';
        }
        std::cout << (after_count > limit ? "more" : "end") << '\0';
        return std::cout ? 0 : 1;
    } catch (const fs::filesystem_error& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
