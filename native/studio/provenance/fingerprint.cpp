// Content-address the trainable checkout, including uncommitted source and native kernels.
#ifndef __clang__
#error LLVM Clang is required.
#endif

#include <glib.h>
#include <sys/stat.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace {
using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
using Checksum = std::unique_ptr<GChecksum, decltype(&g_checksum_free)>;

struct Entry { std::string signature, digest; };
using Entries = std::map<std::string,Entry>;

std::string signature(const struct stat& st) {
    return std::to_string(st.st_dev) + ":" + std::to_string(st.st_ino) + ":" +
           std::to_string(st.st_size) + ":" + std::to_string(st.st_mtim.tv_sec) + ":" +
           std::to_string(st.st_mtim.tv_nsec) + ":" + std::to_string(st.st_ctim.tv_sec) + ":" +
           std::to_string(st.st_ctim.tv_nsec);
}

Entries read_cache(const fs::path& path) {
    Entries entries;
    std::ifstream input(path,std::ios::binary);
    std::string key, stamp, digest;
    while (std::getline(input,key,'\0') && std::getline(input,stamp,'\0') && std::getline(input,digest,'\0'))
        if (!key.empty() && digest.size()==64 && digest.find_first_not_of("0123456789abcdef")==std::string::npos)
            entries.emplace(key,Entry{stamp,digest});
    return entries;
}

std::string hash(FILE* file) {
    Checksum sha(g_checksum_new(G_CHECKSUM_SHA256),g_checksum_free);
    std::array<unsigned char,65536> buffer;
    std::size_t n;
    while ((n=std::fread(buffer.data(),1,buffer.size(),file))>0) g_checksum_update(sha.get(),buffer.data(),n);
    if (std::ferror(file)) throw std::runtime_error("Error reading source file.");
    return g_checksum_get_string(sha.get());
}

Entry identify(const fs::path& file, const Entry* prior) {
    struct stat st{};
    if (::stat(file.c_str(),&st)!=0 || !S_ISREG(st.st_mode))
        throw std::runtime_error("Missing training source: " + file.string());
    std::string id = signature(st);
    if (prior && prior->signature == id) return *prior;
    File fp(std::fopen(file.c_str(),"rb"),std::fclose);
    if (!fp || ::fstat(fileno(fp.get()),&st)!=0)
        throw std::runtime_error("Cannot open training source: " + file.string());
    id = signature(st);
    const auto digest = hash(fp.get());
    if (::stat(file.c_str(),&st)!=0 || signature(st)!=id)
        throw std::runtime_error("Source changed during capture: " + file.string());
    return {id,digest};
}

void walk(const fs::path& root, const fs::path& relative, std::set<std::string>& paths) {
    static const std::set<std::string> extensions{
        ".py",".cu",".cuh",".cuda",".ptx",".c",".cpp",".cc",".h",".hpp",".so",".json",".toml"
    };
    const auto base = root / relative;
    if (fs::is_regular_file(base)) {
        paths.emplace(relative.native());
        return;
    }
    for (fs::recursive_directory_iterator it(base),end; it!=end; ++it) {
        const auto& file = it->path();
        const auto name = file.filename().native();
        if (it->is_directory()) {
            if (name=="__pycache__" || name.front()=='.' || name=="CMakeFiles" || name=="tests")
                it.disable_recursion_pending();
        } else if ((name=="Makefile" || extensions.count(file.extension().native())) && it->is_regular_file()) {
            paths.emplace(file.lexically_relative(root).native());
        }
    }
}

void inventory(const fs::path& root, const fs::path& cache) {
    const auto saved = read_cache(cache);
    std::set<std::string> paths;
    for (const auto* item : {"train.py","traiNNer","trainner_config","trainner_data","native/studio/data","pyproject.toml"})
        walk(root,item,paths);
    for (const auto& key : paths) {
        auto found=saved.find(key);
        auto entry=identify(root/key, found==saved.end()?nullptr:&found->second);
        std::cout << key << '\0' << entry.signature << '\0' << entry.digest << '\0';
    }
    if (!std::cout) throw std::runtime_error("Could not write source inventory.");
}
}

int main(int argc,char** argv) {
    try {
        if (argc!=3) throw std::runtime_error("Usage: fingerprint CHECKOUT CACHE");
        inventory(fs::absolute(argv[1]),argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
