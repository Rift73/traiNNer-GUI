#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace host {

namespace fs = std::filesystem;
using std::string;
using std::wstring;

constexpr wchar_t title[] = L"traiNNer-GUI";

struct Exit : std::runtime_error {
    Exit(int code, const string& message) : std::runtime_error(message), code(code) {}
    int code;
};

class Handle {
public:
    explicit Handle(HANDLE handle = nullptr) : handle_(handle) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    HANDLE get() const { return handle_; }
    explicit operator bool() const { return handle_ && handle_ != INVALID_HANDLE_VALUE; }
    void reset(HANDLE next = nullptr) {
        if (*this) CloseHandle(handle_);
        handle_ = next;
    }
private:
    HANDLE handle_;
};

struct ProcessResult {
    DWORD code{};
    string out;
    string err;
};

class SetupDialogs {
public:
    explicit SetupDialogs(bool interactive);
    ~SetupDialogs();
    SetupDialogs(const SetupDialogs&) = delete;
    SetupDialogs& operator=(const SetupDialogs&) = delete;

    bool choose_managed();
    bool confirm(const wchar_t* message);
    fs::path folder(const wchar_t* prompt, const fs::path& initial = {});
    void progress(const wchar_t* message);

private:
    bool interactive_;
    struct Progress;
    Progress* progress_ = nullptr;
    void stop_progress();
};

wstring wide(const string& text);
string utf8(const wstring& text);
wstring env(const wchar_t* key);
fs::path launcher_folder();
fs::path profile_folder();
fs::path local_ntfs_folder(const fs::path& path);
bool same_text(const wstring& first, const wstring& second);
bool elevated();
bool service_present(const wchar_t* name);
bool virtual_machine_platform_enabled();
unsigned long windows_build_number();
string trim_lines(string text);
bool wsl_registration_error(const ProcessResult& result);
ProcessResult wsl_status();
ProcessResult repair_wsl_package();
DWORD enable_vm_platform();
string boot_time();
ProcessResult wsl(const std::vector<wstring>& args, DWORD timeout = 60000, HANDLE input = nullptr,
                  const string& ready = {}, const std::function<void()>& while_ready = {});
DWORD install_wsl();
string sha256(HANDLE file);
void stream(DWORD id, const string& text);
string read_text(const fs::path& path);
void atomic_text(const fs::path& path, const string& text);
bool valid_project_id(const string& id);
string project_id(const fs::path& path, bool create);

} // namespace host
