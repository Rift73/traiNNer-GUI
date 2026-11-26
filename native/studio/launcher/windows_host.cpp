#include "windows_host.hpp"

#include <bcrypt.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wbemidl.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <exception>
#include <memory>
#include <regex>
#include <type_traits>

namespace host {
namespace {

using Microsoft::WRL::ComPtr;

constexpr size_t output_limit = 65536;

std::runtime_error win_error(const string& message, DWORD error = GetLastError()) {
    return std::runtime_error(message + " (Windows error " + std::to_string(error) + ").");
}

fs::path system_exe(const wchar_t* name) {
    wchar_t path[MAX_PATH]{};
    const auto len = GetSystemDirectoryW(path, MAX_PATH);
    if (!len || len >= MAX_PATH) throw win_error("Could not locate the Windows system directory");
    return fs::path(path) / name;
}

wstring quote(const wstring& argument) {
    if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == wstring::npos) return argument;
    wstring out(1, L'"');
    size_t backslashes = 0;
    for (wchar_t ch : argument) {
        if (ch == L'\\') {
            ++backslashes;
            continue;
        }
        out.append(ch == L'"' ? 2 * backslashes + 1 : backslashes, L'\\');
        backslashes = 0;
        out.push_back(ch);
    }
    out.append(2 * backslashes, L'\\');
    return out + L'"';
}

struct Capture {
    explicit Capture(SECURITY_ATTRIBUTES& attributes) {
        HANDLE read{}, write{};
        if (!CreatePipe(&read, &write, &attributes, 0))
            throw win_error("Could not capture WSL output");
        reader.reset(read);
        writer.reset(write);
        if (!SetHandleInformation(reader.get(), HANDLE_FLAG_INHERIT, 0))
            throw win_error("Could not configure WSL output");
    }

    void drain(string& text) const {
        char buffer[4096];
        DWORD available{}, count{};
        // Return to the wait loop even if the child is writing continuously.
        for (int chunks = 0; chunks < 16; ++chunks) {
            if (!PeekNamedPipe(reader.get(), nullptr, 0, nullptr, &available, nullptr)) {
                if (GetLastError() == ERROR_BROKEN_PIPE) return;
                throw win_error("Could not read WSL output");
            }
            if (!available) return;
            if (!ReadFile(reader.get(), buffer, std::min(available, DWORD(sizeof(buffer))), &count, nullptr))
                throw win_error("Could not read WSL output");
            text.append(buffer, count);
            if (text.size() > output_limit) text.erase(0, text.size() - output_limit);
        }
    }

    Handle reader, writer;
};

void write_all(HANDLE file, const string& contents) {
    DWORD written{};
    if (!WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()), &written, nullptr)
        || written != contents.size())
        throw win_error("Could not write the desktop settings");
}

struct Bstr {
    explicit Bstr(const wchar_t* value) : value(SysAllocString(value)) {
        if (!this->value) throw std::bad_alloc();
    }
    ~Bstr() { SysFreeString(value); }
    BSTR value;
};

struct Variant {
    Variant() { VariantInit(&value); }
    ~Variant() { VariantClear(&value); }
    VARIANT value{};
};

void query_system(const wchar_t* query, const wchar_t* property, Variant& result) {
    ComPtr<IWbemLocator> locator;
    ComPtr<IWbemServices> service;
    ComPtr<IEnumWbemClassObject> rows;
    ComPtr<IWbemClassObject> row;
    ULONG count{};
    auto check = [&](HRESULT hr, const char* stage) {
        if (FAILED(hr)) {
            char hex[16];
            snprintf(hex, sizeof(hex), "0x%08lX", static_cast<unsigned long>(hr));
            throw std::runtime_error("Windows could not read " + utf8(property) + " (" + stage
                                     + ", " + hex + ").");
        }
    };
    check(CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&locator)),
          "create WMI locator");
    check(locator->ConnectServer(Bstr(L"ROOT\\CIMV2").value, nullptr, nullptr, nullptr,
                                 WBEM_FLAG_CONNECT_USE_MAX_WAIT, nullptr, nullptr, &service), "connect to WMI");
    check(CoSetProxyBlanket(service.Get(), RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                            RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE),
          "set WMI proxy security");
    check(service->ExecQuery(Bstr(L"WQL").value, Bstr(query).value,
                             WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &rows),
          "run WMI query");
    const HRESULT next = rows->Next(10000, 1, &row, &count);
    check(next, "read WMI result");
    if (!count)
        throw std::runtime_error("Windows returned " + string(next == WBEM_S_TIMEDOUT ? "a timeout" : "no row")
                                 + " for " + utf8(property) + ".");
    check(row->Get(property, 0, &result.value, nullptr, nullptr), "read WMI property");
}

} // namespace

wstring wide(const string& text) {
    if (text.empty()) return {};
    const int len = static_cast<int>(text.size());
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.data(), len, nullptr, 0);
    if (!count) throw win_error("Could not decode UTF-8");
    wstring result(count, 0);
    MultiByteToWideChar(CP_UTF8, 0, text.data(), len, result.data(), count);
    return result;
}

string utf8(const wstring& text) {
    if (text.empty()) return {};
    const int len = static_cast<int>(text.size());
    const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), len,
                                          nullptr, 0, nullptr, nullptr);
    if (!count) throw win_error("Could not encode a Windows path");
    string result(count, 0);
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), len, result.data(), count,
                        nullptr, nullptr);
    return result;
}

wstring env(const wchar_t* key) {
    const DWORD count = GetEnvironmentVariableW(key, nullptr, 0);
    if (!count) return {};
    wstring value(count, 0);
    value.resize(GetEnvironmentVariableW(key, value.data(), count));
    return value;
}

fs::path launcher_folder() {
    wstring path(32768, 0);
    const auto len = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!len || len >= path.size()) throw win_error("Could not locate the launcher");
    path.resize(len);
    return fs::path(path).parent_path();
}

fs::path profile_folder() {
    const fs::path local = env(L"LOCALAPPDATA");
    if (!local.is_absolute()) throw std::runtime_error("Could not locate Local AppData.");
    return local / title;
}

fs::path local_ntfs_folder(const fs::path& path) {
    if (!path.is_absolute() || !fs::is_directory(path))
        throw std::runtime_error("Choose an existing local NTFS folder for the workspace.");
    Handle directory(CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr));
    if (!directory) throw win_error("Could not open the workspace folder");
    wstring real(32768, 0);
    const auto len = GetFinalPathNameByHandleW(directory.get(), real.data(),
                                               static_cast<DWORD>(real.size()), FILE_NAME_NORMALIZED);
    if (!len || len >= real.size()) throw win_error("Could not resolve the workspace folder");
    real.resize(len);
    if (real.rfind(L"\\\\?\\", 0) == 0) real.erase(0, 4);
    wchar_t volume[MAX_PATH]{}, filesystem[32]{};
    if (real.size() < 3 || real[1] != L':' || real[2] != L'\\'
        || !GetVolumePathNameW(real.c_str(), volume, MAX_PATH)
        || GetDriveTypeW(volume) != DRIVE_FIXED
        || !GetVolumeInformationW(volume, nullptr, 0, nullptr, nullptr, nullptr, filesystem, 32)
        || !same_text(filesystem, L"NTFS"))
        throw std::runtime_error("Choose a folder on a local fixed NTFS drive.");
    return real;
}

bool same_text(const wstring& first, const wstring& second) {
    return CompareStringOrdinal(first.c_str(), -1, second.c_str(), -1, TRUE) == CSTR_EQUAL;
}

bool elevated() {
    HANDLE raw{};
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw))
        throw win_error("Could not check the Windows account");
    Handle token(raw);
    TOKEN_ELEVATION state{};
    DWORD size{};
    if (!GetTokenInformation(token.get(), TokenElevation, &state, sizeof(state), &size))
        throw win_error("Could not check Windows elevation");
    return state.TokenIsElevated != 0;
}

bool service_present(const wchar_t* name) {
    const auto close = &CloseServiceHandle;
    std::unique_ptr<std::remove_pointer_t<SC_HANDLE>, decltype(close)>
        manager(OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT), close);
    if (!manager) throw win_error("Could not inspect Windows services");
    std::unique_ptr<std::remove_pointer_t<SC_HANDLE>, decltype(close)>
        service(OpenServiceW(manager.get(), name, SERVICE_QUERY_STATUS), close);
    if (service) return true;
    const auto error = GetLastError();
    if (error != ERROR_SERVICE_DOES_NOT_EXIST) throw win_error("Could not inspect a Windows service", error);
    return false;
}

bool virtual_machine_platform_enabled() {
    Variant result;
    query_system(L"SELECT InstallState FROM Win32_OptionalFeature WHERE Name='VirtualMachinePlatform'",
                 L"InstallState", result);
    if (result.value.vt != VT_I4) throw std::runtime_error("Could not read Virtual Machine Platform status.");
    return result.value.lVal == 1;
}

unsigned long windows_build_number() {
    wchar_t build[16]{};
    DWORD bytes = sizeof(build);
    const auto error = RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
                                    L"CurrentBuildNumber", RRF_RT_REG_SZ, nullptr, build, &bytes);
    if (error != ERROR_SUCCESS) throw win_error("Could not read the Windows build number", error);
    return std::stoul(build);
}

string boot_time() {
    Variant result;
    query_system(L"SELECT LastBootUpTime FROM Win32_OperatingSystem", L"LastBootUpTime", result);
    if (result.value.vt != VT_BSTR || !result.value.bstrVal)
        throw std::runtime_error("Could not read the last Windows reboot time.");
    return utf8(result.value.bstrVal);
}

string trim_lines(string text) {
    if (text.find('\0') != string::npos && text.size() % 2 == 0) {
        wstring decoded;
        for (size_t i = 0; i < text.size(); i += 2) {
            const auto lo = static_cast<unsigned char>(text[i]);
            const auto hi = static_cast<unsigned char>(text[i + 1]);
            decoded.push_back(static_cast<wchar_t>(lo | (hi << 8)));
        }
        if (!decoded.empty() && decoded.front() == 0xfeff) decoded.erase(0, 1);
        text = utf8(decoded);
    }
    const auto begin = text.find_first_not_of(" \r\n\t");
    if (begin == string::npos) return {};
    return text.substr(begin, text.find_last_not_of(" \r\n\t") - begin + 1);
}

bool wsl_registration_error(const ProcessResult& result) {
    if (result.code == static_cast<DWORD>(REGDB_E_CLASSNOTREG)) return true;
    for (const auto* output : {&result.err, &result.out}) {
        auto message = trim_lines(*output);
        std::transform(message.begin(), message.end(), message.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (message.find("regdb_e_classnotreg") != string::npos
            || message.find("0x80040154") != string::npos) return true;
    }
    return false;
}

static ProcessResult capture_process(const fs::path& exe, const std::vector<wstring>& args,
                                     DWORD timeout, HANDLE source, const string& ready,
                                     const std::function<void()>& while_ready, bool probing_wsl = false) {
    wstring command = quote(exe.wstring());
    for (const auto& arg : args) command += L" " + quote(arg);

    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    Capture out(attributes), err(attributes);
    Handle input, keep_open;
    if (while_ready) {
        if (source || ready.empty()) throw std::logic_error("Invalid WSL activation handshake");
        HANDLE read{}, write{};
        if (!CreatePipe(&read, &write, &attributes, 0)) throw win_error("Could not open the activation pipe");
        input.reset(read);
        keep_open.reset(write);
        if (!SetHandleInformation(write, HANDLE_FLAG_INHERIT, 0))
            throw win_error("Could not configure the activation pipe");
    } else if (source) {
        HANDLE duplicate{};
        if (!DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &duplicate, 0, TRUE,
                             DUPLICATE_SAME_ACCESS))
            throw win_error("Could not pass the source archive to WSL");
        input.reset(duplicate);
    } else {
        input.reset(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
        if (!input) throw win_error("Could not configure WSL input");
    }
    // The long-lived WSL client must not retain the calling terminal's pipes.
    // Inherit only the three handles this launcher created for the child.
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input.get();
    startup.StartupInfo.hStdOutput = out.writer.get();
    startup.StartupInfo.hStdError = err.writer.get();
    SIZE_T attr_size{};
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
    if (!attr_size) throw win_error("Could not size the WSL process attributes");
    std::vector<unsigned char> storage(attr_size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(list, 1, 0, &attr_size))
        throw win_error("Could not initialize WSL process attributes");
    std::unique_ptr<std::remove_pointer_t<LPPROC_THREAD_ATTRIBUTE_LIST>,
                    decltype(&DeleteProcThreadAttributeList)> cleanup(list, DeleteProcThreadAttributeList);
    HANDLE inherited[] = {input.get(), out.writer.get(), err.writer.get()};
    if (!UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                   sizeof(inherited), nullptr, nullptr))
        throw win_error("Could not limit inherited WSL handles");
    startup.lpAttributeList = list;
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
                        nullptr, nullptr, &startup.StartupInfo, &process))
        throw win_error("Could not start " + utf8(exe.filename().wstring()));
    Handle child(process.hProcess), thread(process.hThread);
    out.writer.reset();
    err.writer.reset();
    input.reset();

    ProcessResult result;
    std::exception_ptr callback_error;
    const auto started = GetTickCount64();
    for (;;) {
        out.drain(result.out);
        err.drain(result.err);
        if (!ready.empty() && (result.out.find(ready + "\n") != string::npos
                               || result.out.find(ready + "\r\n") != string::npos)) {
            if (!while_ready) return result;  // Studio's hidden WSL client stays attached.
            if (keep_open) {
                try {
                    while_ready();  // Windows commits while Linux still holds the idle gate.
                } catch (...) {
                    callback_error = std::current_exception();
                }
                keep_open.reset();
            }
        }
        const auto state = WaitForSingleObject(child.get(), 25);
        if (state == WAIT_OBJECT_0) break;
        if (state == WAIT_FAILED) throw win_error("Could not wait for WSL");
        // Inbox wsl.exe can wait for a key for a minute even with redirected input.
        // Its stable registration error is enough to offer repair immediately.
        const bool repair_prompt = probing_wsl && wsl_registration_error(result);
        if (repair_prompt || (timeout != INFINITE && GetTickCount64() - started > timeout)) {
            if (!TerminateProcess(child.get(), ERROR_TIMEOUT) && WaitForSingleObject(child.get(), 0) != WAIT_OBJECT_0)
                throw win_error("Could not stop " + utf8(exe.filename().wstring()));
            if (WaitForSingleObject(child.get(), 5000) != WAIT_OBJECT_0)
                throw std::runtime_error("Timed out stopping " + utf8(exe.filename().wstring()) + ".");
            if (probing_wsl) {
                result.code = repair_prompt ? static_cast<DWORD>(REGDB_E_CLASSNOTREG) : ERROR_TIMEOUT;
                return result;
            }
            throw std::runtime_error(utf8(exe.filename().wstring()) + " did not respond in time.");
        }
    }
    out.drain(result.out);
    err.drain(result.err);
    if (!GetExitCodeProcess(child.get(), &result.code)) throw win_error("Could not read the WSL result");
    if (callback_error) std::rethrow_exception(callback_error);
    if (keep_open && !result.code) throw std::runtime_error("Linux exited before confirming the activation gate.");
    return result;
}

ProcessResult wsl(const std::vector<wstring>& args, DWORD timeout, HANDLE source,
                  const string& ready, const std::function<void()>& while_ready) {
    return capture_process(system_exe(L"wsl.exe"), args, timeout, source, ready, while_ready);
}

ProcessResult wsl_status() {
    const auto exe = system_exe(L"wsl.exe");
    if (!fs::exists(exe)) return {ERROR_FILE_NOT_FOUND, {}, "Windows could not find wsl.exe."};
    return capture_process(exe, {L"--status"}, 15000, nullptr, {}, {}, true);
}

ProcessResult repair_wsl_package() {
    // Run the App Installer alias as the caller. WinGet selects and verifies Microsoft's
    // MSI (WiX), and Windows Installer requests any elevation it needs.
    PWSTR app_data{};
    if (FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &app_data)))
        throw std::runtime_error("Could not locate Windows Package Manager.");
    const fs::path alias = fs::path(app_data) / L"Microsoft\\WindowsApps\\winget.exe";
    CoTaskMemFree(app_data);
    if (GetFileAttributesW(alias.c_str()) == INVALID_FILE_ATTRIBUTES)
        return {ERROR_FILE_NOT_FOUND, {}, "Windows Package Manager (WinGet) is unavailable."};
    return capture_process(alias, {L"install", L"--exact", L"--id", L"Microsoft.WSL", L"--source", L"winget",
                                   L"--installer-type", L"wix", L"--force", L"--silent",
                                   L"--accept-package-agreements", L"--accept-source-agreements",
                                   L"--disable-interactivity"}, INFINITE, nullptr, {}, {});
}

static DWORD elevated_setup(const fs::path& exe, const wchar_t* arguments) {
    SHELLEXECUTEINFOW setup{};
    setup.cbSize = sizeof(setup);
    setup.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI;
    setup.lpVerb = L"runas";
    setup.lpFile = exe.c_str();
    setup.lpParameters = arguments;
    const auto cwd = exe.parent_path();
    setup.lpDirectory = cwd.c_str();
    setup.nShow = SW_SHOWNORMAL;
    if (!ShellExecuteExW(&setup)) {
        const auto error = GetLastError();
        if (error == ERROR_CANCELLED) throw Exit(ERROR_CANCELLED, "Windows setup was cancelled.");
        throw win_error("Could not start " + utf8(exe.filename().wstring()), error);
    }
    Handle process(setup.hProcess);
    DWORD code{};
    if (!process || WaitForSingleObject(process.get(), INFINITE) != WAIT_OBJECT_0
        || !GetExitCodeProcess(process.get(), &code))
        throw win_error("Could not read the result of " + utf8(exe.filename().wstring()));
    return code;
}

DWORD enable_vm_platform() {
    return elevated_setup(system_exe(L"dism.exe"),
                         L"/Online /Enable-Feature /FeatureName:VirtualMachinePlatform /All /NoRestart");
}

DWORD install_wsl() {
    return elevated_setup(system_exe(L"wsl.exe"), L"--install --no-distribution");
}

string sha256(HANDLE file) {
    BCRYPT_ALG_HANDLE algorithm{};
    BCRYPT_HASH_HANDLE hash{};
    std::array<unsigned char, 32> digest{};
    const auto error = std::runtime_error("Could not verify the source archive.");
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) throw error;
    const auto close_provider = [](void* value) { BCryptCloseAlgorithmProvider(value, 0); };
    const std::unique_ptr<void, decltype(close_provider)> provider(algorithm, close_provider);
    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) throw error;
    const auto close_hash = [](void* value) { BCryptDestroyHash(value); };
    const std::unique_ptr<void, decltype(close_hash)> context(hash, close_hash);
    unsigned char buffer[65536];
    DWORD size{};
    do {
        if (!ReadFile(file, buffer, sizeof(buffer), &size, nullptr)
            || BCryptHashData(hash, buffer, size, 0) < 0)
            throw error;
    } while (size);
    if (BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0
        || !SetFilePointerEx(file, LARGE_INTEGER{}, nullptr, FILE_BEGIN))
        throw error;
    constexpr char hex[] = "0123456789abcdef";
    string result;
    for (const auto byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}

void stream(DWORD id, const string& text) {
    DWORD written{};
    auto handle = GetStdHandle(id);
    if ((!handle || handle == INVALID_HANDLE_VALUE) && AttachConsole(ATTACH_PARENT_PROCESS))
        handle = GetStdHandle(id);
    if (handle && handle != INVALID_HANDLE_VALUE) {
        DWORD mode{};
        if (GetConsoleMode(handle, &mode)) {
            const auto value = wide(text);
            WriteConsoleW(handle, value.data(), static_cast<DWORD>(value.size()), &written, nullptr);
        } else {
            WriteFile(handle, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
        }
    }
}

string read_text(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not read " + utf8(path.wstring()) + ".");
    string contents(8192, 0);
    file.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (file.bad() || !file.eof()) throw std::runtime_error("Invalid file " + utf8(path.wstring()) + ".");
    contents.resize(static_cast<size_t>(file.gcount()));
    return contents;
}

void atomic_text(const fs::path& path, const string& text) {
    auto temporary = path;
    temporary += L".tmp";
    // The setup lock serializes writers in this profile. The temporary file is reusable after interruption.
    Handle file(CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file) throw win_error("Could not save the desktop settings");
    write_all(file.get(), text);
    if (!FlushFileBuffers(file.get())) throw win_error("Could not flush the desktop settings");
    file.reset();
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw win_error("Could not save the desktop settings");
}

bool valid_project_id(const string& id) {
    static const std::regex uuid("[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}");
    return std::regex_match(id, uuid);
}

string project_id(const fs::path& path, bool create) {
    const auto marker = path / L".trainner-gui-project.json";
    const auto read_marker = [&]() -> string {
        Handle file(CreateFileW(marker.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
        if (!file) {
            const auto error = GetLastError();
            if (create && error == ERROR_FILE_NOT_FOUND) return {};
            throw win_error("Could not read .trainner-gui-project.json", error);
        }
        BY_HANDLE_FILE_INFORMATION info{};
        char raw[4097]{};
        DWORD size{};
        if (!GetFileInformationByHandle(file.get(), &info)
            || (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))
            || !ReadFile(file.get(), raw, sizeof(raw), &size, nullptr)
            || size > 4096)
            throw std::runtime_error("The workspace marker must be a regular JSON file of at most 4 KiB.");
        string text(raw, size);
        if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);
        static const std::regex version_first(
            R"marker(\s*\{\s*"version"\s*:\s*1\s*,\s*"project_id"\s*:\s*"([^"]+)"\s*\}\s*)marker");
        static const std::regex id_first(
            R"marker(\s*\{\s*"project_id"\s*:\s*"([^"]+)"\s*,\s*"version"\s*:\s*1\s*\}\s*)marker");
        std::smatch match;
        if (!std::regex_match(text, match, version_first) && !std::regex_match(text, match, id_first))
            throw std::runtime_error("Invalid .trainner-gui-project.json. Expected version 1 and a project UUID.");
        string id = match[1];
        std::transform(id.begin(), id.end(), id.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (!valid_project_id(id))
            throw std::runtime_error("Invalid project UUID in .trainner-gui-project.json.");
        return id;
    };

    auto id = read_marker();
    if (!id.empty()) return id;
    GUID guid{};
    wchar_t guid_text[39]{};
    if (FAILED(CoCreateGuid(&guid)) || !StringFromGUID2(guid, guid_text, 39))
        throw std::runtime_error("Could not create a workspace UUID.");
    id = utf8(wstring(guid_text + 1, 36));
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    const auto temporary = path / (L".trainner-gui-project." + wide(id) + L".tmp");
    Handle file(CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file) throw win_error("Could not create the workspace marker");
    try {
        write_all(file.get(), "{\"version\":1,\"project_id\":\"" + id + "\"}\n");
        if (!FlushFileBuffers(file.get())) throw win_error("Could not flush the workspace marker");
    } catch (...) {
        file.reset();
        DeleteFileW(temporary.c_str());
        throw;
    }
    file.reset();
    if (MoveFileExW(temporary.c_str(), marker.c_str(), MOVEFILE_WRITE_THROUGH)) return id;
    const auto error = GetLastError();
    DeleteFileW(temporary.c_str());
    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
        id = read_marker();
        if (!id.empty()) return id;
    }
    throw win_error("Could not publish the workspace marker", error);
}

struct SetupDialogs::Progress {
    ComPtr<IProgressDialog> window;
};

SetupDialogs::SetupDialogs(bool interactive) : interactive_(interactive) {
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)))
        throw std::runtime_error("Could not initialize Windows setup.");
}

SetupDialogs::~SetupDialogs() {
    stop_progress();
    CoUninitialize();
}

void SetupDialogs::stop_progress() {
    if (!progress_) return;
    progress_->window->StopProgressDialog();
    delete progress_;
    progress_ = nullptr;
}

bool SetupDialogs::choose_managed() {
    const TASKDIALOG_BUTTON choices[] = {{100, L"Set up dedicated Ubuntu"},
                                          {101, L"Use an existing WSL checkout"}};
    TASKDIALOGCONFIG dialog{};
    dialog.cbSize = sizeof(dialog);
    dialog.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    dialog.pszWindowTitle = title;
    dialog.pszMainInstruction = L"Choose a trainer environment";
    dialog.pszContent = L"Set up Ubuntu 24.04 for this app, or choose an existing traiNNer-GUI checkout.";
    dialog.cButtons = 2;
    dialog.pButtons = choices;
    dialog.nDefaultButton = 100;
    int chosen{};
    if (FAILED(TaskDialogIndirect(&dialog, &chosen, nullptr, nullptr)))
        throw std::runtime_error("Could not open the setup dialog.");
    if (chosen == IDCANCEL) throw Exit(ERROR_CANCELLED, "Setup was cancelled.");
    return chosen == 100;
}

bool SetupDialogs::confirm(const wchar_t* message) {
    stop_progress();
    return MessageBoxW(nullptr, message, title, MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) == IDYES;
}

fs::path SetupDialogs::folder(const wchar_t* prompt, const fs::path& initial) {
    ComPtr<IFileOpenDialog> dialog;
    FILEOPENDIALOGOPTIONS options{};
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dialog)))
        || FAILED(dialog->GetOptions(&options))
        || FAILED(dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST))
        || FAILED(dialog->SetTitle(prompt)))
        throw std::runtime_error("Could not open the folder picker.");
    ComPtr<IShellItem> start;
    if (!initial.empty() && SUCCEEDED(SHCreateItemFromParsingName(initial.c_str(), nullptr,
                                                                  IID_PPV_ARGS(&start))))
        dialog->SetFolder(start.Get());
    const auto status = dialog->Show(nullptr);
    if (status == HRESULT_FROM_WIN32(ERROR_CANCELLED)) throw Exit(ERROR_CANCELLED, "Setup was cancelled.");
    ComPtr<IShellItem> selected;
    PWSTR path{};
    if (FAILED(status) || FAILED(dialog->GetResult(&selected))
        || FAILED(selected->GetDisplayName(SIGDN_FILESYSPATH, &path)))
        throw std::runtime_error("Could not read the selected folder.");
    std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> allocated(path, CoTaskMemFree);
    return fs::path(allocated.get());
}

void SetupDialogs::progress(const wchar_t* message) {
    if (!interactive_) return;
    if (!progress_) {
        auto created = std::make_unique<Progress>();
        if (FAILED(CoCreateInstance(CLSID_ProgressDialog, nullptr, CLSCTX_INPROC_SERVER,
                                    IID_PPV_ARGS(&created->window))))
            throw std::runtime_error("Could not open the setup progress dialog.");
        created->window->SetTitle(title);
        created->window->StartProgressDialog(nullptr, nullptr,
                                             PROGDLG_MARQUEEPROGRESS | PROGDLG_NOCANCEL |
                                             PROGDLG_NOMINIMIZE, nullptr);
        progress_ = created.release();
    }
    progress_->window->SetLine(1, message, FALSE, nullptr);
}

} // namespace host
