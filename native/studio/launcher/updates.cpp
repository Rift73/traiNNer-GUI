#include "updates.hpp"
#include "windows_host.hpp"

#include <softpub.h>
#include <winhttp.h>
#include <wintrust.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Foundation.Collections.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <utility>
#include <string_view>
#include <tuple>

#ifndef TRAINNER_GUI_RELEASE_TAG
#define TRAINNER_GUI_RELEASE_TAG ""
#endif

namespace updates {
namespace {

using host::fs::path;
using host::Handle;
using host::wide;
using host::utf8;
using namespace std::chrono_literals;
namespace json = winrt::Windows::Data::Json;

constexpr wchar_t exe_name[] = L"traiNNer-GUI.exe";
constexpr wchar_t source_name[] = L"trainner-gui-source.tar.gz";
constexpr wchar_t zip_name[] = L"traiNNer-GUI-Windows-x64.zip";
constexpr unsigned releases_per_page = 30;
constexpr wchar_t download_prefix[] = L"/Rift73/traiNNer-GUI/releases/download/";
constexpr wchar_t worker_flag[] = L"--trainner-update-worker";
constexpr wchar_t forwarded_flag[] = L"--trainner-update-forwarded";
constexpr wchar_t stage_flag[] = L"--stage-update";
constexpr wchar_t apply_flag[] = L"--apply-update";
constexpr const wchar_t* package_files[] = {
    exe_name, source_name, L"README-Windows.md", L"THIRD-PARTY-WEIGHTS.txt"
};
constexpr std::uint64_t max_zip_size = 2ull * 1024 * 1024 * 1024;
constexpr auto discovery_interval = 4h;
constexpr auto retry_interval = 10min;
constexpr DWORD stage_timeout = 30 * 60 * 1000;
constexpr DWORD apply_timeout = 90 * 1000;
constexpr DWORD busy = 75;

struct Version {
    std::array<std::uint32_t, 3> core{};
    bool stable = true;
    std::uint32_t preview = 0;

    bool operator<(const Version& other) const {
        return std::tie(core, stable, preview) < std::tie(other.core, other.stable, other.preview);
    }
};

template<typename Integer>
bool take_number(std::string_view& text, Integer& result) {
    if (text.empty() || (text.size() > 1 && text.front() == '0'
                         && text[1] >= '0' && text[1] <= '9'))
        return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{}) return false;
    text.remove_prefix(static_cast<size_t>(parsed.ptr - text.data()));
    return true;
}

bool take_prefix(std::string_view& text, std::string_view prefix) {
    if (text.substr(0, prefix.size()) != prefix) return false;
    text.remove_prefix(prefix.size());
    return true;
}

std::optional<Version> parse_version(std::string_view tag) {
    Version version;
    if (!take_prefix(tag, "v") || !take_number(tag, version.core[0])
        || !take_prefix(tag, ".") || !take_number(tag, version.core[1])
        || !take_prefix(tag, ".") || !take_number(tag, version.core[2]))
        return {};
    if (tag.empty()) return version;
    version.stable = false;
    if (!take_prefix(tag, "-preview.") || !take_number(tag, version.preview) || !tag.empty())
        return {};
    return version;
}

struct Release {
    std::string tag;
    Version version;
    std::uint64_t id;

    path folder(const path& root) const {
        return root / L"versions" / (tag + "-" + std::to_string(id));
    }

    std::string record() const {
        return tag + " " + std::to_string(id) + "\n";
    }
};

// GitHub keeps the tag when a ZIP is replaced. Only the superseded Preview 3
// assets qualify; an arbitrary equal-version receipt is still current.
constexpr std::uint64_t original_preview3_asset = 790855138;
constexpr std::uint64_t second_preview3_asset = 793527253;
template<class Number>
bool superseded_preview3_asset(Number id) {
    return id == original_preview3_asset || id == second_preview3_asset;
}

bool replaced_preview3(const Release& release) {
    return release.tag == "v0.1.0-preview.3" && superseded_preview3_asset(release.id);
}

struct Asset {
    Release release;
    std::uint64_t size;
    std::string digest;
};

struct UpdateBaseline {
    Version version;
    bool include_version;

    // A bare version represents an installed release and excludes the same tag.
    UpdateBaseline(const Version& version, bool include_version = false)
        : version(version), include_version(include_version) {}

    bool accepts(const Version& candidate) const {
        return include_version ? !(candidate < version) : version < candidate;
    }

    void exclude_seen(const Version& seen) {
        if (seen < version) return;
        version = seen;
        include_version = false;
    }
};

UpdateBaseline update_baseline(const Version& own, const std::optional<Release>& current) {
    // A manually opened newer package can adopt its own tag through the verified cache.
    if (!current) return {own, false};
    if (current->version < own) return {own, true};
    if (replaced_preview3(*current) && !(own < current->version)) return {own, true};
    return {current->version, false};
}

[[noreturn]] void win_error(const char* message) {
    throw std::runtime_error(std::string(message) + " (Windows " + std::to_string(GetLastError()) + ").");
}

bool plain_path(const path& file, bool directory = false) {
    const auto attributes = GetFileAttributesW(file.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT)
        && static_cast<bool>(attributes & FILE_ATTRIBUTE_DIRECTORY) == directory;
}

Handle open_file(const path& file, DWORD share = FILE_SHARE_READ | FILE_SHARE_DELETE) {
    const auto opened = CreateFileW(file.c_str(), GENERIC_READ, share, nullptr,
                                    OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (opened == INVALID_HANDLE_VALUE) return Handle{};
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(opened, &info)
        || (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))) {
        CloseHandle(opened);
        return Handle{};
    }
    return Handle(opened);
}

bool complete_bundle(const path& folder) {
    if (!plain_path(folder, true)) return false;
    return std::all_of(std::begin(package_files), std::end(package_files),
                       [&](const wchar_t* name) { return plain_path(folder / name); });
}

std::optional<Release> selection(const path& root, const wchar_t* name) {
    if (!plain_path(root, true) || !plain_path(root / L"versions", true)
        || !plain_path(root / name))
        return {};
    const auto file = open_file(root / name);
    if (!file) return {};
    char data[128];
    DWORD size{};
    if (!ReadFile(file.get(), data, sizeof(data), &size, nullptr) || !size || size == sizeof(data))
        return {};
    std::string_view record(data, size);
    if (record.back() != '\n') return {};
    record.remove_suffix(1);
    const auto separator = record.find(' ');
    if (separator == std::string_view::npos) return {};
    const auto tag = record.substr(0, separator);
    const auto version = parse_version(tag);
    record.remove_prefix(separator + 1);
    std::uint64_t id{};
    if (!version || !take_number(record, id) || !id || !record.empty()) return {};
    Release release{std::string(tag), *version, id};
    if (replaced_preview3(release) && std::wstring_view(name) != L"current") return {};
    if (!complete_bundle(release.folder(root))) return {};
    return release;
}

void ensure_directory(const path& directory) {
    std::filesystem::create_directories(directory);
    if (!plain_path(directory, true)) throw std::runtime_error("Invalid Windows update directory.");
}

bool recent(const path& marker, std::chrono::seconds duration) {
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(marker, error);
    if (error) return false;
    const auto age = decltype(stamp)::clock::now() - stamp;
    return age >= decltype(age)::zero() && age < duration;
}

path discovery_stamp(const path& root, const char* name, const UpdateBaseline& baseline) {
    // An older install's check must not delay discovery of the manually opened tag.
    if (baseline.include_version) {
        const auto revision = std::string_view(TRAINNER_GUI_RELEASE_TAG) == "v0.1.0-preview.3" ? "-r3" : "";
        return root / (std::string(name) + "-" + TRAINNER_GUI_RELEASE_TAG + revision);
    }
    return root / name;
}

bool discovery_due(const path& root, const UpdateBaseline& baseline) {
    return !recent(discovery_stamp(root, "checked", baseline), discovery_interval)
        && !recent(discovery_stamp(root, "check-attempt", baseline), retry_interval);
}

void log(const path& root, const std::string& message) {
    if (plain_path(root, true))
        std::ofstream(root / L"update.log", std::ios::binary | std::ios::trunc) << message << '\n';
}

std::wstring quote(const std::wstring& arg) {
    std::wstring quoted(1, L'"');
    size_t slashes = 0;
    for (const auto ch : arg) {
        if (ch == L'\\') {
            ++slashes;
        } else {
            quoted.append(ch == L'"' ? 2 * slashes + 1 : slashes, L'\\');
            quoted.push_back(ch);
            slashes = 0;
        }
    }
    quoted.append(2 * slashes, L'\\');
    return quoted + L'"';
}

Handle start_process(const path& exe, const std::vector<std::wstring>& args,
                     bool forwarded = false, HANDLE output = nullptr) {
    auto command = quote(exe.wstring());
    for (const auto& arg : args) command += L" " + quote(arg);

    if (forwarded) AttachConsole(ATTACH_PARENT_PROCESS);
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    const DWORD std_ids[] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
    std::array<Handle, 3> streams;
    HANDLE handles[3];
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto original = forwarded ? GetStdHandle(std_ids[i]) : (i == 1 ? output : nullptr);
        HANDLE copied{};
        if (original && original != INVALID_HANDLE_VALUE) {
            if (!DuplicateHandle(GetCurrentProcess(), original, GetCurrentProcess(), &copied,
                                 0, TRUE, DUPLICATE_SAME_ACCESS))
                win_error("Could not forward launcher input or output");
        } else {
            copied = CreateFileW(L"NUL", i == 0 ? GENERIC_READ : GENERIC_WRITE,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        }
        streams[i].reset(copied);
        if (!streams[i]) win_error("Could not set up launcher input or output");
        handles[i] = copied;
    }

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = handles[0];
    startup.StartupInfo.hStdOutput = handles[1];
    startup.StartupInfo.hStdError = handles[2];
    SIZE_T bytes{};
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    if (!bytes) win_error("Could not size launcher process attributes");
    std::vector<unsigned char> storage(bytes);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes))
        win_error("Could not set up launcher process attributes");
    const std::unique_ptr<std::remove_pointer_t<LPPROC_THREAD_ATTRIBUTE_LIST>,
                          decltype(&DeleteProcThreadAttributeList)> cleanup(attributes, DeleteProcThreadAttributeList);
    if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   handles, sizeof(handles), nullptr, nullptr))
        win_error("Could not limit inherited launcher handles");
    startup.lpAttributeList = attributes;
    PROCESS_INFORMATION child{};
    const DWORD flags = EXTENDED_STARTUPINFO_PRESENT | (forwarded ? 0 : CREATE_NO_WINDOW);
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE, flags, nullptr, nullptr,
                        &startup.StartupInfo, &child))
        win_error("Could not start the update process");
    const Handle thread(child.hThread);
    return Handle(child.hProcess);
}

DWORD wait_for_process(const Handle& child, DWORD timeout) {
    auto status = WaitForSingleObject(child.get(), timeout);
    if (status == WAIT_TIMEOUT) {
        // Report a result from a child that exited just before the deadline.
        if (TerminateProcess(child.get(), ERROR_TIMEOUT))
            status = WaitForSingleObject(child.get(), 5000);
        else
            status = WaitForSingleObject(child.get(), 0);
        if (status != WAIT_OBJECT_0) throw std::runtime_error("The update process did not stop in time.");
    }
    DWORD code{};
    if (status != WAIT_OBJECT_0 || !GetExitCodeProcess(child.get(), &code))
        win_error("Could not read the update process result");
    return code;
}

using Internet = std::unique_ptr<void, decltype(&WinHttpCloseHandle)>;

struct Response {
    Internet connection{nullptr, WinHttpCloseHandle};
    Internet request{nullptr, WinHttpCloseHandle};
    DWORD status{};
};

Response get(HINTERNET session, const wchar_t* host, const std::wstring& target,
             const wchar_t* headers = WINHTTP_NO_ADDITIONAL_HEADERS) {
    Response response;
    response.connection.reset(WinHttpConnect(session, host, INTERNET_DEFAULT_HTTPS_PORT, 0));
    if (!response.connection) win_error("Could not connect to GitHub");
    response.request.reset(WinHttpOpenRequest(response.connection.get(), L"GET", target.c_str(),
                                               nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                               WINHTTP_FLAG_SECURE));
    if (!response.request) win_error("Could not open the GitHub request");
    DWORD disable = WINHTTP_DISABLE_REDIRECTS | WINHTTP_DISABLE_COOKIES | WINHTTP_DISABLE_AUTHENTICATION;
    DWORD enable = WINHTTP_ENABLE_SSL_REVOCATION;
    auto request = response.request.get();
    if (!WinHttpSetOption(request, WINHTTP_OPTION_DISABLE_FEATURE, &disable, sizeof(disable))
        || !WinHttpSetOption(request, WINHTTP_OPTION_ENABLE_FEATURE, &enable, sizeof(enable)))
        win_error("Could not configure the GitHub request");
    if (!WinHttpSendRequest(request, headers, headers ? DWORD(-1) : 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
        || !WinHttpReceiveResponse(request, nullptr))
        win_error("Could not read the GitHub response");
    DWORD size = sizeof(response.status);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &response.status, &size, WINHTTP_NO_HEADER_INDEX))
        win_error("GitHub did not return an HTTP status");
    return response;
}

template<typename Consume>
std::uint64_t read_body(const Response& response, std::uint64_t limit, Consume consume) {
    if (response.status != 200)
        throw std::runtime_error("GitHub returned HTTP " + std::to_string(response.status) + ".");
    std::array<char, 65536> buffer;
    std::uint64_t total = 0;
    const auto started = GetTickCount64();
    for (;;) {
        DWORD count{};
        if (!WinHttpReadData(response.request.get(), buffer.data(), static_cast<DWORD>(buffer.size()), &count))
            win_error("Could not read GitHub response bytes");
        if (!count) return total;
        total += count;
        if (total > limit) throw std::runtime_error("GitHub response exceeded the expected size.");
        if (GetTickCount64() - started >= 15 * 60 * 1000)
            throw std::runtime_error("The GitHub response timed out.");
        consume(buffer.data(), count);
    }
}

struct ListedAsset {
    std::string tag;
    Version version;
    json::JsonObject asset;
};

void consider_releases(const json::JsonArray& releases, const UpdateBaseline& baseline,
                       std::optional<ListedAsset>& best) {
    for (const auto& entry : releases) {
        const auto release = entry.GetObject();
        if (release.GetNamedBoolean(L"draft", true)) continue;
        auto tag = utf8(std::wstring(release.GetNamedString(L"tag_name")));
        const auto version = parse_version(tag);
        if (!version || !baseline.accepts(*version) || (best && !(best->version < *version))) continue;
        if (baseline.version.stable && !version->stable) continue;
        if (version->stable && release.GetNamedBoolean(L"prerelease", true)) continue;
        for (const auto& item : release.GetNamedArray(L"assets")) {
            const auto asset = item.GetObject();
            if (asset.GetNamedString(L"name") == zip_name && asset.GetNamedString(L"state") == L"uploaded") {
                if (tag == "v0.1.0-preview.3" && superseded_preview3_asset(asset.GetNamedNumber(L"id"))) continue;
                best = ListedAsset{std::move(tag), *version, asset};
                break;
            }
        }
    }
}

Asset asset_metadata(const ListedAsset& item) {
    const auto size = item.asset.GetNamedNumber(L"size");
    const auto id = item.asset.GetNamedNumber(L"id");
    const auto digest = utf8(std::wstring(item.asset.GetNamedString(L"digest", L"")));
    if (!std::isfinite(size) || size <= 0 || size > max_zip_size || size != std::floor(size)
        || !std::isfinite(id) || id <= 0 || id >= 9007199254740992.0 || id != std::floor(id)
        || digest.size() != 71 || digest.compare(0, 7, "sha256:") != 0
        || digest.find_first_not_of("0123456789abcdef", 7) != std::string::npos)
        throw std::runtime_error("GitHub did not provide valid Windows asset size and SHA-256 metadata.");
    return {{item.tag, item.version, static_cast<std::uint64_t>(id)},
            static_cast<std::uint64_t>(size), digest.substr(7)};
}

std::optional<Asset> latest_asset(HINTERNET session, const UpdateBaseline& baseline) {
    std::optional<ListedAsset> best;
    const auto started = std::chrono::steady_clock::now();
    for (unsigned page = 1;; ++page) {
        if (std::chrono::steady_clock::now() - started > 2min)
            throw std::runtime_error("Checking the GitHub release list timed out.");
        const auto target = L"/repos/Rift73/traiNNer-GUI/releases?per_page="
            + std::to_wstring(releases_per_page) + L"&page=" + std::to_wstring(page);
        const auto response = get(session, L"api.github.com", target, L"Accept: application/vnd.github+json");
        std::string body;
        read_body(response, 8 * 1024 * 1024, [&](const char* data, DWORD count) { body.append(data, count); });
        const auto releases = json::JsonArray::Parse(wide(body));
        consider_releases(releases, baseline, best);
        if (releases.Size() < releases_per_page) break;
    }
    if (!best) return {};
    return asset_metadata(*best);
}

std::wstring redirect_location(const Response& response) {
    DWORD bytes{};
    auto request = response.request.get();
    WinHttpQueryHeaders(request, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX,
                        nullptr, &bytes, WINHTTP_NO_HEADER_INDEX);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes || bytes > 16384)
        throw std::runtime_error("GitHub did not return a usable asset redirect.");
    std::wstring location(bytes / sizeof(wchar_t), L'\0');
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX,
                             location.data(), &bytes, WINHTTP_NO_HEADER_INDEX))
        win_error("Could not read the GitHub asset redirect");
    location.resize(bytes / sizeof(wchar_t));
    return location;
}

Response download_response(HINTERNET session, const Release& release) {
    auto response = get(session, L"github.com", download_prefix + wide(release.tag) + L"/" + zip_name);
    if (response.status != 302) return response;
    const auto location = redirect_location(response);
    for (const auto host : {L"release-assets.githubusercontent.com", L"objects.githubusercontent.com"}) {
        const auto prefix = std::wstring(L"https://") + host + L"/";
        if (location.compare(0, prefix.size(), prefix) == 0)
            return get(session, host, location.substr(prefix.size() - 1));
    }
    throw std::runtime_error("GitHub redirected the Windows asset outside its release asset hosts.");
}

void download_zip(HINTERNET session, const Asset& asset, const path& zip) {
    std::filesystem::remove(zip);
    const auto response = download_response(session, asset.release);
    Handle file(CreateFileW(zip.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file) win_error("Could not open the partial Windows download");
    const auto size = read_body(response, asset.size, [&](const char* bytes, DWORD count) {
        DWORD written{};
        if (!WriteFile(file.get(), bytes, count, &written, nullptr) || written != count)
            win_error("Could not write the partial Windows download");
    });
    if (size != asset.size) throw std::runtime_error("The Windows download was incomplete.");
    if (!FlushFileBuffers(file.get())) win_error("Could not flush the Windows download");
}

void verify_launcher(const path& exe) {
    IMAGE_DOS_HEADER dos{};
    IMAGE_NT_HEADERS64 pe{};
    std::ifstream file(exe, std::ios::binary);
    file.read(reinterpret_cast<char*>(&dos), sizeof(dos));
    if (!file || dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < static_cast<LONG>(sizeof(dos)))
        throw std::runtime_error("The Windows bundle has an invalid launcher.");
    file.seekg(dos.e_lfanew);
    file.read(reinterpret_cast<char*>(&pe), sizeof(pe));
    if (!file || pe.Signature != IMAGE_NT_SIGNATURE || pe.FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64
        || pe.FileHeader.SizeOfOptionalHeader < sizeof(pe.OptionalHeader)
        || pe.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC
        || pe.OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_SECURITY)
        throw std::runtime_error("The Windows bundle does not contain an x64 launcher.");
    file.close();

    WINTRUST_FILE_INFO target{};
    target.cbStruct = sizeof(target);
    target.pcwszFilePath = exe.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_WHOLECHAIN;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &target;
    data.dwProvFlags = WTD_REVOCATION_CHECK_CHAIN_EXCLUDE_ROOT;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    const auto trust = WinVerifyTrust(reinterpret_cast<HWND>(INVALID_HANDLE_VALUE), &policy, &data);
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(reinterpret_cast<HWND>(INVALID_HANDLE_VALUE), &policy, &data);
    // The first-party ZIP digest authenticates packages built without Authenticode.
    // A present but invalid embedded signature must never be treated as unsigned.
    const auto& signature = pe.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_SECURITY];
    if (trust != ERROR_SUCCESS && (trust != TRUST_E_NOSIGNATURE || signature.Size || signature.VirtualAddress))
        throw std::runtime_error("The Windows launcher signature did not verify.");
}

path system_tar() {
    wchar_t directory[MAX_PATH];
    const auto length = GetSystemDirectoryW(directory, MAX_PATH);
    if (!length || length >= MAX_PATH) win_error("Could not locate Windows tar.exe");
    return path(directory) / L"tar.exe";
}

bool stage_zip(const path& root, const Asset& asset, const path& zip) {
    const auto destination = asset.release.folder(root);
    if (complete_bundle(destination)) return true;
    const auto archive = open_file(zip, FILE_SHARE_READ);
    LARGE_INTEGER size{};
    if (!archive || !GetFileSizeEx(archive.get(), &size)
        || static_cast<std::uint64_t>(size.QuadPart) != asset.size
        || host::sha256(archive.get()) != asset.digest)
        return false;

    auto staging = destination;
    staging += L".partial";
    ensure_directory(staging);
    const auto tar = system_tar();
    for (const auto name : package_files) {
        const auto target = staging / name;
        std::filesystem::remove(target);
        Handle file(CreateFileW(target.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                FILE_ATTRIBUTE_NORMAL, nullptr));
        if (!file) win_error("Could not open a staged Windows bundle member");
        // tar writes a named member to our file handle, never to an archive-supplied path.
        const auto child = start_process(tar, {L"-xOf", zip.wstring(), L"--", name}, false, file.get());
        if (wait_for_process(child, 60000) || !GetFileSizeEx(file.get(), &size) || size.QuadPart <= 0)
            throw std::runtime_error("The Windows ZIP is missing a usable " + utf8(name) + ".");
        if (!FlushFileBuffers(file.get())) win_error("Could not flush a staged Windows bundle member");
    }
    verify_launcher(staging / exe_name);
    if (!MoveFileExW(staging.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH))
        win_error("Could not publish the staged Windows bundle");
    return true;
}

Handle update_lock(const path& root) {
    return Handle(CreateFileW((root / L"update.lock").c_str(), GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                              FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
}

path staging_stamp(const path& root, const Release& release) {
    return root / ("stage-" + std::to_string(release.id));
}

void prepare(const path& root, const Release& release) {
    const auto stamp = staging_stamp(root, release);
    if (recent(stamp, retry_interval)) return;
    host::atomic_text(stamp, release.record());
    const auto child = start_process(release.folder(root) / exe_name, {stage_flag});
    const auto result = wait_for_process(child, stage_timeout);
    if (result != 0) {
        log(root, "Could not stage " + release.tag + " (exit " + std::to_string(result) + ").");
        return;
    }
    host::atomic_text(root / L"pending", release.record());
    log(root, "Prepared " + release.tag + " for activation on the next ordinary launch.");
}

bool needs_preparation(const std::optional<Release>& candidate, const std::optional<Release>& pending,
                       const UpdateBaseline& baseline) {
    return candidate && baseline.accepts(candidate->version) && (!pending || pending->version < candidate->version);
}

void background(const path& profile, const path& root, const Version& own_version) {
    ensure_directory(profile);
    ensure_directory(root);
    const auto lock = update_lock(root);
    if (!lock) return;
    ensure_directory(root / L"versions");

    const auto current = selection(root, L"current");
    const auto pending = selection(root, L"pending");
    const auto candidate = selection(root, L"candidate");
    const auto baseline = update_baseline(own_version, current);
    if (needs_preparation(candidate, pending, baseline)) prepare(root, *candidate);
    if (!discovery_due(root, baseline)) return;
    auto available = baseline;
    if (pending) available.exclude_seen(pending->version);
    if (candidate) available.exclude_seen(candidate->version);

    host::atomic_text(discovery_stamp(root, "check-attempt", baseline), "\n");
    winrt::init_apartment();
    const Internet session(WinHttpOpen(L"traiNNer-GUI Windows updater/1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0), WinHttpCloseHandle);
    if (!session || !WinHttpSetTimeouts(session.get(), 5000, 10000, 10000, 15000))
        win_error("Could not initialize GitHub release discovery");
    const auto asset = latest_asset(session.get(), available);
    const auto checked = discovery_stamp(root, "checked", baseline);
    if (!asset) {
        host::atomic_text(checked, "\n");
        return;
    }
    const auto zip = root / L"release.zip.partial";
    if (!stage_zip(root, *asset, zip)) {
        download_zip(session.get(), *asset, zip);
        if (!stage_zip(root, *asset, zip)) {
            std::filesystem::remove(zip);
            throw std::runtime_error("The downloaded Windows ZIP failed SHA-256 or size verification.");
        }
    }
    host::atomic_text(root / L"candidate", asset->release.record());
    host::atomic_text(checked, "\n");
    std::error_code ignored;
    std::filesystem::remove(zip, ignored);
    prepare(root, asset->release);
}

std::optional<Release> activate(const path& root, const Version& own_version,
                                std::optional<Release> current) {
    auto pending = selection(root, L"pending");
    auto newer = [&] {
        return pending && update_baseline(own_version, current).accepts(pending->version);
    };
    if (!newer()) return current;
    const auto lock = update_lock(root);
    if (!lock) return current;

    current = selection(root, L"current");
    pending = selection(root, L"pending");
    if (!newer()) return current;
    try {
        const auto child = start_process(pending->folder(root) / exe_name, {apply_flag});
        const auto result = wait_for_process(child, apply_timeout);
        if (result == busy) return current;
        if (result == ERROR_NOT_READY) {
            // The prepared Linux slot no longer matches the selected profile.
            // Keep the verified files and let the worker prepare them again.
            std::error_code ignored;
            std::filesystem::remove(root / L"pending", ignored);
            std::filesystem::remove(staging_stamp(root, *pending), ignored);
        }
        if (result != 0) {
            log(root, "Could not apply " + pending->tag + " (exit " + std::to_string(result) + ").");
            return current;
        }
    } catch (const std::exception& error) {
        log(root, error.what());
        return current;
    }

    // The paired backend has committed. Keep pending until current is durable.
    try {
        host::atomic_text(root / L"current", pending->record());
        std::error_code ignored;
        std::filesystem::remove(root / L"pending", ignored);
    } catch (const std::exception& error) {
        log(root, error.what());
    }
    return pending;
}

bool local_profile(const path& profile) {
    const auto name = profile.root_name().native();
    return profile.is_absolute() && name.size() == 2 && name[1] == L':'
        && GetDriveTypeW(profile.root_path().c_str()) == DRIVE_FIXED;
}

void schedule(const path& root, const UpdateBaseline& baseline, const Release* active) {
    const auto candidate = selection(root, L"candidate");
    const auto pending = selection(root, L"pending");
    if (!discovery_due(root, baseline)
        && (!needs_preparation(candidate, pending, baseline)
            || recent(staging_stamp(root, *candidate), retry_interval)))
        return;
    if (active) {
        // The shortcut can outlive several releases. Use the installed updater.
        start_process(active->folder(root) / exe_name, {worker_flag});
        return;
    }
    std::wstring exe(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
    if (!length || length >= exe.size()) win_error("Could not locate the current launcher");
    exe.resize(length);
    start_process(exe, {worker_flag});
}

} // namespace

void check_managed_source(const path& profile_folder, const std::string& payload) {
    const auto root = profile_folder / L"updates";
    const auto current = selection(root, L"current");
    if (!current) return;
    const auto own_version = parse_version(TRAINNER_GUI_RELEASE_TAG);
    const auto source = open_file(current->folder(root) / source_name, FILE_SHARE_READ);
    if (!own_version || !update_baseline(*own_version, current).accepts(*own_version)
        || !source || host::sha256(source.get()) != payload)
        throw host::Exit(busy, "The installed Windows release does not match the selected managed backend "
                               "or is already as new as this update.");
}

std::optional<int> dispatch(const path& exe_folder, const path& profile_folder,
                            std::vector<std::wstring>& args, bool packaged) {
    const bool forwarded_call = !args.empty() && args.front() == forwarded_flag;
    const bool worker = args.size() == 1 && args.front() == worker_flag;
    const bool backend = args.size() == 1 && (args.front() == stage_flag || args.front() == apply_flag);
    const bool internal = forwarded_call || worker || backend;
    if (!packaged && !internal) return {};
    if (host::elevated()) {
        if (internal) return ERROR_ACCESS_DENIED;
        return {};
    }
    if (forwarded_call) {
        args.erase(args.begin());
        AttachConsole(ATTACH_PARENT_PROCESS);
        return {};
    }
    if (backend) return {};
    const auto own_version = parse_version(TRAINNER_GUI_RELEASE_TAG);
    if (!packaged || !own_version || !local_profile(profile_folder)) {
        if (worker) return 1;
        return {};
    }

    const auto root = profile_folder / L"updates";
    if (worker) {
        try {
            background(profile_folder, root, *own_version);
            return 0;
        } catch (const winrt::hresult_error& error) {
            log(root, utf8(std::wstring(error.message())));
        } catch (const std::exception& error) {
            log(root, error.what());
        }
        return 1;
    }

    const bool ordinary = std::none_of(args.begin(), args.end(), [](const auto& arg) {
        return arg == L"--configure" || arg == L"--help" || arg == L"-h" || arg == L"--phone-status";
    });
    auto current = selection(root, L"current");
    if (ordinary)
        current = activate(root, *own_version, std::move(current));
    // An equal tag can now be the committed result of adopting a downloaded package.
    const auto* active = current && !(current->version < *own_version) ? &*current : nullptr;
    if (ordinary) {
        try {
            schedule(root, update_baseline(*own_version, current),
                     active && !replaced_preview3(*active) ? active : nullptr);
        } catch (const std::exception& error) {
            log(root, error.what());
        }
    }
    if (!active) return {};
    // A user opening the replacement to finish setup must get its repair UI,
    // even if a previous attempt left an older same-tag updater receipt.
    if (replaced_preview3(*active) &&
        (std::find(args.begin(), args.end(), L"--configure") != args.end()
         || !std::filesystem::exists(profile_folder / L"desktop.conf")))
        return {};
    const auto folder = active->folder(root);
    if (host::same_text(exe_folder.lexically_normal().native(), folder.lexically_normal().native()))
        return {};
    auto forwarded = args;
    forwarded.insert(forwarded.begin(), forwarded_flag);
    const auto child = [&] {
        try {
            return start_process(folder / exe_name, forwarded, true);
        } catch (const std::exception& error) {
            log(root, error.what());
            return Handle{};
        }
    }();
    if (!child) return {};
    return static_cast<int>(wait_for_process(child, INFINITE));
}

} // namespace updates
