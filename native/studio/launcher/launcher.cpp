// Select or set up the local Linux trainer and open its Studio service.
#include "windows_host.hpp"
#include "updates.hpp"

#include <shellapi.h>

#include <algorithm>
#include <cwchar>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>

#ifndef TRAINNER_GUI_PAYLOAD_SHA256
#define TRAINNER_GUI_PAYLOAD_SHA256 ""
#endif

namespace {

using namespace host;

constexpr wchar_t managed_distro[] = L"traiNNer-GUI";
constexpr wchar_t managed_root[] = L"/opt/trainner-gui";
constexpr wchar_t managed_releases[] = L"/opt/trainner-gui-releases/";
constexpr wchar_t managed_user[] = L"trainer";
constexpr wchar_t archive_name[] = L"trainner-gui-source.tar.gz";
constexpr char payload_hash[] = TRAINNER_GUI_PAYLOAD_SHA256;
constexpr wchar_t registry_key[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Lxss";
constexpr char usage[] =
    "traiNNer-GUI.exe [--url | --phone-status] [--state-dir LinuxPath]\n"
    "traiNNer-GUI.exe --configure [--managed [--cpu-only] | --distro Name --root LinuxPath]"
    " [--project-dir WindowsPath] [--non-interactive]\n";

struct Options {
    bool url = false;
    bool apply_update = false;
    bool stage_update = false;
    bool phone_status = false;
    bool cpu_only = false;
    bool configure = false;
    bool non_interactive = false;
    bool managed = false;
    bool help = false;
    wstring state, distro, root;
    fs::path project;
};

struct Settings {
    wstring distro, root;
    fs::path project;
    string project_id, payload;
    bool managed = false;
};

struct Pending {
    fs::path project;
    string project_id, reboot_after;
    bool cpu_only = false;
};

struct Profile {
    explicit Profile(fs::path folder) : folder(std::move(folder)) {}
    fs::path config() const { return folder / L"desktop.conf"; }
    fs::path last_managed() const { return folder / L"previous-managed.conf"; }
    fs::path pending() const { return folder / L"setup.conf"; }
    fs::path log() const { return folder / L"setup.log"; }
    fs::path distro_location() const { return folder / L"wsl"; }
    fs::path folder;
};

Options parse_options(const std::vector<wstring>& args) {
    Options options;
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        auto value = [&]() -> wstring {
            if (i + 1 >= args.size() || args[i + 1].empty() || args[i + 1].rfind(L"--", 0) == 0)
                throw std::runtime_error(usage);
            return args[++i];
        };
        if (arg == L"--apply-update" && args.size() == 1) options.apply_update = true;
        else if (arg == L"--stage-update" && args.size() == 1) options.stage_update = true;
        else if (arg == L"--url") options.url = true;
        else if (arg == L"--phone-status") options.phone_status = true;
        else if (arg == L"--cpu-only") options.cpu_only = true;
        else if (arg == L"--configure") options.configure = true;
        else if (arg == L"--non-interactive") options.non_interactive = true;
        else if (arg == L"--managed") options.managed = true;
        else if (arg == L"--help" || arg == L"-h") options.help = true;
        else if (arg == L"--state-dir") options.state = value();
        else if (arg == L"--distro") options.distro = value();
        else if (arg == L"--root") options.root = value();
        else if (arg == L"--project-dir") options.project = value();
        else throw std::runtime_error(usage);
    }
    const bool existing = !options.distro.empty() || !options.root.empty();
    if ((options.url && options.phone_status) || (options.cpu_only && !options.managed)
        || (!options.configure && (options.managed || existing || !options.project.empty() || options.non_interactive))
        || (options.configure && (options.url || options.phone_status || !options.state.empty()))
        || (existing && (options.managed || options.distro.empty() || options.root.empty())))
        throw std::runtime_error(usage);
    if (options.configure && options.non_interactive
        && ((!options.managed && !existing) || options.project.empty()))
        throw std::runtime_error("Noninteractive setup needs --project-dir and either --managed"
                                 " or both --distro and --root.");
    return options;
}

void check_checkout(const Settings& value) {
    if (value.distro.empty() || value.distro.front() == L'-'
        || value.distro.find_first_of(L"/\\\"\r\n") != wstring::npos
        || value.root.empty() || value.root.front() != L'/'
        || value.root.find_first_of(L"\"\r\n") != wstring::npos)
        throw std::runtime_error("Provide a WSL distribution name and an absolute Linux checkout path.");
}

bool is_digest(const string& hash) {
    return hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == string::npos;
}

void check_settings(const Settings& value) {
    check_checkout(value);
    if (value.project.empty() != value.project_id.empty()
        || (!value.project.empty() && (!value.project.is_absolute() || !valid_project_id(value.project_id)))
        || (value.managed && (value.distro != managed_distro || value.project.empty() || !is_digest(value.payload)
                              || (value.root != managed_root && value.root != managed_releases + wide(value.payload)))))
        throw std::runtime_error("Invalid desktop settings. Run with --configure.");
}

std::vector<string> fields(const string& text, size_t count) {
    std::istringstream input(text);
    std::vector<string> result;
    for (string line; std::getline(input, line);) result.push_back(line);
    if (result.size() != count || text.find('\0') != string::npos)
        throw std::runtime_error("Invalid desktop settings. Run with --configure.");
    return result;
}

std::optional<Settings> read_settings(const fs::path& file) {
    if (!fs::exists(file)) return {};
    const auto values = fields(read_text(file), 7);
    if (values[0] != "traiNNer-GUI 1" || (values[1] != "managed" && values[1] != "existing"))
        throw std::runtime_error("Unsupported desktop settings. Run with --configure.");
    Settings result{wide(values[2]), wide(values[3]), wide(values[4]), values[5], values[6],
                    values[1] == "managed"};
    check_settings(result);
    return result;
}

std::optional<Settings> load_settings(const Profile& profile) {
    return read_settings(profile.config());
}

void save_settings(const Profile& profile, const Settings& value) {
    check_settings(value);
    atomic_text(profile.config(), string("traiNNer-GUI 1\n") + (value.managed ? "managed\n" : "existing\n")
                + utf8(value.distro) + "\n" + utf8(value.root) + "\n" + utf8(value.project.wstring())
                + "\n" + value.project_id + "\n" + value.payload + "\n");
}

Pending load_pending(const Profile& profile) {
    if (!fs::exists(profile.pending())) return {};
    const auto values = fields(read_text(profile.pending()), 4);
    if (!valid_project_id(values[1]) || (values[3] != "cpu" && values[3] != "cuda"))
        throw std::runtime_error("Invalid pending setup settings.");
    return {wide(values[0]), values[1], values[2], values[3] == "cpu"};
}

void save_pending(const Profile& profile, const Pending& value) {
    atomic_text(profile.pending(), utf8(value.project.wstring()) + "\n" + value.project_id
                + "\n" + value.reboot_after + "\n" + (value.cpu_only ? "cpu\n" : "cuda\n"));
}

wstring registry_string(const wchar_t* key, const wchar_t* name) {
    DWORD size{};
    const auto status = RegGetValueW(HKEY_CURRENT_USER, key, name, RRF_RT_REG_SZ, nullptr, nullptr, &size);
    if (status == ERROR_FILE_NOT_FOUND) return {};
    if (status != ERROR_SUCCESS)
        throw std::runtime_error("Could not read the current user's WSL registration.");
    wstring result(size / sizeof(wchar_t), 0);
    if (RegGetValueW(HKEY_CURRENT_USER, key, name, RRF_RT_REG_SZ,
                     nullptr, result.data(), &size) != ERROR_SUCCESS)
        throw std::runtime_error("Could not read the current user's WSL registration.");
    if (!result.empty()) result.resize(std::wcslen(result.c_str()));
    return result;
}

wstring default_distro() {
    const auto id = registry_string(registry_key, L"DefaultDistribution");
    if (id.empty()) return {};
    const auto key = wstring(registry_key) + L"\\" + id;
    return registry_string(key.c_str(), L"DistributionName");
}

struct Registration {
    fs::path location;
    DWORD version;
};

std::optional<Registration> registered_distro(const wstring& name) {
    HKEY raw{};
    const auto status = RegOpenKeyExW(HKEY_CURRENT_USER, registry_key, 0, KEY_READ, &raw);
    if (status == ERROR_FILE_NOT_FOUND) return {};
    if (status != ERROR_SUCCESS) throw std::runtime_error("Could not inspect WSL registrations.");
    std::unique_ptr<std::remove_pointer_t<HKEY>, decltype(&RegCloseKey)> key(raw, RegCloseKey);
    for (DWORD index = 0;; ++index) {
        wchar_t id[256]{};
        DWORD count = 256;
        const auto next = RegEnumKeyExW(key.get(), index, id, &count, nullptr, nullptr, nullptr, nullptr);
        if (next == ERROR_NO_MORE_ITEMS) return {};
        if (next != ERROR_SUCCESS) throw std::runtime_error("Could not inspect WSL registrations.");
        const auto subkey = wstring(registry_key) + L"\\" + id;
        if (!same_text(registry_string(subkey.c_str(), L"DistributionName"), name)) continue;
        DWORD version{}, bytes = sizeof(version);
        if (RegGetValueW(HKEY_CURRENT_USER, subkey.c_str(), L"Version", RRF_RT_REG_DWORD, nullptr,
                         &version, &bytes) != ERROR_SUCCESS)
            throw std::runtime_error("Could not verify the managed WSL version.");
        auto location = registry_string(subkey.c_str(), L"BasePath");
        if (location.rfind(L"\\\\?\\", 0) == 0) location.erase(0, 4);
        return Registration{location, version};
    }
}

bool managed_registered(const Profile& profile) {
    const auto registration = registered_distro(managed_distro);
    if (!registration) return false;
    if (registration->version != 2
        || !same_text(registration->location.lexically_normal().wstring(),
                      profile.distro_location().lexically_normal().wstring()))
        throw std::runtime_error("A different distribution already uses the name traiNNer-GUI."
                                 " Choose an existing checkout or resolve the name conflict.");
    return true;
}

std::optional<Settings> unc_checkout(const fs::path& folder) {
    auto path = folder.wstring();
    std::replace(path.begin(), path.end(), L'\\', L'/');
    if (path.rfind(L"//?/UNC/", 0) == 0) path = L"//" + path.substr(8);
    for (const wstring prefix : {L"//wsl.localhost/", L"//wsl$/"}) {
        if (path.size() < prefix.size() || !same_text(path.substr(0, prefix.size()), prefix)) continue;
        const size_t end = path.find(L'/', prefix.size());
        if (end != wstring::npos) {
            Settings checkout;
            checkout.distro = path.substr(prefix.size(), end - prefix.size());
            checkout.root = path.substr(end);
            return checkout;
        }
    }
    return {};
}

std::vector<wstring> linux_command(const Settings& value, bool in_root, const wchar_t* user = nullptr) {
    check_checkout(value);
    std::vector<wstring> arguments{L"--distribution", value.distro, L"--cd",
                                    in_root ? value.root : L"/"};
    if (user || value.managed)
        arguments.insert(arguments.end(), {L"--user", user ? user : managed_user});
    arguments.emplace_back(L"--exec");
    return arguments;
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

wstring to_linux(const Settings& value, const fs::path& path) {
    auto arguments = linux_command(value, false);
    arguments.insert(arguments.end(), {L"/usr/bin/wslpath", L"-a", L"-u", path.wstring()});
    const auto result = wsl(arguments);
    const auto converted = trim_lines(result.out);
    if (result.code || converted.empty() || converted.front() != '/'
        || converted.find_first_of("\r\n") != string::npos)
        throw std::runtime_error("WSL could not open the selected Windows folder. "
                                 + trim_lines(result.err));
    return wide(converted);
}

std::optional<Settings> legacy_settings(const fs::path& folder, bool env_only) {
    Settings selected;
    selected.distro = env(L"TRAINNER_GUI_DISTRO");
    selected.root = env(L"TRAINNER_GUI_ROOT");
    const bool override = !selected.distro.empty() || !selected.root.empty();
    if (env_only != override) return {};
    const auto unc = unc_checkout(folder);
    if (unc) {
        if (selected.distro.empty()) selected.distro = unc->distro;
        if (selected.root.empty()) selected.root = unc->root;
    } else {
        if (!override && !fs::is_regular_file(folder / L"trainner-gui")) return {};
        if (selected.distro.empty()) selected.distro = default_distro();
        if (selected.root.empty() && !selected.distro.empty()) {
            Settings context = selected;
            context.root = L"/";
            selected.root = to_linux(context, folder);
        }
    }
    if (selected.distro.empty() && !override) return {};
    check_checkout(selected);
    return selected;
}

void log_result(const Profile& profile, const string& stage, const ProcessResult& result) {
    std::ofstream log(profile.log(), std::ios::binary | std::ios::app);
    log << '[' << stage << "] exit " << result.code << '\n'
        << trim_lines(result.out) << '\n' << trim_lines(result.err) << '\n';
    log.close();
    if (!log) throw std::runtime_error("Could not write the desktop setup log.");
}

void log_checked(const Profile& profile, const string& stage, const ProcessResult& result) {
    log_result(profile, stage, result);
    if (result.code) throw std::runtime_error(stage + " failed. See " + utf8(profile.log().wstring()) + ".");
}

bool need_windows_setup() {
    if (!service_present(L"WslService") || !service_present(L"vmcompute")
        || !service_present(L"HNS") || !virtual_machine_platform_enabled())
        return true;
    // Match WSL's prerequisite checks so the named install stays unelevated.
    const auto build = windows_build_number();
    if ((build < 22000 && !service_present(L"LxssManager"))
        || (build >= 22621 && !service_present(L"vfpext")))
        return true;
    const auto status = wsl({L"--status"});
    if (!status.code) return false;
    const auto message = trim_lines(status.err) + '\n' + trim_lines(status.out);
    if (message.find("WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED") != string::npos
        || message.find("0x8007019e") != string::npos)
        return true;
    throw std::runtime_error("Windows Subsystem for Linux is unavailable.\n" + message);
}

void register_managed(const Profile& profile, SetupDialogs& dialogs) {
    const auto help = wsl({L"--help"});
    const auto supported = trim_lines(help.out) + trim_lines(help.err);
    if (supported.find("--name") == string::npos || supported.find("--location") == string::npos
        || supported.find("--set-default-user") == string::npos)
        throw std::runtime_error("Update Windows Subsystem for Linux to a version that supports"
                                 " named WSL 2 installs, then run setup again.");
    dialogs.progress(L"Downloading and installing Ubuntu 24.04.");
    const auto result = wsl({L"--install", L"Ubuntu-24.04", L"--name", managed_distro,
                             L"--version", L"2", L"--no-launch", L"--location",
                             profile.distro_location().wstring()}, INFINITE);
    log_checked(profile, "Ubuntu 24.04 registration", result);
    if (!managed_registered(profile))
        throw std::runtime_error("WSL did not register Ubuntu. See " + utf8(profile.log().wstring()) + ".");
}

std::vector<wstring> studio_command(const Settings& value) {
    auto arguments = linux_command(value, true);
    if (!value.project.empty()) {
        const auto project = local_ntfs_folder(value.project);
        if (project_id(project, false) != value.project_id)
            throw std::runtime_error("The workspace marker has changed. Reconnect the selected drive"
                                     " or choose a workspace with --configure.");
        arguments.insert(arguments.end(), {L"/usr/bin/env",
            L"TRAINNER_GUI_PROJECT_DIR=" + to_linux(value, project),
            L"TRAINNER_GUI_PROJECT_ID=" + wide(value.project_id)});
    }
    arguments.emplace_back(L"./trainner-gui");
    return arguments;
}

void verify_project(const Profile& profile, const Settings& value) {
    auto arguments = studio_command(value);
    arguments.emplace_back(L"project-info");
    const auto result = wsl(arguments);
    log_result(profile, "Studio workspace check", result);
    if (result.code || trim_lines(result.out) != value.project_id) {
        auto detail = trim_lines(result.err);
        if (detail.size() > 600) detail.resize(600);
        throw std::runtime_error("Studio could not verify the selected workspace. Use an updated"
                                 " traiNNer-GUI checkout or set up dedicated Ubuntu.\n" + detail
                                 + "\nSee " + utf8(profile.log().wstring()) + ".");
    }
}

void open_payload(const fs::path& folder, const string& expected_hash, Handle& source) {
    const auto archive = folder / archive_name;
    source.reset(CreateFileW(archive.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!source || sha256(source.get()) != expected_hash)
        throw std::runtime_error("The source archive is missing or belongs to a different release."
                                 " Extract the entire Windows ZIP and open the launcher from that folder.");
}

ProcessResult provision_payload(const Profile& profile, const Settings& selected, const Handle& source,
                                const std::vector<wstring>& extra, const string& stage) {
    auto arguments = linux_command(selected, false, L"root");
    arguments.insert(arguments.end(), {L"/bin/bash", L"-ec",
        LR"(set -o pipefail
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
tar -xzf - --no-same-owner --no-same-permissions -C "$staging"
chmod +x "$staging/trainner-gui" "$staging/install.sh"
exec 0</dev/null
/bin/bash "$staging/scripts/provision-wsl.sh" --user trainer "$@")", L"trainner-gui"});
    if (!SetFilePointerEx(source.get(), LARGE_INTEGER{}, nullptr, FILE_BEGIN))
        throw std::runtime_error("Could not rewind the verified Linux payload.");
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    const auto result = wsl(arguments, INFINITE, source.get());
    log_result(profile, stage, result);
    return result;
}

Settings stage_managed_upgrade(const Profile& profile, const fs::path& folder, const Settings& current) {
    if (!managed_registered(profile))
        throw std::runtime_error("The dedicated managed distribution is unavailable.");
    Settings candidate = current;
    candidate.payload = payload_hash;
    candidate.root = managed_releases + wide(candidate.payload);
    check_settings(candidate);
    Handle source;
    open_payload(folder, candidate.payload, source);
    std::vector<wstring> args{L"--upgrade-from", current.root, L"--release", wide(candidate.payload)};
    args.emplace_back(L"--probe");
    const auto idle = provision_payload(profile, current, source, args, "Check update readiness");
    if (idle.code)
        throw Exit(static_cast<int>(idle.code), "The managed update is pending. " + trim_lines(idle.err));
    args.pop_back();
    const auto result = provision_payload(profile, current, source, args, "Stage managed update");
    if (result.code)
        throw Exit(static_cast<int>(result.code), "Could not prepare the managed update. See "
                   + utf8(profile.log().wstring()) + ".");
    verify_project(profile, candidate);
    return candidate;
}

void prepare_managed(const Profile& profile, const fs::path& folder, Settings& selected,
                     Pending pending, SetupDialogs& dialogs, const Options& options) {
    if (selected.payload.empty())
        throw std::runtime_error("Use the Windows release ZIP for dedicated Ubuntu setup.");
    Handle source;
    open_payload(folder, selected.payload, source);
    if (!pending.reboot_after.empty() && pending.reboot_after == boot_time())
        throw Exit(ERROR_SUCCESS_REBOOT_REQUIRED, "Restart Windows and open traiNNer-GUI.exe again.");
    const bool windows_setup = need_windows_setup();
    if (windows_setup) {
        if (options.non_interactive)
            throw std::runtime_error("Windows Subsystem for Linux needs setup. Run --configure"
                                     " interactively to approve it.");
        if (!dialogs.confirm(L"Install Windows Subsystem for Linux?\n\n"
                             L"Windows will ask for administrator approval. After setup, restart"
                             L" Windows and open traiNNer-GUI.exe again."))
            throw Exit(ERROR_CANCELLED, "Windows setup was cancelled.");
    }
    const bool resume_project = !pending.project.empty()
        && same_text(pending.project.wstring(), selected.project.wstring());
    selected.project_id = project_id(selected.project, !resume_project);
    if (resume_project && pending.project_id != selected.project_id)
        throw std::runtime_error("The workspace marker changed since setup started."
                                 " Reconnect the selected drive or select a different workspace.");
    pending = {selected.project, selected.project_id, {},
               options.cpu_only || (resume_project && pending.cpu_only)};
    if (windows_setup) {
        pending.reboot_after = boot_time();
        save_pending(profile, pending);
        dialogs.progress(L"Installing Windows Subsystem for Linux.");
        try {
            install_wsl();
        } catch (const std::exception& error) {
            pending.reboot_after.clear();
            save_pending(profile, pending);
            log_result(profile, "Windows WSL setup", {1, {}, error.what()});
            throw;
        }
        log_result(profile, "Windows WSL setup requested; restart required", {});
        throw Exit(ERROR_SUCCESS_REBOOT_REQUIRED, "Restart Windows and open traiNNer-GUI.exe again.");
    }
    save_pending(profile, pending);
    if (!managed_registered(profile)) register_managed(profile, dialogs);
    dialogs.progress(L"Preparing the Linux trainer. This can take several minutes.");
    auto provision = [&](bool cpu_only) {
        std::vector<wstring> switches;
        if (selected.root != managed_root) switches = {L"--release", wide(selected.payload)};
        if (cpu_only) switches.emplace_back(L"--cpu-only");
        return provision_payload(profile, selected, source, switches,
                                 cpu_only ? "CPU-only runtime setup" : "CUDA runtime setup");
    };
    auto result = provision(pending.cpu_only);
    if (result.code == 4 && !pending.cpu_only && !options.non_interactive) {
        if (dialogs.confirm(L"CUDA training is unavailable. Set up the CPU-only workbench?"
                            L"\n\nThe workbench can open projects and run CPU recipes."))
        {
            pending.cpu_only = true;
            save_pending(profile, pending);
            dialogs.progress(L"Preparing the CPU-only workbench.");
            result = provision(true);
        }
    }
    if (result.code) {
        auto detail = trim_lines(result.err);
        const auto last = detail.rfind("traiNNer-GUI setup:");
        if (last != string::npos) detail = detail.substr(last);
        if (detail.size() > 750) detail = detail.substr(detail.size() - 750);
        throw Exit(static_cast<int>(result.code), "Linux trainer setup failed.\n" + detail
                   + "\nSee " + utf8(profile.log().wstring()) + ". Rerun setup from the same ZIP.");
    }
    log_checked(profile, "Linux login user",
                wsl({L"--manage", managed_distro, L"--set-default-user", managed_user}));
}

void lock_settings(const Profile& profile, Handle& lock, DWORD wait_ms = 0) {
    const auto deadline = GetTickCount64() + wait_ms;
    do {
        lock.reset(CreateFileW((profile.folder / L"setup.lock").c_str(), GENERIC_WRITE, 0, nullptr,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
        if (lock) return;
        if (GetLastError() != ERROR_SHARING_VIOLATION)
            throw std::runtime_error("Could not lock the desktop settings.");
        if (GetTickCount64() >= deadline)
            throw Exit(75, "Desktop setup or update is already running.");
        Sleep(50);
    } while (true);
}

void activate_managed(const Profile& profile, const Settings& current, const Settings& candidate) {
    auto gate = linux_command(candidate, true);
    gate.insert(gate.end(), {L"/usr/bin/python3", L"-I", candidate.root + L"/studio/update_gate.py",
                             L"activate", L"--old-root", current.root});
    // Preview 1 did not register alternate state bases. Include the one the
    // original shortcut selected even when its previous service is closed.
    const auto state = env(L"TRAINNER_GUI_WINDOWS_UPDATE_STATE");
    if (!state.empty()) {
        const auto base = state.front() == L'/' || state.front() == L'~' ? state : current.root + L"/" + state;
        gate.insert(gate.end(), {L"--state-dir", base});
    }
    bool committed = false;
    try {
        const auto result = wsl(gate, 60000, nullptr, "TRAINNER_GUI_UPDATE_GATE_READY", [&]() {
            save_settings(profile, candidate);
            committed = true;
        });
        log_result(profile, "Activate managed update", result);
        if (result.code)
            throw Exit(static_cast<int>(result.code), "The managed update is pending. " + trim_lines(result.err));
    } catch (const std::exception& error) {
        if (!committed) throw;
        // The backend pointer is already durable. Reporting failure here would
        // make the Windows updater launch an older executable against it.
        OutputDebugStringA(error.what());
    }
}

void apply_update(const Profile& profile, const fs::path& folder, bool stage_only, bool migrate_original = false) {
    if (elevated() || !is_digest(payload_hash))
        throw std::runtime_error("Apply updates from a packaged launcher as your ordinary Windows user.");
    if (fs::exists(profile.pending())) throw Exit(75, "Initial setup is in progress.");
    // Take the config snapshot with the same lock used by setup and activation.
    Handle snapshot_lock;
    fs::create_directories(profile.folder);
    lock_settings(profile, snapshot_lock);
    const auto selected = load_settings(profile);
    const auto expected = selected ? read_text(profile.config()) : string{};
    // Applying the same payload again can finish a previously interrupted Windows commit.
    if (selected && selected->managed && selected->root != managed_root && selected->payload != payload_hash)
        updates::check_managed_source(profile.folder, selected->payload);
    snapshot_lock.reset();
    if (!selected) return;  // No workspace has been configured yet.
    if (!selected->managed) {
        if (!selected->project.empty()) verify_project(profile, *selected);
        return;
    }
    if (!managed_registered(profile)) throw std::runtime_error("The dedicated Ubuntu installation is unavailable.");
    if (selected->payload == payload_hash) return;
    Settings candidate = *selected;
    candidate.payload = payload_hash;
    candidate.root = managed_releases + wide(candidate.payload);
    check_settings(candidate);

    // A published ready marker lives in the root-owned side-by-side installation.
    // Recheck the original account and runtime mode in case the user reconfigured it.
    auto ready = linux_command(*selected, false);
    ready.insert(ready.end(), {L"/bin/bash", L"-c",
        LR"(read -r _ account uid mode < "$1/.managed-ready" &&
            read -r _ other other_uid other_mode < "$2/.managed-ready" &&
            [[ $account == "$other" && $uid == "$other_uid" && $mode == "$other_mode" ]])",
        L"trainner-gui", selected->root, candidate.root});
    if (wsl(ready).code) {
        if (!stage_only && !migrate_original)
            throw Exit(ERROR_NOT_READY, "The update needs to be prepared for this workspace.");
        candidate = stage_managed_upgrade(profile, folder, *selected);
    } else {
        verify_project(profile, candidate);
    }
    if (stage_only) return;

    Handle lock;
    lock_settings(profile, lock);
    if (fs::exists(profile.pending()) || read_text(profile.config()) != expected)
        throw Exit(75, "The selected workspace changed while preparing the update.");
    activate_managed(profile, *selected, candidate);
}

Settings setup(const Options& options, const Profile& profile, const fs::path& folder, bool interactive) {
    if (elevated())
        throw std::runtime_error("Open setup as your normal Windows user. Windows will ask for"
                                 " administrator approval if WSL needs to be installed.");

    fs::create_directories(profile.folder);
    Handle lock;
    lock_settings(profile, lock);
    SetupDialogs dialogs(interactive);
    const auto pending = load_pending(profile);
    const bool resume = !options.configure && !pending.project.empty();
    const bool managed = options.managed || (options.distro.empty() && (resume || dialogs.choose_managed()));
    Settings selected;
    if (managed) {
        selected.distro = managed_distro;
        selected.root = managed_root;
        selected.payload = payload_hash;
        selected.managed = true;
    } else if (!options.distro.empty()) {
        selected.distro = options.distro;
        selected.root = options.root;
    } else {
        const auto initial = unc_checkout(folder) ? folder : fs::path(L"\\\\wsl.localhost\\");
        const auto checkout = unc_checkout(dialogs.folder(L"Choose the traiNNer-GUI checkout in WSL", initial));
        if (!checkout) throw std::runtime_error("Choose a folder under \\\\wsl.localhost or \\\\wsl$.");
        selected = *checkout;
    }
    check_checkout(selected);
    auto project = options.project;
    if (project.empty()) {
        if (managed && !pending.project.empty()) project = pending.project;
        else project = dialogs.folder(L"Choose a local NTFS workspace folder");
    }
    selected.project = local_ntfs_folder(project);
    std::optional<Settings> previous_release;

    // Explicit configuration also repairs an unreadable previous selection.
    auto recover_selection = [&](const fs::path& file) -> std::optional<Settings> {
        try {
            return read_settings(file);
        } catch (const std::runtime_error& error) {
            log_result(profile, "Recover desktop settings during setup", {1, {}, error.what()});
            return {};
        }
    };
    if (managed) {
        if (is_digest(selected.payload) && managed_registered(profile)) {
            auto current = recover_selection(profile.config());
            if (!current || !current->managed) current = recover_selection(profile.last_managed());
            if (current && current->managed && current->payload == selected.payload) {
                selected.root = current->root;
            } else {
                // Preview 1 could switch to a custom checkout without remembering
                // the managed selection. A ready fixed root is its upgrade source.
                auto probe = linux_command(selected, false);
                probe.insert(probe.end(), {L"/usr/bin/test", L"-f", wstring(managed_root) + L"/.managed-ready"});
                if ((current && current->managed) || !wsl(probe).code) {
                    previous_release = current && current->managed ? *current : selected;
                    previous_release->project = selected.project;
                    previous_release->project_id = project_id(selected.project, true);
                    selected = stage_managed_upgrade(profile, folder, *previous_release);
                }
            }
        }
        prepare_managed(profile, folder, selected, pending, dialogs, options);
    } else {
        const auto registration = registered_distro(selected.distro);
        if (!registration || registration->version != 2)
            throw std::runtime_error("Choose a checkout in a registered WSL 2 distribution.");
        selected.project_id = project_id(selected.project, true);
    }

    dialogs.progress(L"Checking the selected workspace.");
    verify_project(profile, selected);
    if (previous_release) {
        activate_managed(profile, *previous_release, selected);
    } else {
        if (!selected.managed) {
            const auto previous = recover_selection(profile.config());
            if (previous && previous->managed)
                atomic_text(profile.last_managed(), read_text(profile.config()));
        }
        save_settings(profile, selected);
    }
    std::error_code ignored;
    fs::remove(profile.pending(), ignored);
    return selected;
}

string service_url(const string& output) {
    constexpr char prefix[] = "http://127.0.0.1:";
    std::istringstream lines(output);
    string url;
    for (string line; std::getline(lines, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        constexpr size_t start = sizeof(prefix) - 1;
        if (line.size() <= start || line.rfind(prefix, 0) != 0 || line.back() != '/') continue;
        const auto port = line.substr(start, line.size() - start - 1);
        if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != string::npos)
            continue;
        if (const int number = std::stoi(port); number > 0 && number <= 65535) url = std::move(line);
    }
    if (url.empty()) throw std::runtime_error("The trainer did not return a valid loopback URL.");
    return url;
}

string launch(const Settings& value, const Options& options, const Profile& profile) {
    if (value.managed) {
        if (elevated())
            throw std::runtime_error("Open traiNNer-GUI as your normal Windows user.");
        if (!managed_registered(profile))
            throw std::runtime_error("The managed Ubuntu distribution is missing. Run with --configure.");
    }
    auto arguments = studio_command(value);
    const bool hold_client = !value.project.empty() && !options.phone_status;
    if (!options.state.empty()) arguments.insert(arguments.end(), {L"--state-dir", options.state});
    if (options.phone_status) arguments.insert(arguments.end(), {L"tailscale", L"status"});
    else {
        arguments.emplace_back(L"url");
        if (hold_client) arguments.emplace_back(L"--hold-wsl");
    }
    const auto result = wsl(arguments, 90000, nullptr, hold_client ? "TRAINNER_GUI_WSL_CLIENT_READY" : "");
    if (result.code)
        throw std::runtime_error(string(options.phone_status ? "Could not read phone status in WSL.\n"
                                     : "Could not start the trainer service in WSL.\n")
                                 + trim_lines(result.err) + '\n' + trim_lines(result.out));
    if (options.phone_status) {
        const auto status = trim_lines(result.out);
        if (status.empty() || status.front() != '{' || status.back() != '}')
            throw std::runtime_error("Studio did not return phone status JSON.");
        return status;
    }
    return service_url(result.out);
}

void remember_legacy(const Profile& profile, const Settings& selected) {
    fs::create_directories(profile.folder);
    Handle lock(CreateFileW((profile.folder / L"setup.lock").c_str(), GENERIC_WRITE, 0, nullptr,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (lock && !fs::exists(profile.config()) && !fs::exists(profile.pending()))
        save_settings(profile, selected);
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    bool interactive = true;
    try {
        int argc{};
        auto argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (!argv) throw std::runtime_error("Could not read the launcher arguments.");
        std::vector<wstring> args(argv + 1, argv + argc);
        LocalFree(argv);
        interactive = std::none_of(args.begin(), args.end(), [](const wstring& arg) {
            return arg == L"--url" || arg == L"--phone-status" || arg == L"--non-interactive"
                || arg == L"--apply-update" || arg == L"--stage-update"
                || arg == L"--help" || arg == L"-h";
        });
        SetEnvironmentVariableW(L"WSL_UTF8", L"1");
        const auto folder = launcher_folder();
        const Profile profile(profile_folder());
        for (size_t index = 0; index + 1 < args.size(); ++index) {
            if (args[index] == L"--state-dir")
                SetEnvironmentVariableW(L"TRAINNER_GUI_WINDOWS_UPDATE_STATE", args[index + 1].c_str());
        }
        if (const auto result = updates::dispatch(folder, profile.folder, args, *payload_hash != 0))
            return *result;
        const auto options = parse_options(args);
        if (options.help) {
            stream(STD_OUTPUT_HANDLE, usage);
            return 0;
        }
        if (options.apply_update || options.stage_update) {
            apply_update(profile, folder, options.stage_update);
            return 0;
        }
        if (options.configure) {
            setup(options, profile, folder, interactive);
            return 0;
        }

        auto selected = load_settings(profile);
        if (!selected) selected = legacy_settings(folder, true);
        bool inferred = false;
        if (!selected && !fs::exists(profile.pending())) {
            selected = legacy_settings(folder, false);
            inferred = selected.has_value();
        }
        if (!selected) {
            if (!interactive)
                throw std::runtime_error("Desktop setup is required. Run traiNNer-GUI.exe --configure.");
            selected = setup(options, profile, folder, true);
        }
        // An original Preview 1 shortcut has no updater. Opening the replacement
        // once migrates its managed backend and preserves the recorded workspace.
        if (selected->managed && selected->root == managed_root && *payload_hash
            && selected->payload != payload_hash && !options.phone_status) {
            try {
                apply_update(profile, folder, false, true);
            } catch (const std::exception& error) {
                log_result(profile, "Update deferred; using the installed backend", {1, {}, error.what()});
            }
        }
        // Serialize selecting and opening a managed service with the short
        // activation transaction. The service itself outlives this lock.
        Handle launch_lock;
        if (selected->managed) {
            lock_settings(profile, launch_lock, 15000);
            selected = load_settings(profile);
            if (!selected) throw std::runtime_error("The desktop configuration is missing.");
        }
        const auto url = launch(*selected, options, profile);
        launch_lock.reset();
        if (inferred && interactive) remember_legacy(profile, *selected);
        if (options.url || options.phone_status) stream(STD_OUTPUT_HANDLE, url + "\n");
        else if (reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", wide(url).c_str(),
                                                        nullptr, nullptr, SW_SHOWNORMAL)) <= 32)
            throw std::runtime_error("Open this address in your browser.\n" + url);
        return 0;
    } catch (const std::exception& error) {
        const auto exit = dynamic_cast<const Exit*>(&error);
        stream(STD_ERROR_HANDLE, string(error.what()) + "\n");
        if (interactive && (!exit || exit->code != ERROR_CANCELLED)) {
            auto message = wide(error.what());
            if (message.size() > 1800) message = message.substr(0, 1800) + L"\n...";
            MessageBoxW(nullptr, message.c_str(), title,
                        MB_OK | (exit && exit->code == ERROR_SUCCESS_REBOOT_REQUIRED ? MB_ICONINFORMATION : MB_ICONERROR));
        }
        return exit ? exit->code : 1;
    }
}
