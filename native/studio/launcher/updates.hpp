#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace updates {

// Call before parsing the ordinary options. Arguments exclude argv[0].
// Internal forwarding removes its own flag. An empty result means continue
// in this launcher, including for --stage-update and --apply-update.
std::optional<int> dispatch(const std::filesystem::path& exe_folder,
                            const std::filesystem::path& profile_folder,
                            std::vector<std::wstring>& args, bool packaged);

} // namespace updates
