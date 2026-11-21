// Index the trainer's existing file log. Nothing runs in the optimizer process.
#ifndef __clang__
#error "Build the native training tools with LLVM Clang."
#endif
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace {
using View = std::string_view;
constexpr std::size_t max_line = 256 * 1024;

void strip_left(View& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
}
bool take(View& s, View token) {
    if (s.substr(0, token.size()) != token) return false;
    s.remove_prefix(token.size());
    return true;
}
bool counter(View& s, std::uint64_t& out) {
    strip_left(s);
    out = 0;
    bool any = false;
    while (!s.empty()) {
        if (s.front() == ',' && s.size() > 1 && s[1] >= '0' && s[1] <= '9') {
            s.remove_prefix(1);
            continue;
        }
        if (s.front() < '0' || s.front() > '9') break;
        auto digit = static_cast<unsigned>(s.front() - '0');
        if (out > (UINT64_MAX - digit) / 10) return false;
        out = 10 * out + digit;
        any = true;
        s.remove_prefix(1);
    }
    return any;
}
bool number(View& s, double& d) {
    strip_left(s);
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    if (s.empty()) return false;
    const auto r = std::from_chars(s.data(), s.data()+s.size(), d);
    if (r.ec != std::errc{}) return false;
    s.remove_prefix(static_cast<std::size_t>(r.ptr-s.data()));
    return true;
}
bool name_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '/' || c == '-';
}
void point(std::uint64_t step, View name, double value) {
    std::cout << "P\t" << step << '\t' << name << '\t';
    if (std::isfinite(value)) {
        char chars[64];
        auto r = std::to_chars(chars,chars+sizeof(chars),value);
        std::cout.write(chars,r.ptr-chars);
    } else {
        std::cout << (std::isnan(value) ? "nan" : value > 0 ? "+inf" : "-inf");
    }
    std::cout << '\n';
}

// The upstream NeoSR file handler retains terminal color around [ iter: ... ].
std::string plain_neo(View raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size();) {
        if (raw[i] == '\x1b' && i + 1 < raw.size() && raw[i + 1] == '[') {
            auto end = i + 2;
            while (end < raw.size() && ((raw[end] >= '0' && raw[end] <= '9') || raw[end] == ';')) ++end;
            if (end < raw.size() && raw[end] == 'm') {
                i = end + 1;
                continue;
            }
        }
        out.push_back(raw[i++]);
    }
    return out;
}

void neo_block(std::uint64_t step, View key, View data) {
    double value;
    if (!number(data, value)) return;
    if (key == "performance") {
        if (!take(data, " it/s")) return;
        point(step, "performance/iterations_per_second", value);
        if (take(data, ", data ") && number(data, value) && take(data, "%"))
            point(step, "performance/data_wait_percent", value);
    } else if (key == "mem") {
        double total;
        if (take(data, " / ") && number(data, total) && take(data, " GiB")) {
            point(step, "memory/peak_allocated_GiB", value);
            point(step, "memory/device_total_GiB", total);
        }
    } else if (key == "lr" || key == "lr_d") {
        unsigned group = 0;
        do {
            point(step, std::string(key) + "/" + std::to_string(group++), value);
        } while (number(data, value));
    } else if (key != "eta") {
        strip_left(data);
        if (data.empty()) point(step, key, value);
    }
}

void parse_neo(View raw) {
    const std::string storage = plain_neo(raw);
    View s(storage);
    std::uint64_t epoch, step;
    if (!counter(s, epoch) || !take(s, " ] [ iter:") || !counter(s, step) || !take(s, " ]"))
        return;
    point(step, "epoch", static_cast<double>(epoch));
    strip_left(s);
    while (take(s, "[ ")) {
        auto end = s.find(']');
        if (end == View::npos) return;
        View block = s.substr(0, end);
        auto colon = block.find(':');
        if (colon != View::npos && colon > 0 && colon <= 256) {
            View key = block.substr(0, colon);
            if (std::all_of(key.begin(), key.end(), name_char))
                neo_block(step, key, block.substr(colon + 1));
        }
        s.remove_prefix(end + 1);
        strip_left(s);
    }
}

void parse(View line) {
    // Match the progress message boundary. Tracebacks and serialized options can
    // contain snippets of training output and must not create extra points.
    constexpr View header = " INFO: [epoch:";
    auto marker = line.find(header);
    if (marker == View::npos) {
        constexpr View neo_header = " INFO: [ epoch:";
        marker = line.find(neo_header);
        if (marker >= 19 && marker <= 32) parse_neo(line.substr(marker + neo_header.size()));
        return;
    }
    if (marker > 32 || marker < 19) return;
    View s = line.substr(marker + header.size());
    std::uint64_t epoch, step;
    if (!counter(s,epoch) || !take(s,", iter:") || !counter(s,step) || !take(s,", lr:(")) return;
    auto end_lr = s.find(")]");
    if (end_lr == View::npos) return;
    point(step,"epoch",static_cast<double>(epoch));
    auto rates = s.substr(0,end_lr);
    double value;
    int group = 0;
    while (number(rates,value)) {
        point(step,"lr/"+std::to_string(group++),value);
        if (!take(rates,",")) break;
    }
    s.remove_prefix(end_lr+2);
    strip_left(s);
    while (!s.empty() && s.front() == '[') {
        auto end = s.find(']');
        if (end == View::npos) return;
        View block = s.substr(1,end-1);
        if (take(block,"performance:") && number(block,value)) {
            point(step,"performance/iterations_per_second",value);
        } else if (take(block,"data wait:") && number(block,value) && take(block,"%") && value >= 0 && value <= 100) {
            point(step,"performance/data_wait_percent",value);
        } else if (take(block,"wall:") && number(block,value)) {
            point(step,"performance/wall_updates_per_second",value);
        } else if (take(block,"peak VRAM:") && number(block,value)) {
            point(step,"memory/peak_allocated_GiB",value);
            if (take(block," GB allocated,") && number(block,value))
                point(step,"memory/peak_reserved_GiB",value);
        }
        s.remove_prefix(end+1);
        strip_left(s);
    }
    while (!s.empty()) {
        auto end = std::find_if_not(s.begin(),s.end(),name_char);
        auto n = static_cast<std::size_t>(end-s.begin());
        if (!n || n > 256) return;
        auto key = s.substr(0,n);
        s.remove_prefix(n);
        if (!take(s,":") || !number(s,value)) return;
        if (!s.empty() && s.front() != ' ' && s.front() != '\t' && s.front() != '\r') return;
        point(step,key,value);
        strip_left(s);
    }
}

bool arg(const char* text, std::uint64_t& out) {
    View s(text);
    auto r = std::from_chars(s.data(),s.data()+s.size(),out);
    return r.ec == std::errc{} && r.ptr == s.data()+s.size();
}
}

int main(int argc, char** argv) {
    std::uint64_t start{}, end{}, budget{};
    if (argc != 4 || !arg(argv[1],start) || !arg(argv[2],end) || !arg(argv[3],budget)
        || end < start || !budget) return 2;
    std::ios::sync_with_stdio(false);
    std::array<char,65536> buf;
    std::string line;
    auto pos = start, consumed = start;
    bool overlong = false;
    while (pos < end) {
        auto n = std::fread(buf.data(),1,static_cast<std::size_t>(
            std::min<std::uint64_t>(end-pos,buf.size())),stdin);
        if (!n) {
            if (std::ferror(stdin)) return 1;
            break;
        }
        for (std::size_t i=0;i<n;++i) {
            ++pos;
            if (buf[i] == '\n') {
                if (!overlong) parse(line);
                line.clear();
                overlong = false;
                consumed = pos;
                if (consumed-start >= budget) {
                    std::cout << "C\t" << consumed << '\n';
                    return std::cout ? 0 : 1;
                }
            } else if (line.size() < max_line) {
                line.push_back(buf[i]);
            } else {
                overlong = true;
            }
        }
    }
    // A trainer may have written only part of the last record. Read it again
    // after the next append instead of publishing a truncated metric row.
    std::cout << "C\t" << consumed << '\n';
    return std::cout ? 0 : 1;
}
