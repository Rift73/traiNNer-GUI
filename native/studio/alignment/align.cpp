// Estimate translations of real patch pairs on the low-resolution pixel grid.
#ifndef __clang__
#error This target is built with LLVM Clang.
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int integer(std::string_view value, int minimum, int maximum) {
    int result = 0;
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end || result < minimum || result > maximum)
        throw std::runtime_error("Invalid patch dimensions or search range.");
    return result;
}

struct Gradient {
    float x, y;
};

using Channels = std::array<std::vector<Gradient>, 3>;

template <class Sample>
Channels gradients(const std::vector<Sample>& rgb, int width, int height) {
    std::vector<std::array<float,3>> planes(width * height);
    for (std::size_t i = 0; i < planes.size(); ++i) {
        const float red = rgb[3*i], green = rgb[3*i+1], blue = rgb[3*i+2];
        const float y = (77*red + 150*green + 29*blue) / 256;
        planes[i] = {y, blue-y, red-y};
    }
    Channels result;
    for (auto& channel : result) channel.resize(planes.size());
    for (int y = 1; y < height-1; ++y) {
        for (int x = 1; x < width-1; ++x) {
            const int i = y*width + x;
            for (int c = 0; c < 3; ++c) {
                // Sobel gradients suppress isolated quantization noise while
                // retaining the signs and orientations of edges.
                const auto at = [&](int dx,int dy){ return planes[i+dy*width+dx][c]; };
                result[c][i] = {
                    (at(1,-1)+2*at(1,0)+at(1,1)-at(-1,-1)-2*at(-1,0)-at(-1,1))/8,
                    (at(-1,1)+2*at(0,1)+at(1,1)-at(-1,-1)-2*at(0,-1)-at(1,-1))/8
                };
            }
        }
    }
    return result;
}

struct Surface {
    explicit Surface(int radius) : side(2*radius+1), values(side*side,0.) {}
    int side;
    std::vector<double> values;
    int informative = 0;
};

Surface correlate(const std::vector<Gradient>& reference, const std::vector<Gradient>& low,
                  int width, int height, int radius) {
    const int border = 1 + radius;
    Surface surface(radius);
    double reference_energy = 0;
    const int points = (width-2*border)*(height-2*border);
    for (int y = border; y < height-border; ++y)
        for (int x = border; x < width-border; ++x) {
            const auto g = reference[y*width+x];
            reference_energy += g.x*g.x + g.y*g.y;
        }
    // Both wire formats use 0..255 intensities. Keep the existing minimum
    // gradient energy in those units for every source depth.
    if (reference_energy < points * .0625) return surface;
    bool observed_texture = false;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            double dot = 0, observed_energy = 0;
            for (int y = border; y < height-border; ++y) {
                auto ref = reference.data() + y*width + border;
                auto obs = low.data() + (y+dy)*width + border + dx;
                for (int x = 0; x < width-2*border; ++x) {
                    dot += ref[x].x*obs[x].x + ref[x].y*obs[x].y;
                    observed_energy += obs[x].x*obs[x].x + obs[x].y*obs[x].y;
                }
            }
            const int index = (dy+radius)*surface.side + dx+radius;
            if (observed_energy >= points * .0625) {
                observed_texture = true;
                surface.values[index] = std::clamp(dot/std::sqrt(reference_energy*observed_energy),-1.0,1.0);
            } else {
                surface.values[index] = -1;
            }
        }
    }
    surface.informative = observed_texture;
    return surface;
}

double refine(double before, double peak, double after) {
    const double curvature = 2*(before - 2*peak + after);
    return curvature < -1e-8 ? std::clamp((before-after)/curvature,-.5,.5) : 0.;
}

void print_peak(const Surface& surface) {
    if (!surface.informative) {
        std::cout << "null";
        return;
    }
    const int radius = surface.side/2;
    const auto& values = surface.values;
    const int index = std::max_element(values.begin(),values.end())-values.begin();
    const int x = index%surface.side, y = index/surface.side;
    const bool boundary = !x || !y || x==2*radius || y==2*radius;
    double second = -surface.informative;
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        if (std::abs(i%surface.side-x)>1 || std::abs(i/surface.side-y)>1)
            second = std::max(second,values[i]);
    }
    const double fx = x>0 && x<2*radius ? refine(values[index-1],values[index],values[index+1]) : 0;
    const double fy = y>0 && y<2*radius ? refine(values[index-surface.side],values[index],values[index+surface.side]) : 0;
    const double n = surface.informative;
    std::cout << "{\"dx\":" << x-radius+fx << ",\"dy\":" << y-radius+fy
              << ",\"atLimit\":" << (boundary?"true":"false")
              << ",\"peak\":" << values[index]/n
              << ",\"zero\":" << values[radius*surface.side+radius]/n
              << ",\"peakMargin\":" << (values[index]-second)/n
              << ",\"patchCount\":" << surface.informative << '}';
}

template <class Sample>
void read_pixels(std::vector<Sample>& data) {
    if (!std::cin.read(reinterpret_cast<char*>(data.data()),data.size() * sizeof(Sample)))
        throw std::runtime_error("Incomplete RGB patch.");
}

template <class Sample>
void estimate(int width,int height,int radius,int patches) {
    if (std::min(width,height)-2*(radius+1) < 8)
        throw std::runtime_error("Not enough interior pixels for this search range.");
    const std::array<const char*,3> labels{"luma","blueMinusLuma","redMinusLuma"};
    std::array<Surface,3> totals{Surface(radius),Surface(radius),Surface(radius)};
    std::vector<Sample> pixels(3*width*height);
    std::cout << std::setprecision(7) << "{\"patches\":[";
    for (int p=0; p<patches; ++p) {
        read_pixels(pixels);
        const auto reference = gradients(pixels,width,height);
        read_pixels(pixels);
        const auto observed = gradients(pixels,width,height);
        std::cout << (p?",{":"{");
        for (int c=0;c<3;++c) {
            auto local = correlate(reference[c],observed[c],width,height,radius);
            std::cout << (c?",\"":"\"") << labels[c] << "\":";
            print_peak(local);
            if (local.informative) {
                for (std::size_t i=0;i<local.values.size();++i)
                    totals[c].values[i] += local.values[i];
                ++totals[c].informative;
            }
        }
        std::cout << '}';
    }
    std::cout << "],\"estimate\":{";
    for (int c=0;c<3;++c) {
        std::cout << (c?",\"":"\"") << labels[c] << "\":";
        print_peak(totals[c]);
    }
    std::cout << "}}\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const bool wide = argc == 6 && std::string_view(argv[5]) == "rgb-f32";
        if (argc != 5 && !wide)
            throw std::runtime_error("Usage: align WIDTH HEIGHT RADIUS PATCHES [rgb-f32]");
        const int width = integer(argv[1],16,128), height = integer(argv[2],16,128);
        const int radius = integer(argv[3],2,12), patches = integer(argv[4],1,16);
        if (wide)
            estimate<float>(width, height, radius, patches);
        else
            estimate<unsigned char>(width, height, radius, patches);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
