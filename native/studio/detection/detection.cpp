// Streaming evaluator for traiNNer's single-IoU box protocol.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {
struct Hit {
    double score;
    bool matched;
};

struct Category {
    std::vector<Hit> ranked;
    std::uint64_t support = 0;
};

struct Evaluation {
    explicit Evaluation(std::size_t classes) : categories(classes) {}
    std::vector<Category> categories;
};

template <typename T> T area(const T* box) {
    return (box[2] - box[0]) * (box[3] - box[1]);
}

template <typename T> T iou(const T* box, T prediction_area, const float* target, float target_area) {
    const T width = std::max(T(0), std::min(box[2], T(target[2])) - std::max(box[0], T(target[0])));
    const T height = std::max(T(0), std::min(box[3], T(target[3])) - std::max(box[1], T(target[1])));
    const T intersection = width * height;
    return intersection / (prediction_area + T(target_area) - intersection);
}

struct ImageCategory {
    std::vector<std::size_t> predictions, regular, ignored;
};

template <typename T>
int add(Evaluation& evaluation, const T* boxes, const double* scores, const std::int64_t* labels,
        std::size_t count, const float* targets, const std::int64_t* target_labels,
        const std::uint8_t* ignore, std::size_t target_count) {
    const auto classes = evaluation.categories.size();
    std::vector<ImageCategory> groups(classes);
    std::vector<float> areas(target_count);
    for (std::size_t i = 0; i < target_count; ++i) {
        if (target_labels[i] < 1 || std::uint64_t(target_labels[i]) > classes) return 1;
        auto& group = groups[target_labels[i] - 1];
        (ignore[i] ? group.ignored : group.regular).push_back(i);
        areas[i] = area(targets + 4 * i);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (labels[i] < 1 || std::uint64_t(labels[i]) > classes || !std::isfinite(scores[i])) return 1;
        groups[labels[i] - 1].predictions.push_back(i);
    }
    for (std::size_t c = 0; c < classes; ++c) {
        auto& group = groups[c];
        auto& stats = evaluation.categories[c];
        stats.support += group.regular.size();
        if (group.predictions.empty()) continue;
        std::stable_sort(group.predictions.begin(), group.predictions.end(),
                         [scores](auto a, auto b) { return scores[a] > scores[b]; });
        std::vector<std::uint8_t> matched(group.regular.size());
        for (const auto p : group.predictions) {
            const auto* box = boxes + 4 * p;
            const auto prediction_area = area(box);
            const auto overlap = [&](std::size_t index) {
                return iou(box, prediction_area, targets + 4 * index, areas[index]);
            };
            auto best = group.regular.size();
            T best_iou = 0;
            for (std::size_t i = 0; i < group.regular.size(); ++i) {
                if (matched[i]) continue;
                const auto value = overlap(group.regular[i]);
                if (value >= T(0.5) && value > best_iou) {
                    best_iou = value;
                    best = i;
                }
            }
            if (best != group.regular.size()) {
                matched[best] = 1;
                stats.ranked.push_back({scores[p], true});
            } else if (std::none_of(group.ignored.begin(), group.ignored.end(),
                       [&](auto index) { return overlap(index) >= T(0.5); })) {
                stats.ranked.push_back({scores[p], false});
            }
        }
    }
    return 0;
}
} // namespace

extern "C" {
void* detection_create(std::size_t classes) noexcept {
    try { return new Evaluation(classes); } catch (...) { return nullptr; }
}

void detection_free(void* state) noexcept { delete static_cast<Evaluation*>(state); }

int detection_add(void* state, int wide, const void* boxes, const double* scores,
                  const std::int64_t* labels, std::size_t count, const float* targets,
                  const std::int64_t* target_labels, const std::uint8_t* ignore,
                  std::size_t target_count) noexcept {
    if (!state || (count && (!boxes || !labels || !scores)) ||
        (target_count && (!targets || !target_labels || !ignore))) return 1;
    auto& evaluation = *static_cast<Evaluation*>(state);
    try {
        if (wide) return add(evaluation, static_cast<const double*>(boxes), scores, labels, count,
                             targets, target_labels, ignore, target_count);
        return add(evaluation, static_cast<const float*>(boxes), scores, labels, count,
                   targets, target_labels, ignore, target_count);
    } catch (...) { return 2; }
}

// After detection_finish, each category is in stable descending score order.
// Return whole equal-confidence groups. The caller can resume at offset without
// retaining another copy of the detections or splitting a tie at a buffer edge.
int detection_score_groups(const void* state, std::size_t category, std::size_t* offset,
                           std::size_t capacity, double* scores, std::uint64_t* tp,
                           std::uint64_t* fp, std::size_t* written) noexcept {
    if (!state || !offset || !capacity || !scores || !tp || !fp || !written) return 1;
    const auto& categories = static_cast<const Evaluation*>(state)->categories;
    if (category >= categories.size()) return 1;
    const auto& hits = categories[category].ranked;
    if (*offset > hits.size()) return 1;
    auto pos = *offset;
    std::size_t n = 0;
    while (pos < hits.size() && n < capacity) {
        const auto start = pos;
        const double score = hits[pos].score;
        std::uint64_t matched = 0;
        do {
            matched += hits[pos++].matched;
        } while (pos < hits.size() && hits[pos].score == score);
        scores[n] = score;
        tp[n] = matched;
        fp[n] = pos - start - matched;
        ++n;
    }
    *written = n;
    *offset = pos;
    return 0;
}

int detection_finish(void* state, std::size_t classes, double* ap, std::uint64_t* support,
                     std::uint64_t* tp, std::uint64_t* fp) noexcept {
    if (!state || !ap || !support || !tp || !fp) return 1;
    auto& categories = static_cast<Evaluation*>(state)->categories;
    if (classes != categories.size()) return 1;
    try {
        for (std::size_t i = 0; i < classes; ++i) {
            auto& category = categories[i];
            auto& hits = category.ranked;
            std::stable_sort(hits.begin(), hits.end(),
                             [](const Hit& a, const Hit& b) { return a.score > b.score; });
            auto true_count = std::uint64_t(std::count_if(hits.begin(), hits.end(),
                                                         [](const Hit& h) { return h.matched; }));
            support[i] = category.support;
            tp[i] = true_count;
            fp[i] = hits.size() - true_count;
            double envelope = 0, sum = 0;
            for (auto end = hits.size(); end > 0; --end) {
                envelope = std::max(envelope, double(true_count) / double(end));
                if (hits[end - 1].matched) {
                    sum += envelope;
                    --true_count;
                }
            }
            ap[i] = category.support ? sum / double(category.support) : 0;
        }
        return 0;
    } catch (...) { return 2; }
}
}
