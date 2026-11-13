#include "core/text_diff.hpp"

#include <cstdint>
#include <utility>

namespace waffle {
namespace {

std::vector<std::string_view> words(std::string_view text) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < text.size()) {
        size_t j = i;
        const bool space = text[i] == ' ' || text[i] == '\n' || text[i] == '\t' || text[i] == '\r';
        if (space) {
            while (j < text.size() && (text[j] == ' ' || text[j] == '\n' || text[j] == '\t' || text[j] == '\r')) ++j;
        } else {
            while (j < text.size() && text[j] != ' ' && text[j] != '\n' && text[j] != '\t' && text[j] != '\r') ++j;
        }
        out.push_back(text.substr(i, j - i));
        i = j;
    }
    return out;
}

} // namespace

std::vector<DiffSpan> word_diff(std::string_view left, std::string_view right) {
    const auto a = words(left);
    const auto b = words(right);
    const size_t n = a.size();
    const size_t m = b.size();
    // Cap the table so a huge generation does not allocate a square of millions of cells.
    constexpr size_t kMax = 4000;
    if (n > kMax || m > kMax) {
        std::vector<DiffSpan> coarse;
        if (!left.empty()) coarse.push_back({DiffOp::Delete, std::string(left)});
        if (!right.empty()) coarse.push_back({DiffOp::Insert, std::string(right)});
        return coarse;
    }
    std::vector<std::pair<DiffOp, std::string_view>> raw;
    size_t i = n, j = m;
    std::vector<int> dp((n + 1) * (m + 1));
    auto at = [&](size_t r, size_t c) -> int& { return dp[r * (m + 1) + c]; };
    for (size_t r = 1; r <= n; ++r) {
        for (size_t c = 1; c <= m; ++c) {
            if (a[r - 1] == b[c - 1])
                at(r, c) = at(r - 1, c - 1) + static_cast<int>(a[r - 1].size());
            else
                at(r, c) = std::max(at(r - 1, c), at(r, c - 1));
        }
    }
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) {
            raw.push_back({DiffOp::Equal, a[i - 1]});
            --i;
            --j;
        } else if (at(i - 1, j) >= at(i, j - 1)) {
            raw.push_back({DiffOp::Delete, a[i - 1]});
            --i;
        } else {
            raw.push_back({DiffOp::Insert, b[j - 1]});
            --j;
        }
    }
    while (i > 0) raw.push_back({DiffOp::Delete, a[--i]});
    while (j > 0) raw.push_back({DiffOp::Insert, b[--j]});
    std::vector<DiffSpan> out;
    for (auto it = raw.rbegin(); it != raw.rend(); ++it) {
        if (!out.empty() && out.back().op == it->first)
            out.back().text += it->second;
        else
            out.push_back({it->first, std::string(it->second)});
    }
    return out;
}

} // namespace waffle
