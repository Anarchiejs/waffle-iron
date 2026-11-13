// Word-level difference of two texts, used by the comparator.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace waffle {

enum class DiffOp { Equal, Delete, Insert };

struct DiffSpan {
    DiffOp op = DiffOp::Equal;
    std::string text;
};

std::vector<DiffSpan> word_diff(std::string_view left, std::string_view right);

} // namespace waffle
