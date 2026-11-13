#pragma once

#include <string>
#include <vector>

namespace waffle::cli {

int run(const std::vector<std::string>& args);
std::string version_text();

} // namespace waffle::cli
