// Metal device probe for the macOS monitor. Utilization, temperature, power, and
// clocks are not available through a public API; memory figures come from the
// system device (recommended working set and this process's current allocation).
#pragma once

#include <cstdint>
#include <string>

namespace waffle {

struct MetalProbe {
    std::string name;
    std::string error;
    uint64_t used = 0;
    uint64_t total = 0;
};

bool metal_gpu_probe(MetalProbe& out);

} // namespace waffle
