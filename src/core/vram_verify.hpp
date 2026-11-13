// One VRAM check shared by the planner view and `waffle plan`.
// Loads the model, decodes a short prompt so compute buffers exist, and reports the increase.
#pragma once

#include "core/types.hpp"

#include <string>

namespace waffle {

class Engine;
class Telemetry;

struct VramVerification {
    uint64_t estimated_gpu = 0;
    uint64_t before = 0;
    uint64_t after = 0;
    uint64_t measured_gpu = 0; // after - before, or 0 when telemetry has no sample
    std::string device_name;
    std::string note;
};

VramVerification verify_model_vram(Engine& engine, Telemetry* telemetry, const ModelInfo& model,
                                   const VramPlanInput& input, const LoadParams& load);

} // namespace waffle
