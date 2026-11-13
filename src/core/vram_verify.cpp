#include "core/vram_verify.hpp"

#include "core/engine.hpp"
#include "core/telemetry.hpp"
#include "core/vram_planner.hpp"

#include <llama.h>

#include <algorithm>

namespace waffle {

VramVerification verify_model_vram(Engine& engine, Telemetry* telemetry, const ModelInfo& model,
                                   const VramPlanInput& input, const LoadParams& load) {
    VramVerification out;
    out.estimated_gpu = estimate_vram(model, input).total_gpu();
    if (telemetry) {
        if (auto sample = telemetry->sample()) out.before = sample->vram_used;
        out.device_name = telemetry->device_name();
    }
    engine
        .submit([&](EngineSession& session) {
            session.load(load);
            auto tokens = session.tokenize("Hello", true);
            if (tokens.empty()) tokens.push_back(0);
            if (tokens.size() > session.info().n_ctx) tokens.resize(session.info().n_ctx);
            session.decode(tokens, 0);
            llama_synchronize(session.ctx());
            if (out.device_name.empty()) out.device_name = session.device_memory().name;
        })
        .get();
    if (telemetry) {
        if (auto sample = telemetry->sample()) {
            out.after = sample->vram_used;
            out.measured_gpu = out.after > out.before ? out.after - out.before : 0;
        } else {
            out.note = telemetry->available() ? "Telemetry returned no sample after the load" : telemetry->error();
        }
    } else {
        out.note = "No telemetry";
    }
    if (telemetry && telemetry->reports_process_memory() && out.measured_gpu)
        out.note = "Metal reports this process allocation, not device VRAM";
    return out;
}

} // namespace waffle
