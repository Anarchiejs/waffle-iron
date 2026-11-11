// State shared by every interface view.
// Everything is read and written on the interface thread; long work goes through Engine
// (std::future) or std::async for work that does not touch llama.cpp (GGUF scan, campaign).
#pragma once

#include "core/engine.hpp"
#include "core/telemetry.hpp"
#include "core/types.hpp"
#include "store/store.hpp"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace waffle::ui {

struct Toast {
    std::string text;
    bool error = false;
    double expires_at = 0; // ImGui::GetTime()
};

struct AppContext {
    Engine& engine;
    Telemetry& telemetry;
    Store& store;

    // Library: filled by the Library view (scan + Store), read by the other views.
    std::vector<ModelInfo> library;
    int selected_model = -1; // index into library, -1 = none

    // Telemetry history (filled by App at ~10 Hz, kept 10 minutes): monitor and bench.
    std::deque<GpuSample> gpu_history;

    // Live generation throughput, published by the view that is generating (sampler, comparator).
    std::deque<std::pair<double, double>> tps_history; // (t_seconds, tokens/s)

    // Page requested by a view (for example "open in the planner"); App applies it, then sets -1.
    int requested_page = -1;

    std::deque<Toast> toasts;
    void notify(std::string text, bool error = false); // defined in app.cpp

    const ModelInfo* selected() const {
        return selected_model >= 0 && selected_model < static_cast<int>(library.size())
                   ? &library[static_cast<size_t>(selected_model)]
                   : nullptr;
    }
};

} // namespace waffle::ui
