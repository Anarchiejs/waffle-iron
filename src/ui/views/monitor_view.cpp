#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <imgui.h>
#include <implot.h>
#include <vector>
#include <algorithm>
namespace banc::ui {
namespace {
class MonitorView final : public View {
    int minutes_ = 1;
public:
    const char* title() const override { return "Moniteur"; }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        if (!ctx.telemetry.available()) {
            ImGui::TextWrapped("NVML indisponible : %s. Les mesures GPU ne sont pas disponibles.",
                               ctx.telemetry.error().c_str());
        }
        ImGui::SliderInt("Historique (minutes)", &minutes_, 1, 10);
        if (!ctx.gpu_history.empty()) {
            const auto& g = ctx.gpu_history.back();
            ImGui::PushFont(nullptr, ImGui::GetFontSize() * 1.15f);
            ImGui::Text("GPU %u %%   |   VRAM %s / %s", g.gpu_util_pct, format_bytes(g.vram_used).c_str(),
                        format_bytes(g.vram_total).c_str());
            ImGui::Text("%u °C   |   %.1f W   |   %u MHz", g.temperature_c, g.power_w, g.sm_clock_mhz);
            ImGui::PopFont();
        }
        const double end = ctx.gpu_history.empty() ? 0 : ctx.gpu_history.back().t_seconds;
        std::vector<double> t, util, used, total, temp, power, clock;
        for (const auto& g : ctx.gpu_history)
            if (g.t_seconds >= end - minutes_ * 60) {
                t.push_back(g.t_seconds - end);
                util.push_back(g.gpu_util_pct);
                used.push_back(static_cast<double>(g.vram_used) / GiB);
                total.push_back(static_cast<double>(g.vram_total) / GiB);
                temp.push_back(g.temperature_c);
                power.push_back(g.power_w);
                clock.push_back(g.sm_clock_mhz);
            }
        auto plot = [&](const char* name, const char* unit, const std::vector<double>& values,
                        const std::vector<double>* other = nullptr) {
            if (ImPlot::BeginPlot(name, {-1, 145})) {
                ImPlot::SetupAxes("Secondes avant maintenant", unit);
                ImPlot::SetupAxisLimits(ImAxis_X1, -minutes_ * 60, 0, ImPlotCond_Always);
                ImPlot::SetupAxis(ImAxis_Y1, unit, ImPlotAxisFlags_AutoFit);
                if (!t.empty()) ImPlot::PlotLine(name, t.data(), values.data(), static_cast<int>(t.size()));
                if (other && !t.empty())
                    ImPlot::PlotLine("Capacité", t.data(), other->data(), static_cast<int>(t.size()));
                ImPlot::EndPlot();
            }
        };
        plot("Utilisation GPU", "%", util);
        plot("VRAM", "Go", used, &total);
        plot("Température", "°C", temp);
        plot("Puissance", "W", power);
        plot("Fréquence SM", "MHz", clock);
        std::vector<double> tx, rate;
        const double now = ImGui::GetTime();
        for (const auto& [time, value] : ctx.tps_history)
            if (time >= now - minutes_ * 60) {
                tx.push_back(time - now);
                rate.push_back(value);
            }
        if (ImPlot::BeginPlot("Génération", {-1, 145})) {
            ImPlot::SetupAxes("Secondes avant maintenant", "tokens/s");
            ImPlot::SetupAxisLimits(ImAxis_X1, -minutes_ * 60, 0, ImPlotCond_Always);
            if (!tx.empty()) ImPlot::PlotLine("Débit", tx.data(), rate.data(), static_cast<int>(tx.size()));
            ImPlot::EndPlot();
        }
    }
};
} // namespace
std::unique_ptr<View> make_monitor_view() {
    return std::make_unique<MonitorView>();
}
} // namespace banc::ui
