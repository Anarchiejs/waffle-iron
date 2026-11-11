#include "ui/view.hpp"
#include "ui/theme.hpp"
#include "ui/widgets.hpp"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
namespace waffle::ui {
namespace {
class MonitorView final : public View {
    int minutes_ = 1;
    struct Series {
        const char* title;
        const char* unit;
        const std::vector<double>* x;
        const std::vector<double>* y;
        ImVec4 color;
        double y_min, y_max; // y_max <= y_min : ajustement automatique
        const std::vector<double>* ceiling = nullptr;
    };
    void chart(const Series& s, ImVec2 size) {
        ImGui::PushID(s.title);
        if (begin_card("##graphe", size, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            push_font(fonts().semibold, .95f);
            ImGui::TextUnformatted(s.title);
            ImGui::PopFont();
            if (!s.y->empty()) {
                char value[48];
                std::snprintf(value, sizeof value, "%.*f %s", std::abs(s.y->back()) < 10 ? 1 : 0, s.y->back(), s.unit);
                push_font(fonts().semibold, .95f);
                const float w = ImGui::CalcTextSize(value).x;
                ImGui::SameLine(ImGui::GetContentRegionMax().x - w);
                ImGui::PushStyleColor(ImGuiCol_Text, s.color);
                ImGui::TextUnformatted(value);
                ImGui::PopStyleColor();
                ImGui::PopFont();
            }
            ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, em(.3f)));
            if (ImPlot::BeginPlot("##courbe", {-1, ImGui::GetContentRegionAvail().y},
                                  ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect |
                                      (s.ceiling ? 0 : ImPlotFlags_NoLegend))) {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoHighlight);
                ImPlot::SetupAxisFormat(ImAxis_X1, "%.0f s");
                ImPlot::SetupAxisLimits(ImAxis_X1, -minutes_ * 60, 0, ImPlotCond_Always);
                const bool fixed = s.y_max > s.y_min;
                ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                                  ImPlotAxisFlags_NoHighlight | (fixed ? 0 : ImPlotAxisFlags_AutoFit));
                if (fixed) ImPlot::SetupAxisLimits(ImAxis_Y1, s.y_min, s.y_max, ImPlotCond_Always);
                if (s.ceiling)
                    ImPlot::SetupLegend(ImPlotLocation_SouthEast, ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_NoButtons);
                const int n = static_cast<int>(s.x->size());
                if (n > 0) {
                    ImPlotSpec fill;
                    fill.FillColor = s.color;
                    fill.FillAlpha = .16f;
                    ImPlot::PlotShaded("##aire", s.x->data(), s.y->data(), n, 0.0, fill);
                    ImPlotSpec line;
                    line.LineColor = s.color;
                    line.LineWeight = 2.0f;
                    ImPlot::PlotLine(s.ceiling ? "In use" : "##line", s.x->data(), s.y->data(), n, line);
                    if (s.ceiling) {
                        ImPlotSpec cap;
                        cap.LineColor = col::faint;
                        cap.LineWeight = 1.5f;
                        ImPlot::PlotLine("Capacity", s.x->data(), s.ceiling->data(), n, cap);
                    }
                }
                ImPlot::EndPlot();
            }
            ImPlot::PopStyleVar();
        }
        end_card();
        ImGui::PopID();
    }

public:
    const char* title() const override { return "Monitor"; }
    const char* subtitle() const override {
        return "Live GPU telemetry: load, memory, temperature, power, and generation throughput.";
    }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        if (!ctx.telemetry.available()) {
            if (begin_card("##nvml")) {
                card_title("NVML indisponible");
                dim_wrapped((ctx.telemetry.error() + ". GPU measurements are not available.").c_str());
            }
            end_card();
        }
        // History window: segmented buttons.
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Window");
        for (int m : {1, 2, 5, 10}) {
            ImGui::SameLine();
            char label[16];
            std::snprintf(label, sizeof label, "%d min", m);
            const bool on = minutes_ == m;
            if (on ? primary_button(label) : ImGui::Button(label)) minutes_ = m;
        }
        ImGui::Dummy({0, em(.1f)});

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
        std::vector<double> tx, rate;
        const double now = ImGui::GetTime();
        for (const auto& [time, value] : ctx.tps_history)
            if (time >= now - minutes_ * 60) {
                tx.push_back(time - now);
                rate.push_back(value);
            }

        std::vector<Tile> tiles;
        if (!ctx.gpu_history.empty()) {
            const auto& g = ctx.gpu_history.back();
            char buf[6][48];
            std::snprintf(buf[0], 48, "%u %%", g.gpu_util_pct);
            std::snprintf(buf[1], 48, "/ %s", format_bytes(g.vram_total).c_str());
            std::snprintf(buf[2], 48, "%u °C", g.temperature_c);
            std::snprintf(buf[3], 48, "%.1f W", g.power_w);
            std::snprintf(buf[4], 48, "%u MHz", g.sm_clock_mhz);
            tiles = {{"GPU load", buf[0], "", col::accent},
                     {"VRAM used", format_bytes(g.vram_used), buf[1], col::teal},
                     {"Temperature", buf[2], "", col::amber},
                     {"Puissance", buf[3], "", col::mauve},
                     {"SM clock", buf[4], "", col::accent}};
        }
        {
            char rate_text[32] = "—";
            if (!rate.empty()) std::snprintf(rate_text, sizeof rate_text, "%.1f", rate.back());
            tiles.push_back({"Generation", rate_text, rate.empty() ? "none in progress" : "tok/s", col::good});
        }
        tile_grid(tiles, em(7.5f));
        ImGui::Dummy({0, em(.2f)});

        const double vram_cap = total.empty() ? 0 : total.back();
        const Series series[] = {
            {"GPU load", "%", &t, &util, col::accent, 0, 100},
            {"VRAM", "Go", &t, &used, col::teal, 0, std::max(1.0, vram_cap * 1.05), &total},
            {"Temperature", "°C", &t, &temp, col::amber, 0, 0},
            {"Puissance", "W", &t, &power, col::mauve, 0, 0},
            {"SM clock", "MHz", &t, &clock, col::accent, 0, 0},
            {"Generation throughput", "tok/s", &tx, &rate, col::good, 0, 0},
        };
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const int cols = avail.x >= em(50) ? 3 : avail.x >= em(30) ? 2 : 1;
        const int rows = (6 + cols - 1) / cols;
        const float w = std::floor((avail.x - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols));
        const float h = std::max(em(9.5f), std::floor((avail.y - ImGui::GetStyle().ItemSpacing.y * (rows - 1)) / rows));
        for (int i = 0; i < 6; ++i) {
            if (i % cols) ImGui::SameLine(0, gap);
            chart(series[i], {w, h});
        }
    }
};
} // namespace
std::unique_ptr<View> make_monitor_view() {
    return std::make_unique<MonitorView>();
}
} // namespace waffle::ui
