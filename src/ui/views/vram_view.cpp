#include "ui/view.hpp"
#include "ui/theme.hpp"
#include "ui/widgets.hpp"
#include "core/gguf_reader.hpp"
#include "core/vram_planner.hpp"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cstdio>
#include <future>
#include <map>
#include <ctime>
#include <thread>

namespace waffle::ui {
namespace {
std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return {bytes.begin(), bytes.end()};
}
VramPlanInput plan_input(const LoadParams& p) {
    return {p.n_gpu_layers, p.n_ctx,           p.n_batch,   p.n_ubatch, p.type_k,
            p.type_v,       p.flash_attn != 0, p.n_seq_max, p.swa_full};
}
std::string config_label(const LoadParams& p) {
    return "ngl=" + std::to_string(p.n_gpu_layers) + " ctx=" + std::to_string(p.n_ctx) + " kv=" + p.type_k + "/" +
           p.type_v + " fa=" + std::to_string(p.flash_attn);
}
std::string signed_bytes(int64_t v) {
    return (v < 0 ? "−" : "+") + format_bytes(static_cast<uint64_t>(v < 0 ? -v : v));
}
class VramView final : public View {
    LoadParams params_;
    std::map<std::filesystem::path, ModelInfo> cache_;
    std::future<ModelInfo> reading_;
    std::future<VramCheck> checking_;
    std::filesystem::path requested_, failed_;
    std::string history_path_;
    std::vector<VramCheck> history_;
    std::optional<VramCheck> last_check_;

    // Left column: model and simulated settings. Returns the read model (tensors) or nullptr.
    const ModelInfo* config(AppContext& ctx, const VramEstimate* estimate, const std::optional<GpuSample>& sample) {
        card_title("Configuration", "Simulated load settings.");
        field_label("Model");
        model_combo("##model", ctx, ctx.selected_model);
        const auto* selected = ctx.selected();
        if (!selected) {
            ImGui::Dummy({0, em(.2f)});
            dim_wrapped("Choose a model here or in the library.");
            return nullptr;
        }
        const auto path = selected->path;
        if (!cache_.contains(path)) {
            ImGui::Dummy({0, em(.2f)});
            if (path == failed_) {
                dim_wrapped("The tensor read failed.");
                if (ImGui::Button("Retry the GGUF read", {-FLT_MIN, 0})) failed_.clear();
            } else
                busy_indicator("Reading tensors...");
            return nullptr;
        }
        const auto& model = cache_.at(path);
        params_.model_path = path;
        load_params_editor(params_, static_cast<int>(model.n_layer));
        if (!estimate) return &model;

        ImGui::Dummy({0, em(.2f)});
        ImGui::Separator();
        ImGui::Dummy({0, em(.1f)});
        const uint64_t capacity = sample ? sample->vram_total : 0;
        const uint64_t used = sample ? sample->vram_used : 0;
        ImGui::BeginDisabled(!sample);
        if (ImGui::Button("Max layers that fit", {-FLT_MIN, 0}))
            params_.n_gpu_layers =
                max_gpu_layers_for(model, plan_input(params_), capacity > used ? capacity - used : 0);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Largest layer count whose estimate fits in the VRAM that is free right now.");
        ImGui::BeginDisabled(checking_.valid() || ctx.engine.state() != EngineState::Idle || !sample);
        if (primary_button("Verify for real", {-FLT_MIN, 0})) start_check(ctx, estimate->total_gpu());
        ImGui::EndDisabled();
        if (checking_.valid() && busy_indicator("Verifying...")) ctx.engine.cancel_all();
        ImGui::Dummy({0, em(.1f)});
        push_font(nullptr, .88f);
        dim_wrapped("Verification unloads the current model, measures one isolated load, then unloads it.");
#ifndef WAFFLE_HAS_CUDA
        dim_wrapped("CPU build: the load uses the CPU; the measurement does not validate a CUDA estimate.");
#endif
        if (!sample) dim_wrapped("NVML unavailable: capacity and a real check are unknown.");
        ImGui::PopFont();
        return &model;
    }
    void start_check(AppContext& ctx, uint64_t estimated) {
        auto* telemetry = &ctx.telemetry;
        auto* store = &ctx.store;
        const auto params = params_;
        checking_ = ctx.engine.submit([telemetry, store, params, estimated](EngineSession& session) {
            session.unload();
            // Let NVML observe the free before taking the baseline.
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto before = telemetry->sample();
            if (!before) throw Error("NVML sample unavailable before load");
            try {
                session.load(params);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                auto after = telemetry->sample();
                if (!after) throw Error("NVML sample unavailable after load");
                VramCheck result;
                result.model_path = path_utf8(params.model_path);
                result.config = config_label(params);
                result.estimated_gpu = estimated;
                result.measured_gpu = after->vram_used > before->vram_used ? after->vram_used - before->vram_used : 0;
                const auto now = std::time(nullptr);
                std::tm utc{};
                gmtime_s(&utc, &now);
                char date[32];
                std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", &utc);
                result.timestamp = date;
                session.unload();
                store->save_vram_check(result);
                return result;
            } catch (...) {
                session.unload();
                throw;
            }
        });
    }
    void results(AppContext& ctx, const ModelInfo& model, const VramEstimate& estimate,
                 const std::optional<GpuSample>& sample) {
        const uint64_t capacity = sample ? sample->vram_total : 0;
        const uint64_t used = sample ? sample->vram_used : 0;
        const uint64_t free = capacity > used ? capacity - used : 0;
        std::vector<Tile> tiles = {{"GPU estimate", format_bytes(estimate.total_gpu()), "", col::accent},
                                   {"Hors GPU (RAM)", format_bytes(estimate.total_cpu()), "", col::teal}};
        if (sample) {
            const int64_t margin = static_cast<int64_t>(free) - static_cast<int64_t>(estimate.total_gpu());
            tiles.push_back({"VRAM libre", format_bytes(free), "/ " + format_bytes(capacity), col::mauve});
            tiles.push_back({"Margin", signed_bytes(margin), margin >= 0 ? "it fits" : "over budget",
                             margin >= 0 ? col::good : col::bad});
        }
        tile_grid(tiles, em(7.5f));
        ImGui::Dummy({0, em(.1f)});

        const float chart_h = std::clamp(ImGui::GetContentRegionAvail().y * .5f, em(15), em(24));
        if (begin_card("##Breakdown", {0, chart_h}, ImGuiWindowFlags_NoScrollbar)) {
            card_title("Memory breakdown");
            ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, em(.3f)));
            if (ImPlot::BeginPlot("##Breakdown", {-1, ImGui::GetContentRegionAvail().y},
                                  ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
                ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoHighlight);
                ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoHighlight);
                ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f Go");
                ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal |
                                                              ImPlotLegendFlags_NoButtons);
                ImPlot::SetupAxisLimits(ImAxis_X1, -.6, 1.6, ImPlotCond_Always);
                const double ymax =
                    static_cast<double>(std::max({estimate.total_gpu() + used, estimate.total_cpu(), capacity})) / GiB;
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, std::max(1.0, ymax * 1.12), ImPlotCond_Always);
                const double ticks[] = {0, 1};
                const char* labels[] = {"GPU", "CPU"};
                ImPlot::SetupAxisTicks(ImAxis_X1, ticks, 2, labels);
                const char* components[] = {"Poids", "Cache KV", "Tampons", "Overhead"};
                const double data[] = {
                    static_cast<double>(estimate.weights_gpu) / GiB,  static_cast<double>(estimate.weights_cpu) / GiB,
                    static_cast<double>(estimate.kv_cache_gpu) / GiB, static_cast<double>(estimate.kv_cache_cpu) / GiB,
                    static_cast<double>(estimate.compute_gpu) / GiB,  static_cast<double>(estimate.compute_cpu) / GiB,
                    static_cast<double>(estimate.overhead_gpu) / GiB, 0};
                ImPlotSpec stacked;
                stacked.Flags = ImPlotBarGroupsFlags_Stacked;
                ImPlot::PlotBarGroups(components, data, 4, 2, .55, 0, stacked);
                // Horizontal thresholds drawn as full-width segments (easier to read than infinite lines).
                const double span[] = {-.6, 1.6};
                ImPlotSpec limit_line;
                limit_line.LineWeight = 2;
                if (capacity) {
                    const double limit = static_cast<double>(capacity) / GiB;
                    const double ys[] = {limit, limit};
                    limit_line.LineColor = col::bad;
                    ImPlot::PlotLine("VRAM totale", span, ys, 2, limit_line);
                }
                if (sample) {
                    const double occupied = static_cast<double>(used) / GiB;
                    const double ys[] = {occupied, occupied};
                    limit_line.LineColor = col::dim;
                    ImPlot::PlotLine("Already in use", span, ys, 2, limit_line);
                }
                ImPlot::EndPlot();
            }
            ImPlot::PopStyleVar();
        }
        end_card();

        if (!estimate.notes.empty()) {
            if (begin_card("##Notes")) {
                card_title("Planner notes");
                for (const auto& note : estimate.notes) {
                    ImGui::Bullet();
                    ImGui::SameLine();
                    ImGui::TextWrapped("%s", note.c_str());
                }
            }
            end_card();
        }

        const auto path = path_utf8(model.path);
        if (history_path_ != path) {
            history_ = ctx.store.vram_checks(path);
            history_path_ = path;
        }
        if (begin_card("##Checks")) {
            card_title("Real checks", "NVML measures the whole GPU: other processes can move the result.");
            if (last_check_ && last_check_->model_path == path) {
                const auto& check = *last_check_;
                char gap[32] = "—";
                if (check.estimated_gpu)
                    std::snprintf(gap, sizeof gap, "%+.1f %%",
                                  100 * (static_cast<double>(check.measured_gpu) / check.estimated_gpu - 1));
                tile_grid({{"Last measurement", format_bytes(check.measured_gpu), "", col::teal},
                           {"Gap, measured vs estimate", gap, "", col::amber}},
                          em(11));
            }
            if (history_.empty())
                dim_wrapped("No check recorded for this model.");
            else if (ImGui::BeginTable("historique", 5,
                                       ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                           ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollX)) {
                ImGui::TableSetupColumn("Date (UTC)", 0, 1.2f);
                ImGui::TableSetupColumn("Configuration", 0, 2.4f);
                ImGui::TableSetupColumn("Estimated", 0, .8f);
                ImGui::TableSetupColumn("Measured", 0, .8f);
                ImGui::TableSetupColumn("Gap", 0, .6f);
                ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
                ImGui::TableHeadersRow();
                ImGui::PopStyleColor();
                for (const auto& check : history_) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(check.timestamp.c_str());
                    ImGui::TableNextColumn();
                    push_font(fonts().mono, .85f);
                    ImGui::TextWrapped("%s", check.config.c_str());
                    ImGui::PopFont();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(format_bytes(check.estimated_gpu).c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(format_bytes(check.measured_gpu).c_str());
                    ImGui::TableNextColumn();
                    if (check.estimated_gpu)
                        ImGui::Text("%+.1f %%",
                                    100 * (static_cast<double>(check.measured_gpu) / check.estimated_gpu - 1));
                }
                ImGui::EndTable();
            }
        }
        end_card();
    }

public:
    ~VramView() override {
        if (checking_.valid()) checking_.wait();
    }
    const char* title() const override { return "VRAM planner"; }
    const char* subtitle() const override {
        return "Estimate the memory of a load before you run it, then check it with a real measurement.";
    }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        if (is_ready(reading_)) {
            try {
                auto model = reading_.get();
                const auto key = model.path;
                cache_.insert_or_assign(key, std::move(model));
            } catch (const std::exception& e) {
                failed_ = requested_;
                ctx.notify(e.what(), true);
            }
        }
        if (is_ready(checking_)) {
            try {
                last_check_ = checking_.get();
                history_path_.clear();
                ctx.notify("VRAM check saved");
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
            }
        }
        // Full read (tensors) of the chosen model, in the background.
        const ModelInfo* model = nullptr;
        if (const auto* selected = ctx.selected()) {
            const auto path = selected->path;
            if (!cache_.contains(path) && !reading_.valid() && path != failed_) {
                requested_ = path;
                reading_ = std::async(std::launch::async, [path] { return read_gguf(path, true); });
            }
            if (cache_.contains(path)) model = &cache_.at(path);
        }
        std::optional<VramEstimate> estimate;
        if (model) {
            params_.model_path = model->path;
            estimate = estimate_vram(*model, plan_input(params_));
        }
        const auto sample = ctx.telemetry.sample();

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float gap = em(1);
        const bool wide = avail.x >= em(46);
        if (wide) {
            if (begin_card("##Configuration", {side_width(.32f, 20, 25), avail.y}))
                config(ctx, estimate ? &*estimate : nullptr, sample);
            end_card();
            ImGui::SameLine(0, gap);
            ImGui::BeginChild("##Results", {0, avail.y}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        } else {
            if (begin_card("##Configuration")) config(ctx, estimate ? &*estimate : nullptr, sample);
            end_card();
        }
        // The settings may have just changed: a fresh estimate for display.
        if (model) estimate = estimate_vram(*model, plan_input(params_));
        if (model && estimate)
            results(ctx, *model, *estimate, sample);
        else if (wide)
            empty_state(ctx.selected() ? "Reading the model..." : "No model selected",
                        ctx.selected() ? "Tensors are read to estimate each part of the memory."
                                       : "Choose a model on the left to estimate its VRAM use from the "
                                         "offloaded layers, the context, and the cache type.");
        if (wide) ImGui::EndChild();
    }
};
} // namespace
std::unique_ptr<View> make_vram_view() {
    return std::make_unique<VramView>();
}
} // namespace waffle::ui
