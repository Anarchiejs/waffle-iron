#include "ui/icons.hpp"
#include "ui/view.hpp"
#include "ui/theme.hpp"
#include "ui/widgets.hpp"
#include "core/gguf_reader.hpp"
#include "core/util.hpp"
#include "core/vram_planner.hpp"
#include "core/vram_verify.hpp"
#include <imgui.h>
#include <algorithm>
#include <cmath>
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
    VramPlanInput input{p.n_gpu_layers, p.n_ctx,           p.n_batch,   p.n_ubatch, p.type_k,
                        p.type_v,       p.flash_attn != 0, p.n_seq_max, p.swa_full};
    input.flash_attn_auto = p.flash_attn < 0;
    if (p.flash_attn < 0) input.flash_attn = true;
    return input;
}
std::string config_label(const LoadParams& p) {
    return "ngl=" + std::to_string(p.n_gpu_layers) + " ctx=" + std::to_string(p.n_ctx) + " kv=" + p.type_k + "/" +
           p.type_v + " fa=" + std::to_string(p.flash_attn);
}
std::string signed_bytes(int64_t v) {
    return (v < 0 ? "−" : "+") + format_bytes(static_cast<uint64_t>(v < 0 ? -v : v));
}
bool memory_was_measured(const VramCheck& check) {
    return check.config.find("memory not measured") == std::string::npos;
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
#if defined(__APPLE__)
        const bool need_sample = false;
#else
        const bool need_sample = !sample;
#endif
        ImGui::BeginDisabled(checking_.valid() || ctx.engine.state() != EngineState::Idle || need_sample);
        if (primary_button("Verify for real", {-FLT_MIN, 0})) start_check(ctx, estimate->total_gpu());
        ImGui::EndDisabled();
        if (checking_.valid() && busy_indicator("Verifying...")) ctx.engine.cancel_all();
        ImGui::Dummy({0, em(.1f)});
        push_font(nullptr, .88f);
        dim_wrapped("Verification unloads the current model, measures one isolated load, then unloads it.");
#if !defined(WAFFLE_HAS_CUDA) && !defined(WAFFLE_HAS_METAL)
        dim_wrapped("CPU build: the load uses the CPU; the measurement does not validate a CUDA estimate.");
#elif defined(WAFFLE_HAS_METAL)
        dim_wrapped("Metal build: verification loads the model on this Mac. The estimate's fixed overhead was "
                    "calibrated on CUDA.");
#endif
#if defined(__APPLE__)
        if (!sample)
            dim_wrapped(
                "No Metal memory reading. Verify for real still loads the model; the memory gap is not recorded.");
#else
        if (!sample) dim_wrapped("NVML unavailable: capacity and a real check are unknown.");
#endif
        ImGui::PopFont();
        return &model;
    }
    void start_check(AppContext& ctx, uint64_t estimated) {
        auto* engine = &ctx.engine;
        auto* telemetry = &ctx.telemetry;
        auto* store = &ctx.store;
        const auto params = params_;
        checking_ = std::async(std::launch::async, [engine, telemetry, store, params, estimated] {
            const ModelInfo model = read_gguf(params.model_path, false);
            const auto verified = verify_model_vram(*engine, telemetry, model, plan_input(params), params);
            VramCheck result;
            result.model_path = path_utf8(params.model_path);
            result.config = config_label(params);
            result.estimated_gpu = estimated ? estimated : verified.estimated_gpu;
            const bool measured = verified.note.empty() || verified.note.starts_with("Metal reports");
            if (measured) {
                result.measured_gpu = verified.measured_gpu;
            } else {
                result.config += " (memory not measured)";
                result.measured_gpu = 0;
            }
            result.timestamp = now_iso();
            store->save_vram_check(result);
            return result;
        });
    }
    void results(AppContext& ctx, const ModelInfo& model, const VramEstimate& estimate,
                 const std::optional<GpuSample>& sample) {
        const uint64_t capacity = sample ? sample->vram_total : 0;
        const uint64_t used = sample ? sample->vram_used : 0;
        const uint64_t free = capacity > used ? capacity - used : 0;
        std::vector<Tile> tiles = {{"GPU estimate", format_bytes(estimate.total_gpu()), "", col::accent},
                                   {"Off GPU (RAM)", format_bytes(estimate.total_cpu()), "", col::blue}};
        if (sample) {
            const int64_t margin = static_cast<int64_t>(free) - static_cast<int64_t>(estimate.total_gpu());
            tiles.push_back({"Free VRAM", format_bytes(free), "/ " + format_bytes(capacity), col::berry});
            tiles.push_back({"Margin", signed_bytes(margin), margin >= 0 ? "it fits" : "over budget",
                             margin >= 0 ? col::good : col::bad});
        }
        tile_grid(tiles, em(7.5f));
        ImGui::Dummy({0, em(.1f)});

        // Memory breakdown as a waffle: the GPU at a fixed cell size, so the grid is the card's capacity.
        if (begin_card("##Breakdown")) {
            card_title("Memory breakdown");
            const std::vector<WaffleSlice> gpu = {
                {"Already in use", format_bytes(used), static_cast<double>(used), ImVec4(.42f, .376f, .322f, 1), true},
                {"Weights", format_bytes(estimate.weights_gpu), static_cast<double>(estimate.weights_gpu), col::accent},
                {"KV cache", format_bytes(estimate.kv_cache_gpu), static_cast<double>(estimate.kv_cache_gpu),
                 col::blue},
                {"Buffers", format_bytes(estimate.compute_gpu), static_cast<double>(estimate.compute_gpu), col::mint},
                {"Overhead", format_bytes(estimate.overhead_gpu), static_cast<double>(estimate.overhead_gpu),
                 col::berry},
            };
            double need = 0;
            for (const auto& slice : gpu) need += slice.amount;
            // With a capacity, the grid is the whole card (empty cells are free memory, cells past it are
            // outlined in raspberry). Without one, the grid only shows the estimate.
            const double cap = capacity ? static_cast<double>(capacity) : need;
            constexpr int cells = 96;
            char scale[96];
            std::snprintf(scale, sizeof scale, "GPU · 1 cell = %s · %d cells = %s",
                          format_bytes(static_cast<uint64_t>(cap / cells)).c_str(), cells,
                          capacity ? ("Total VRAM " + format_bytes(capacity)).c_str() : "the estimate");
            push_font(fonts().mono, .8f);
            ImGui::TextDisabled("%s", scale);
            ImGui::PopFont();
            waffle("##gpu", gpu, cells, 24, cap, em(1.6f));
            auto legend = gpu;
            if (!sample) legend.erase(legend.begin());
            if (capacity)
                legend.push_back({need > cap ? "Over budget" : "Free",
                                  format_bytes(static_cast<uint64_t>(need > cap ? need - cap : cap - need)), 0,
                                  need > cap ? col::bad : col::cell_off});
            waffle_legend(legend);
            // Off GPU: one stacked bar, same colors.
            ImGui::Dummy({0, em(.2f)});
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            const float label_w = em(3), value_w = em(6);
            const float bar_w = std::max(em(4), w - label_w - value_w);
            const float bar_h = em(.75f);
            const float line = ImGui::GetTextLineHeight();
            auto* dl = ImGui::GetWindowDrawList();
            dl->AddLine({p.x, p.y}, {p.x + w, p.y}, ImGui::GetColorU32(col::border));
            const float y = p.y + em(.6f);
            dl->AddText({p.x, y}, ImGui::GetColorU32(col::dim), "CPU");
            const ImVec2 b0{p.x + label_w, y + (line - bar_h) * .5f};
            dl->AddRectFilled(b0, {b0.x + bar_w, b0.y + bar_h}, ImGui::GetColorU32(col::cell_off), em(.2f));
            const auto cpu_total = static_cast<double>(estimate.total_cpu());
            const double cpu_scale = std::max({cpu_total, cap, 1.0});
            const std::pair<uint64_t, ImVec4> cpu_parts[] = {{estimate.weights_cpu, col::accent},
                                                             {estimate.kv_cache_cpu, col::blue},
                                                             {estimate.compute_cpu, col::mint}};
            float x = b0.x;
            for (const auto& [bytes, color] : cpu_parts) {
                const float pw = bar_w * static_cast<float>(static_cast<double>(bytes) / cpu_scale);
                if (pw <= 0) continue;
                dl->AddRectFilled({x, b0.y}, {x + pw, b0.y + bar_h}, ImGui::GetColorU32(color));
                x += pw;
            }
            const std::string cpu_text = format_bytes(estimate.total_cpu());
            const float tw = fonts().mono->CalcTextSizeA(em(.88f), FLT_MAX, 0, cpu_text.c_str()).x;
            dl->AddText(fonts().mono, em(.88f), {p.x + w - tw, y + em(.05f)}, ImGui::GetColorU32(col::dim),
                        cpu_text.c_str());
            ImGui::Dummy({w, line + em(.8f)});
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Off GPU (RAM): weights %s · KV cache %s · buffers %s", format_bytes(estimate.weights_cpu).c_str(),
                    format_bytes(estimate.kv_cache_cpu).c_str(), format_bytes(estimate.compute_cpu).c_str());
        }
        end_card();

        if (!estimate.notes.empty()) {
            if (begin_card("##Notes")) {
                card_title("Planner notes");
                for (const auto& note : estimate.notes) {
                    // Small syrup square as the bullet.
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const float sq = em(.4f), line = ImGui::GetTextLineHeight();
                    ImGui::GetWindowDrawList()->AddRectFilled({p.x, p.y + (line - sq) * .5f},
                                                              {p.x + sq, p.y + (line + sq) * .5f},
                                                              ImGui::GetColorU32(col::accent), em(.08f));
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + sq + em(.6f));
                    ImGui::PushStyleColor(ImGuiCol_Text, col::soft);
                    ImGui::TextWrapped("%s", note.c_str());
                    ImGui::PopStyleColor();
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
#if defined(__APPLE__)
            card_title("Real checks",
                       "When Metal reports memory, the figure is this process's allocation, not the whole machine.");
#else
            card_title("Real checks", "NVML measures the whole GPU: other processes can move the result.");
#endif
            if (last_check_ && last_check_->model_path == path) {
                const auto& check = *last_check_;
                char gap[32] = "—";
                const bool measured = memory_was_measured(check);
                if (measured && check.estimated_gpu)
                    std::snprintf(gap, sizeof gap, "%+.1f %%",
                                  100 * (static_cast<double>(check.measured_gpu) / check.estimated_gpu - 1));
                tile_grid(
                    {{"Last measurement", measured ? format_bytes(check.measured_gpu) : "Not measured", "", col::blue},
                     {"Gap, measured vs estimate", gap, "", col::butter}},
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
                    push_font(fonts().mono, .88f);
                    ImGui::TextUnformatted(check.timestamp.c_str());
                    ImGui::PopFont();
                    ImGui::TableNextColumn();
                    push_font(fonts().mono, .85f);
                    ImGui::TextWrapped("%s", check.config.c_str());
                    ImGui::PopFont();
                    push_font(fonts().mono, .92f);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(format_bytes(check.estimated_gpu).c_str());
                    ImGui::TableNextColumn();
                    if (memory_was_measured(check))
                        ImGui::TextUnformatted(format_bytes(check.measured_gpu).c_str());
                    else
                        ImGui::TextDisabled("Not measured");
                    ImGui::TableNextColumn();
                    if (memory_was_measured(check) && check.estimated_gpu) {
                        const double gap = 100 * (static_cast<double>(check.measured_gpu) / check.estimated_gpu - 1);
                        ImGui::TextColored(std::abs(gap) < 5    ? col::good
                                           : std::abs(gap) < 15 ? col::warn
                                                                : col::bad,
                                           "%+.1f %%", gap);
                    } else
                        ImGui::TextDisabled("—");
                    ImGui::PopFont();
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
    const char* icon() const override { return icon_vram; }
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
                if (last_check_ && !memory_was_measured(*last_check_))
                    ctx.notify("Model loaded. GPU memory was not measured.");
                else
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
