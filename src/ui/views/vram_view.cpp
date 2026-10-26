#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include "core/gguf_reader.hpp"
#include "core/vram_planner.hpp"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <future>
#include <map>
#include <ctime>
#include <thread>

namespace banc::ui {
namespace {
std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return {bytes.begin(), bytes.end()};
}
VramPlanInput plan_input(const LoadParams& p) {
    return {p.n_gpu_layers, p.n_ctx, p.n_batch, p.n_ubatch, p.type_k, p.type_v, p.flash_attn != 0, p.n_seq_max, p.swa_full};
}
std::string config_label(const LoadParams& p) {
    return "ngl=" + std::to_string(p.n_gpu_layers) + " ctx=" + std::to_string(p.n_ctx) + " kv=" + p.type_k + "/" +
           p.type_v + " fa=" + std::to_string(p.flash_attn);
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
public:
    ~VramView() override {
        if (checking_.valid()) checking_.wait();
    }
    const char* title() const override { return "Planificateur de VRAM"; }
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
                ctx.notify("Vérification VRAM enregistrée");
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
            }
        }
        model_combo("Modèle", ctx, ctx.selected_model);
        const auto* selected = ctx.selected();
        if (!selected) {
            ImGui::TextDisabled("Sélectionnez un modèle dans la bibliothèque.");
            return;
        }
        const auto path = selected->path;
        if (!cache_.contains(path) && !reading_.valid() && path != failed_) {
            requested_ = path;
            reading_ = std::async(std::launch::async, [path] { return read_gguf(path, true); });
        }
        if (!cache_.contains(path)) {
            if (path == failed_) {
                if (ImGui::Button("Réessayer la lecture GGUF")) failed_.clear();
            } else
                ImGui::TextDisabled("Lecture des tenseurs en arrière-plan...");
            return;
        }
        const auto& model = cache_.at(path);
        params_.model_path = path;
        load_params_editor(params_, static_cast<int>(model.n_layer));
        const auto estimate = estimate_vram(model, plan_input(params_));
        const auto sample = ctx.telemetry.sample();
        const uint64_t capacity = sample ? sample->vram_total : 0;
        const uint64_t used = sample ? sample->vram_used : 0;
        ImGui::Text("Estimation GPU : %s   |   CPU : %s", format_bytes(estimate.total_gpu()).c_str(),
                    format_bytes(estimate.total_cpu()).c_str());
        if (ImPlot::BeginPlot("Répartition mémoire", {-1, 280})) {
            ImPlot::SetupAxes("Placement", "Go");
            ImPlot::SetupAxisLimits(ImAxis_X1, -.6, 1.6, ImPlotCond_Always);
            const double ymax =
                static_cast<double>(std::max({estimate.total_gpu() + used, estimate.total_cpu(), capacity})) / GiB;
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, std::max(1.0, ymax * 1.15), ImPlotCond_Always);
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
            ImPlot::PlotBarGroups(components, data, 4, 2, .65, 0, stacked);
            ImPlotSpec horizontal;
            horizontal.Flags = ImPlotInfLinesFlags_Horizontal;
            if (capacity) {
                const double limit = static_cast<double>(capacity) / GiB;
                ImPlot::PlotInfLines("VRAM totale", &limit, 1, horizontal);
            }
            if (sample) {
                const double occupied = static_cast<double>(used) / GiB;
                ImPlot::PlotInfLines("VRAM déjà utilisée (tous processus)", &occupied, 1, horizontal);
            }
            ImPlot::EndPlot();
        }
        if (!sample) ImGui::TextWrapped("NVML indisponible : capacité et vérification réelle inconnues.");
        ImGui::BeginDisabled(!sample);
        if (ImGui::Button("Couches max qui tiennent"))
            params_.n_gpu_layers =
                max_gpu_layers_for(model, plan_input(params_), capacity > used ? capacity - used : 0);
        ImGui::EndDisabled();
        for (const auto& note : estimate.notes) ImGui::TextWrapped("• %s", note.c_str());
        ImGui::TextWrapped("La vérification décharge le modèle courant, mesure un chargement isolé puis le décharge. "
                           "L'estimation peut différer selon le backend compilé.");
#ifndef BANC_HAS_CUDA
        ImGui::TextWrapped("Build CPU : le chargement utilise le CPU ; la mesure ne valide pas une estimation CUDA.");
#endif
        ImGui::BeginDisabled(checking_.valid() || ctx.engine.state() != EngineState::Idle || !sample);
        if (ImGui::Button("Vérifier réellement")) {
            auto* telemetry = &ctx.telemetry;
            auto* store = &ctx.store;
            const auto params = params_;
            const auto estimated = estimate.total_gpu();
            checking_ = ctx.engine.submit([telemetry, store, params, estimated](EngineSession& session) {
                session.unload();
                // Laisser NVML observer la libération avant de prendre la référence.
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                auto before = telemetry->sample();
                if (!before) throw Error("Mesure NVML indisponible avant chargement");
                try {
                    session.load(params);
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    auto after = telemetry->sample();
                    if (!after) throw Error("Mesure NVML indisponible après chargement");
                    VramCheck result;
                    result.model_path = path_utf8(params.model_path);
                    result.config = config_label(params);
                    result.estimated_gpu = estimated;
                    result.measured_gpu =
                        after->vram_used > before->vram_used ? after->vram_used - before->vram_used : 0;
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
        ImGui::EndDisabled();
        if (checking_.valid() && busy_indicator("Vérification en cours...")) ctx.engine.cancel_all();
        if (last_check_ && last_check_->model_path == path_utf8(path)) {
            const auto& check = *last_check_;
            ImGui::Text("Mesuré : %s", format_bytes(check.measured_gpu).c_str());
            if (check.estimated_gpu)
                ImGui::Text("Écart (mesure - estimation) : %+.1f %%",
                            100 * (static_cast<double>(check.measured_gpu) / check.estimated_gpu - 1));
        }
        if (history_path_ != path_utf8(path)) {
            history_ = ctx.store.vram_checks(path_utf8(path));
            history_path_ = path_utf8(path);
        }
        if (ImGui::CollapsingHeader("Historique des vérifications")) {
            ImGui::TextWrapped("Mesure NVML globale : les autres processus peuvent influencer le résultat.");
            for (const auto& check : history_)
                ImGui::TextWrapped("%s | %s | estimé %s | mesuré %s", check.timestamp.c_str(), check.config.c_str(),
                                   format_bytes(check.estimated_gpu).c_str(), format_bytes(check.measured_gpu).c_str());
        }
    }
};
} // namespace
std::unique_ptr<View> make_vram_view() {
    return std::make_unique<VramView>();
}
} // namespace banc::ui
