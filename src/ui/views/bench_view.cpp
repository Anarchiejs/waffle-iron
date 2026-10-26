#include "core/bench.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <imgui.h>
#include <implot.h>
#include <map>
#include <mutex>
#include <sstream>

namespace banc::ui {
namespace {
struct CampaignOutput {
    std::vector<BenchResult> results;
    std::vector<std::string> errors;
};
struct CampaignProgress {
    std::mutex mutex;
    std::atomic_bool cancel{false};
    float fraction = 0;
    std::string label;
};
static std::vector<std::string> split_values(const char* text) {
    std::vector<std::string> out;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        const auto first = part.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) throw Error("Liste vide ou valeur manquante");
        out.push_back(part.substr(first, part.find_last_not_of(" \t\r\n") - first + 1));
    }
    if (out.empty() || std::string(text).back() == ',') throw Error("Liste vide ou valeur manquante");
    return out;
}
static int integer(const std::string& s, int low, int high) {
    size_t end = 0;
    int v = std::stoi(s, &end);
    if (end != s.size() || v < low || v > high) throw Error("Valeur hors limites : " + s);
    return v;
}
static double history_time(const std::string& stamp) {
    int y = 0, m = 0, d = 0, h = 0, minute = 0, sec = 0;
    if (std::sscanf(stamp.c_str(), "%d-%d-%dT%d:%d:%d", &y, &m, &d, &h, &minute, &sec) < 3) return 0;
    const std::chrono::year_month_day date{std::chrono::year{y}, std::chrono::month{static_cast<unsigned>(m)},
                                           std::chrono::day{static_cast<unsigned>(d)}};
    if (!date.ok()) return 0;
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::seconds>(std::chrono::sys_days(date).time_since_epoch())
                   .count()) +
           h * 3600. + minute * 60. + sec;
}
class BenchView final : public View {
    int model_ = -1;
    char gpu_[128] = "0,20,99", ctx_[128] = "4096", kv_[128] = "f16", fa_[128] = "1", ub_[128] = "512";
    char name_[256] = "Campagne", path_[2048] = {};
    int prompt_ = 512, gen_ = 128, reps_ = 5;
    Engine* engine_ = nullptr;
    std::future<CampaignOutput> future_;
    std::shared_ptr<CampaignProgress> progress_ = std::make_shared<CampaignProgress>();
    CampaignOutput output_;
    std::vector<CampaignRecord> records_;
    std::vector<HistoryPoint> history_;
    bool refresh_ = true;
    int history_model_ = -2;
    CampaignGrid grid(AppContext& ctx) {
        CampaignGrid g;
        if (model_ < 0 || model_ >= static_cast<int>(ctx.library.size())) throw Error("Sélectionnez un modèle");
        g.base.load.model_path = ctx.library[static_cast<size_t>(model_)].path;
        g.base.n_prompt = static_cast<uint32_t>(prompt_);
        g.base.n_gen = static_cast<uint32_t>(gen_);
        g.base.repetitions = reps_;
        if (prompt_ < 1 || gen_ < 1 || reps_ < 1) throw Error("Mesures et répétitions doivent être positives");
        for (const auto& v : split_values(gpu_)) g.n_gpu_layers.push_back(integer(v, -1, 100000));
        for (const auto& v : split_values(ctx_)) g.n_ctx.push_back(static_cast<uint32_t>(integer(v, 32, 10000000)));
        g.kv_type = split_values(kv_);
        for (const auto& v : g.kv_type) (void)ggml_type_from_name(v);
        for (const auto& v : split_values(fa_)) g.flash_attn.push_back(integer(v, -1, 1));
        for (const auto& v : split_values(ub_)) g.n_ubatch.push_back(static_cast<uint32_t>(integer(v, 1, 1000000)));
        const size_t count =
            g.n_gpu_layers.size() * g.n_ctx.size() * g.kv_type.size() * g.flash_attn.size() * g.n_ubatch.size();
        if (count > 10000) throw Error("Limite : 10 000 configurations par campagne");
        return g;
    }
    void results(AppContext& ctx) {
        if (output_.results.empty()) {
            ImGui::TextUnformatted("Aucun résultat");
            return;
        }
        if (ImGui::Button("Enregistrer")) {
            ctx.store.save_campaign(name_, output_.results);
            refresh_ = true;
            ctx.notify("Campagne enregistrée");
        }
        ImGui::SameLine();
        if (ImGui::Button("Copier en Markdown")) ImGui::SetClipboardText(to_markdown(output_.results).c_str());
        ImGui::InputText("Chemin d'export", path_, sizeof(path_));
        for (int kind = 0; kind < 2; ++kind) {
            if (kind) ImGui::SameLine();
            if (ImGui::Button(kind ? "Exporter Markdown" : "Exporter CSV")) {
                auto path = std::filesystem::u8path(path_);
                path.replace_extension(kind ? ".md" : ".csv");
                if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
                std::ofstream file(path, std::ios::binary);
                file << (kind ? to_markdown(output_.results) : to_csv(output_.results));
                file.close();
                if (!file) throw Error("Impossible d'écrire l'export");
                ctx.notify("Export enregistré");
            }
        }
        if (ImGui::BeginTable("mesures", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX)) {
            for (const char* h :
                 {"Configuration", "pp ± σ (tok/s)", "tg ± σ (tok/s)", "TTFT", "VRAM max", "Aberrations"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            for (const auto& r : output_.results) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(config_label(r.config).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.2f ± %.2f", r.pp.mean, r.pp.stddev);
                ImGui::TableNextColumn();
                ImGui::Text("%.2f ± %.2f", r.tg.mean, r.tg.stddev);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(format_duration_ms(r.ttft.mean).c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.vram_peak ? format_bytes(r.vram_peak).c_str() : "Non mesurée");
                ImGui::TableNextColumn();
                ImGui::Text("pp: %zu / tg: %zu / ttft: %zu", r.pp.outliers.size(), r.tg.outliers.size(),
                            r.ttft.outliers.size());
            }
            ImGui::EndTable();
        }
        std::vector<double> xp, xt, pp, tg, ep, et;
        std::vector<std::string> labels;
        std::vector<const char*> ticks;
        for (size_t i = 0; i < output_.results.size(); ++i) {
            const auto& r = output_.results[i];
            xp.push_back(static_cast<double>(i) - .18);
            xt.push_back(static_cast<double>(i) + .18);
            pp.push_back(r.pp.mean);
            tg.push_back(r.tg.mean);
            ep.push_back(r.pp.stddev);
            et.push_back(r.tg.stddev);
            labels.push_back(config_label(r.config));
        }
        for (const auto& label : labels) ticks.push_back(label.c_str());
        if (ImPlot::BeginPlot("Débits", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Configuration", "tokens/s");
            ImPlot::SetupAxisTicks(ImAxis_X1, 0., static_cast<double>(labels.size() - 1),
                                   static_cast<int>(ticks.size()), ticks.data());
            int n = static_cast<int>(pp.size());
            ImPlot::PlotBars("pp", xp.data(), pp.data(), n, .34);
            ImPlot::PlotBars("tg", xt.data(), tg.data(), n, .34);
            ImPlot::PlotErrorBars("σ pp", xp.data(), pp.data(), ep.data(), n);
            ImPlot::PlotErrorBars("σ tg", xt.data(), tg.data(), et.data(), n);
            ImPlot::EndPlot();
        }
    }
    void history(AppContext& ctx) {
        bool changed = model_combo("Modèle historique", ctx, history_model_);
        if (ImGui::Button("Actualiser")) refresh_ = true;
        if (refresh_ || changed) {
            records_ = ctx.store.list_campaigns();
            history_.clear();
            if (history_model_ >= 0 && history_model_ < static_cast<int>(ctx.library.size()))
                history_ = ctx.store.history(ctx.library[static_cast<size_t>(history_model_)].path.string());
            refresh_ = false;
        }
        for (const auto& record : records_) {
            ImGui::PushID(static_cast<int>(record.id));
            ImGui::Text("%s · %s · %d résultats · %s", record.name.c_str(), record.created_at.c_str(), record.n_results,
                        record.model_names.c_str());
            if (ImGui::Button("Charger")) {
                try {
                    output_.results = ctx.store.load_campaign(record.id);
                    output_.errors.clear();
                } catch (const std::exception& e) {
                    ctx.notify(e.what(), true);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Supprimer")) ImGui::OpenPopup("Confirmer");
            if (ImGui::BeginPopup("Confirmer")) {
                ImGui::TextUnformatted("Supprimer cette campagne ?");
                if (ImGui::Button("Confirmer la suppression")) {
                    try {
                        ctx.store.delete_campaign(record.id);
                        refresh_ = true;
                        ImGui::CloseCurrentPopup();
                    } catch (const std::exception& e) {
                        ctx.notify(e.what(), true);
                    }
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        if (!history_.empty() && ImPlot::BeginPlot("Historique des débits", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Date UTC", "tokens/s");
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
            std::map<std::string, std::vector<size_t>> groups;
            for (size_t i = 0; i < history_.size(); ++i) groups[history_[i].config].push_back(i);
            for (const auto& [config, indices] : groups) {
                std::vector<double> x, pp, tg;
                for (size_t i : indices) {
                    x.push_back(history_time(history_[i].timestamp));
                    pp.push_back(history_[i].pp_mean);
                    tg.push_back(history_[i].tg_mean);
                }
                ImPlot::PlotLine((config + " pp").c_str(), x.data(), pp.data(), static_cast<int>(x.size()));
                ImPlot::PlotLine((config + " tg").c_str(), x.data(), tg.data(), static_cast<int>(x.size()));
            }
            if (ImPlot::IsPlotHovered()) {
                double x = ImPlot::GetPlotMousePos().x;
                size_t i = 0;
                for (size_t j = 1; j < history_.size(); ++j)
                    if (std::abs(history_time(history_[j].timestamp) - x) <
                        std::abs(history_time(history_[i].timestamp) - x))
                        i = j;
                if (i < history_.size())
                    ImGui::SetTooltip("%s\n%s\nllama.cpp : %s", history_[i].timestamp.c_str(),
                                      history_[i].config.c_str(), history_[i].llama_version.c_str());
            }
            ImPlot::EndPlot();
        }
    }

public:
    BenchView() {
        const char* home = std::getenv("USERPROFILE");
        auto path = (home ? std::filesystem::path(home) : std::filesystem::current_path()) / "Documents" / "BancLocal" /
                    "campagne.csv";
        auto utf8 = path.u8string();
        std::snprintf(path_, sizeof(path_), "%s", reinterpret_cast<const char*>(utf8.c_str()));
    }
    ~BenchView() override {
        progress_->cancel = true;
        // À l'arrêt, terminer avant la destruction des services partagés.
        if (future_.valid()) {
            if (engine_) engine_->cancel_all();
            future_.wait();
        }
    }
    const char* title() const override { return "Banc d'essai"; }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        engine_ = &ctx.engine;
        try {
            if (is_ready(future_)) {
                output_ = future_.get();
                ctx.notify("Campagne terminée");
            }
            if (future_.valid()) {
                std::string label;
                float fraction;
                {
                    std::lock_guard lock(progress_->mutex);
                    label = progress_->label;
                    fraction = progress_->fraction;
                }
                if (busy_indicator(label.c_str(), fraction)) {
                    progress_->cancel = true;
                    ctx.engine.cancel_all();
                }
            }
            if (ImGui::BeginTabBar("banc")) {
                if (ImGui::BeginTabItem("Campagne")) {
                    ImGui::BeginDisabled(future_.valid());
                    model_combo("Modèle", ctx, model_);
                    ImGui::InputText("Nom", name_, sizeof(name_));
                    ImGui::InputText("Couches GPU", gpu_, sizeof(gpu_));
                    ImGui::InputText("Contextes", ctx_, sizeof(ctx_));
                    ImGui::InputText("Types KV", kv_, sizeof(kv_));
                    ImGui::InputText("Flash attention (-1,0,1)", fa_, sizeof(fa_));
                    ImGui::InputText("Ubatch", ub_, sizeof(ub_));
                    ImGui::InputInt("Tokens consigne", &prompt_);
                    ImGui::InputInt("Tokens génération", &gen_);
                    ImGui::InputInt("Répétitions", &reps_);
                    std::optional<CampaignGrid> parsed;
                    try {
                        parsed = grid(ctx);
                        ImGui::Text("%zu configurations", expand_grid(*parsed).size());
                    } catch (const std::exception& e) {
                        ImGui::TextWrapped("Grille invalide : %s", e.what());
                    }
                    ImGui::BeginDisabled(!parsed.has_value());
                    if (ImGui::Button("Lancer")) {
                        try {
                            progress_ = std::make_shared<CampaignProgress>();
                            auto p = progress_;
                            auto* engine = &ctx.engine;
                            auto* telemetry = &ctx.telemetry;
                            future_ = std::async(std::launch::async, [p, engine, telemetry, g = *parsed] {
                                CampaignOutput out;
                                out.results = run_campaign(
                                    *engine, telemetry, g,
                                    [p](float f, const std::string& label) {
                                        std::lock_guard lock(p->mutex);
                                        p->fraction = f;
                                        p->label = label;
                                        return !p->cancel.load();
                                    },
                                    &out.errors);
                                return out;
                            });
                        } catch (const std::exception& e) {
                            ctx.notify(e.what(), true);
                        }
                    }
                    ImGui::EndDisabled();
                    ImGui::EndDisabled();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Résultats")) {
                    try {
                        results(ctx);
                    } catch (const std::exception& e) {
                        ctx.notify(e.what(), true);
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Historique")) {
                    try {
                        history(ctx);
                    } catch (const std::exception& e) {
                        ctx.notify(e.what(), true);
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            for (const auto& error : output_.errors) ImGui::TextWrapped("Configuration en échec : %s", error.c_str());
        } catch (const std::exception& e) {
            ctx.notify(e.what(), true);
        }
    }
};
} // namespace
std::unique_ptr<View> make_bench_view() {
    return std::make_unique<BenchView>();
}
} // namespace banc::ui
