#include "core/bench.hpp"
#include "core/util.hpp"
#include "ui/icons.hpp"
#include "ui/theme.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <imgui.h>
#include <implot.h>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>

namespace waffle::ui {
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
        if (first == std::string::npos) throw Error("Empty list or missing value");
        out.push_back(part.substr(first, part.find_last_not_of(" \t\r\n") - first + 1));
    }
    if (out.empty() || std::string(text).back() == ',') throw Error("Empty list or missing value");
    return out;
}
static int integer(const std::string& s, int low, int high) {
    size_t end = 0;
    int v = std::stoi(s, &end);
    if (end != s.size() || v < low || v > high) throw Error("Value out of range: " + s);
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
// Splits configuration labels into a part shared by all of them and a part unique to each.
struct SplitLabels {
    std::string common;
    std::vector<std::string> rows;
};
SplitLabels split_labels(const std::vector<std::string>& labels) {
    std::vector<std::vector<std::string>> words;
    for (const auto& label : labels) {
        std::istringstream in(label);
        words.emplace_back(std::istream_iterator<std::string>(in), std::istream_iterator<std::string>());
    }
    SplitLabels out;
    if (words.size() < 2) {
        out.rows = labels;
        return out;
    }
    auto shared = [&](const std::string& w) {
        return std::all_of(words.begin(), words.end(),
                           [&](const auto& ws) { return std::find(ws.begin(), ws.end(), w) != ws.end(); });
    };
    for (const auto& w : words.front())
        if (shared(w)) out.common += (out.common.empty() ? "" : " ") + w;
    for (const auto& ws : words) {
        std::string row;
        for (const auto& w : ws)
            if (!shared(w)) row += (row.empty() ? "" : " ") + w;
        out.rows.push_back(row.empty() ? "(same)" : row);
    }
    return out;
}
class BenchView final : public View {
    int model_ = -1, seen_selection_ = -1;
    char gpu_[128] = "0,20,99", ctx_[128] = "4096", kv_[128] = "f16", fa_[128] = "1", ub_[128] = "512";
    char name_[256] = "Campaign", path_[2048] = {};
    int prompt_ = 512, gen_ = 128, reps_ = 5, warmup_ = 1;
    unsigned seed_ = 42;
    std::vector<int> also_;
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
        if (model_ < 0 || model_ >= static_cast<int>(ctx.library.size())) throw Error("Select a model");
        g.base.load.model_path = ctx.library[static_cast<size_t>(model_)].path;
        g.base.n_prompt = static_cast<uint32_t>(prompt_);
        g.base.n_gen = static_cast<uint32_t>(gen_);
        g.base.repetitions = reps_;
        g.base.warmup = std::max(0, warmup_);
        g.base.seed = seed_;
        for (int index : also_) {
            if (index < 0 || index >= static_cast<int>(ctx.library.size()) || index == model_) continue;
            g.models.push_back(ctx.library[static_cast<size_t>(index)].path);
        }
        if (!g.models.empty()) g.models.insert(g.models.begin(), g.base.load.model_path);
        if (prompt_ < 1 || gen_ < 1 || reps_ < 1)
            throw Error("Prompt, generation, and repetition counts must be positive");
        for (const auto& v : split_values(gpu_)) g.n_gpu_layers.push_back(integer(v, -1, 100000));
        for (const auto& v : split_values(ctx_)) g.n_ctx.push_back(static_cast<uint32_t>(integer(v, 32, 10000000)));
        g.kv_type = split_values(kv_);
        for (const auto& v : g.kv_type) (void)ggml_type_from_name(v);
        for (const auto& v : split_values(fa_)) g.flash_attn.push_back(integer(v, -1, 1));
        for (const auto& v : split_values(ub_)) g.n_ubatch.push_back(static_cast<uint32_t>(integer(v, 1, 1000000)));
        const size_t models = std::max<size_t>(1, g.models.size());
        const size_t count = std::max<size_t>(1, g.n_gpu_layers.size()) * std::max<size_t>(1, g.n_ctx.size()) *
                             std::max<size_t>(1, g.kv_type.size()) * std::max<size_t>(1, g.flash_attn.size()) *
                             std::max<size_t>(1, g.n_ubatch.size()) * models;
        if (count > 10000) throw Error("Limit: 10,000 configurations per campaign");
        return g;
    }
    void campaign(AppContext& ctx) {
        card_title("Campaign", "Several values per field, separated by commas: every combination "
                               "is measured.");
        ImGui::BeginDisabled(future_.valid());
        field_label("Model");
        model_combo("##model", ctx, model_);
        if (ImGui::Button("Also measure the selected model") && model_ >= 0 &&
            std::find(also_.begin(), also_.end(), model_) == also_.end())
            also_.push_back(model_);
        for (size_t i = 0; i < also_.size();) {
            const int index = also_[i];
            const char* label = index >= 0 && index < static_cast<int>(ctx.library.size())
                                    ? ctx.library[static_cast<size_t>(index)].name.c_str()
                                    : "?";
            ImGui::TextUnformatted(label);
            ImGui::SameLine();
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::SmallButton("Remove"))
                also_.erase(also_.begin() + static_cast<std::ptrdiff_t>(i));
            else
                ++i;
            ImGui::PopID();
        }
        field_label("Campaign name");
        ImGui::InputText("##name", name_, sizeof(name_));
        section_header("GRID");
        field_pair(
            [&](float w) {
                field_label("GPU layers", "-1, or a value at least the layer count: everything on the GPU.", w);
                ImGui::InputText("##gpu", gpu_, sizeof(gpu_));
            },
            [&](float w) {
                field_label("Contexts", nullptr, w);
                ImGui::InputText("##ctx", ctx_, sizeof(ctx_));
            });
        field_pair(
            [&](float w) {
                field_label("Types KV", "f16, q8_0, q4_0...", w);
                ImGui::InputText("##kv", kv_, sizeof(kv_));
            },
            [&](float w) {
                field_label("Flash attention", "-1: auto, 0: no, 1: yes", w);
                ImGui::InputText("##fa", fa_, sizeof(fa_));
            });
        field_label("Micro-batches (ubatch)");
        ImGui::InputText("##ub", ub_, sizeof(ub_));
        section_header("MEASUREMENT");
        field_pair(
            [&](float w) {
                field_label("Prompt tokens", nullptr, w);
                ImGui::InputInt("##pp", &prompt_, 0, 0);
            },
            [&](float w) {
                field_label("Generated tokens", nullptr, w);
                ImGui::InputInt("##tg", &gen_, 0, 0);
            });
        field_label("Repetitions", "Each configuration is measured this many times; standard deviation and outliers "
                                   "come from that.");
        ImGui::InputInt("##reps", &reps_, 0, 0);
        field_pair(
            [&](float w) {
                field_label("Warmup", "Discarded runs before the measured ones.", w);
                ImGui::InputInt("##warmup", &warmup_, 0, 0);
            },
            [&](float w) {
                field_label("Seed", nullptr, w);
                ImGui::InputScalar("##seed", ImGuiDataType_U32, &seed_);
            });
        ImGui::Dummy({0, em(.2f)});
        std::optional<CampaignGrid> parsed;
        try {
            parsed = grid(ctx);
            const auto n = expand_grid(*parsed).size();
            char text[64];
            std::snprintf(text, sizeof text, n == 1 ? "%zu configuration" : "%zu configurations", n);
            badge(text, col::accent);
        } catch (const std::exception& e) {
            ImGui::PushStyleColor(ImGuiCol_Text, col::bad);
            ImGui::TextWrapped("%s", e.what());
            ImGui::PopStyleColor();
        }
        // During the campaign, the progress bar replaces the button and stays in the same place.
        if (future_.valid()) {
            ImGui::EndDisabled(); // the Cancel button must stay enabled
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
            return;
        }
        ImGui::BeginDisabled(!parsed.has_value());
        if (primary_button("Run campaign", {-FLT_MIN, 0})) {
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
    }
    void results(AppContext& ctx) {
        if (output_.results.empty()) {
            empty_state("No results", future_.valid() ? "The campaign is running; results appear when it finishes."
                                                      : "Run a campaign, or load one from the history.");
            return;
        }
        if (primary_button("Save")) {
            ctx.store.save_campaign(name_, output_.results);
            refresh_ = true;
            ctx.notify("Campaign saved");
        }
        ImGui::SameLine();
        if (ImGui::Button("Copy as Markdown")) {
            ImGui::SetClipboardText(to_markdown(output_.results).c_str());
            ctx.notify("Table copied");
        }
        ImGui::SameLine();
        const float export_w = ImGui::CalcTextSize("CSV").x + ImGui::CalcTextSize("Markdown").x +
                               ImGui::GetStyle().FramePadding.x * 4 + ImGui::GetStyle().ItemSpacing.x * 2;
        if (ImGui::GetContentRegionAvail().x < em(14) + export_w) ImGui::NewLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - export_w);
        ImGui::InputTextWithHint("##export", "Export path", path_, sizeof(path_));
        for (int kind = 0; kind < 2; ++kind) {
            ImGui::SameLine();
            if (ImGui::Button(kind ? "Markdown" : "CSV")) {
                auto path = waffle::path_from_utf8(path_);
                path.replace_extension(kind ? ".md" : ".csv");
                if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
                std::ofstream file(path, std::ios::binary);
                file << (kind ? to_markdown(output_.results) : to_csv(output_.results));
                file.close();
                if (!file) throw Error("Cannot write the export");
                ctx.notify("Export saved");
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(kind ? "Export as Markdown" : "Export as CSV");
        }
        std::vector<std::string> full;
        full.reserve(output_.results.size());
        for (const auto& r : output_.results) full.push_back(config_label(r.config));
        const auto split = split_labels(full);
        if (!split.common.empty()) {
            ImGui::Dummy({0, em(.1f)});
            ImGui::TextDisabled("Shared:");
            ImGui::SameLine();
            push_font(fonts().mono, .88f);
            ImGui::TextWrapped("%s", split.common.c_str());
            ImGui::PopFont();
        }
        ImGui::Dummy({0, em(.1f)});
        const float avail_h = ImGui::GetContentRegionAvail().y;
        // Just the height of the rows, capped so the chart still has room.
        const float row_h = ImGui::GetTextLineHeight() + ImGui::GetStyle().CellPadding.y * 2;
        const float table_h = std::min(row_h * static_cast<float>(output_.results.size() + 1) + em(.3f),
                                       std::clamp(avail_h * .42f, em(8), em(20)));
        if (ImGui::BeginTable("runs", 7,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit,
                              {0, table_h})) {
            ImGui::TableSetupScrollFreeze(1, 1);
            ImGui::TableSetupColumn("#");
            ImGui::TableSetupColumn("Configuration", ImGuiTableColumnFlags_WidthStretch);
            for (const char* h : {"pp ± σ (tok/s)", "tg ± σ (tok/s)", "TTFT", "VRAM max", "Outliers"})
                ImGui::TableSetupColumn(h);
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::TableHeadersRow();
            ImGui::PopStyleColor();
            int index = 0;
            for (const auto& r : output_.results) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%d", ++index);
                ImGui::TableNextColumn();
                push_font(fonts().mono, .88f);
                ImGui::TextUnformatted(split.rows[static_cast<size_t>(index - 1)].c_str());
                ImGui::PopFont();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", full[static_cast<size_t>(index - 1)].c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.1f ± %.1f", r.pp.mean, r.pp.stddev);
                ImGui::TableNextColumn();
                ImGui::Text("%.1f ± %.1f", r.tg.mean, r.tg.stddev);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(format_duration_ms(r.ttft.mean).c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.vram_peak ? format_bytes(r.vram_peak).c_str() : "Not measured");
                ImGui::TableNextColumn();
                ImGui::Text("%zu / %zu / %zu", r.pp.outliers.size(), r.tg.outliers.size(), r.ttft.outliers.size());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Dropped outliers: pp / tg / TTFT");
            }
            ImGui::EndTable();
        }
        ImGui::Dummy({0, em(.1f)});
        std::vector<double> xp, xt, pp, tg, ep, et;
        std::vector<std::string> labels;
        std::vector<const char*> ticks;
        xp.reserve(output_.results.size());
        xt.reserve(output_.results.size());
        pp.reserve(output_.results.size());
        tg.reserve(output_.results.size());
        ep.reserve(output_.results.size());
        et.reserve(output_.results.size());
        labels.reserve(output_.results.size());
        ticks.reserve(output_.results.size());
        for (size_t i = 0; i < output_.results.size(); ++i) {
            const auto& r = output_.results[i];
            xp.push_back(static_cast<double>(i) - .18);
            xt.push_back(static_cast<double>(i) + .18);
            pp.push_back(r.pp.mean);
            tg.push_back(r.tg.mean);
            ep.push_back(r.pp.stddev);
            et.push_back(r.tg.stddev);
            labels.push_back("#" + std::to_string(i + 1));
        }
        for (const auto& label : labels) ticks.push_back(label.c_str());
        // Prefill (pp) and generation (tg) throughput: very different scales, two axes.
        if (ImPlot::BeginPlot("##Throughput", ImVec2(-1, std::max(em(12), ImGui::GetContentRegionAvail().y)),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
            ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoHighlight);
            ImPlot::SetupAxis(ImAxis_Y1, "Prefill (pp) tok/s", ImPlotAxisFlags_NoHighlight);
            ImPlot::SetupAxis(ImAxis_Y2, "Generation (tg) tok/s",
                              ImPlotAxisFlags_Opposite | ImPlotAxisFlags_NoHighlight);
            // Scales fixed on the highest point (bar + standard deviation) with a margin, so nothing is clipped.
            auto top = [](const std::vector<double>& v, const std::vector<double>& e) {
                double m = 0;
                for (size_t i = 0; i < v.size(); ++i) m = std::max(m, v[i] + e[i]);
                return m > 0 ? m * 1.12 : 1.;
            };
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, top(pp, ep), ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y2, 0, top(tg, et), ImPlotCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_North,
                                ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_NoButtons);
            ImPlot::SetupAxisLimits(ImAxis_X1, -.6, static_cast<double>(labels.size()) - .4, ImPlotCond_Always);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0., static_cast<double>(labels.size() - 1),
                                   static_cast<int>(ticks.size()), ticks.data());
            int n = static_cast<int>(pp.size());
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotBars("Prefill (pp)", xp.data(), pp.data(), n, .34);
            ImPlot::PlotErrorBars("Prefill (pp)", xp.data(), pp.data(), ep.data(), n);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotBars("Generation (tg)", xt.data(), tg.data(), n, .34);
            ImPlot::PlotErrorBars("Generation (tg)", xt.data(), tg.data(), et.data(), n);
            if (ImPlot::IsPlotHovered()) {
                const double x = ImPlot::GetPlotMousePos(ImAxis_X1, ImAxis_Y1).x;
                const long i = std::lround(x);
                if (i >= 0 && i < static_cast<long>(output_.results.size()))
                    ImGui::SetTooltip("#%ld  %s", i + 1,
                                      config_label(output_.results[static_cast<size_t>(i)].config).c_str());
            }
            ImPlot::EndPlot();
        }
    }
    void history(AppContext& ctx) {
        const float button = ImGui::CalcTextSize("Refresh").x + ImGui::GetStyle().FramePadding.x * 2;
        bool changed =
            model_combo("##history", ctx, history_model_,
                        std::min(em(26), ImGui::GetContentRegionAvail().x - button - ImGui::GetStyle().ItemSpacing.x));
        ImGui::SameLine();
        if (ImGui::Button("Refresh")) refresh_ = true;
        if (refresh_ || changed) {
            records_ = ctx.store.list_campaigns();
            history_.clear();
            if (history_model_ >= 0 && history_model_ < static_cast<int>(ctx.library.size()))
                history_ =
                    ctx.store.history(waffle::utf8_from_path(ctx.library[static_cast<size_t>(history_model_)].path));
            refresh_ = false;
        }
        if (records_.empty()) {
            empty_state("No saved campaign", "Campaigns saved from the Results tab "
                                             "appear here.");
            return;
        }
        // Just the height of the rows, capped so the chart still has room.
        const float row_h = ImGui::GetFrameHeight() + ImGui::GetStyle().CellPadding.y * 2;
        const float list_h =
            std::min(row_h * static_cast<float>(records_.size() + 1) + em(.3f),
                     history_.empty() ? ImGui::GetContentRegionAvail().y
                                      : std::clamp(ImGui::GetContentRegionAvail().y * .4f, em(8), em(18)));
        if (ImGui::BeginTable("campaigns", 5,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingFixedFit,
                              {0, list_h})) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Campaign", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Date");
            ImGui::TableSetupColumn("Results");
            ImGui::TableSetupColumn("Models", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("##actions");
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::TableHeadersRow();
            ImGui::PopStyleColor();
            for (const auto& record : records_) {
                ImGui::PushID(static_cast<int>(record.id));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(record.name.c_str());
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextDisabled("%s", record.created_at.c_str());
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::Text("%d", record.n_results);
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(record.model_names.c_str());
                ImGui::TableNextColumn();
                if (ImGui::SmallButton("Load")) {
                    try {
                        output_.results = ctx.store.load_campaign(record.id);
                        output_.errors.clear();
                        ctx.notify("Campaign loaded into Results");
                    } catch (const std::exception& e) {
                        ctx.notify(e.what(), true);
                    }
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Delete")) ImGui::OpenPopup("Confirm");
                if (ImGui::BeginPopup("Confirm")) {
                    ImGui::TextUnformatted("Delete this campaign?");
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(.55f, .22f, .25f, 1));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(.68f, .28f, .31f, 1));
                    if (ImGui::Button("Confirm deletion")) {
                        try {
                            ctx.store.delete_campaign(record.id);
                            refresh_ = true;
                            ImGui::CloseCurrentPopup();
                        } catch (const std::exception& e) {
                            ctx.notify(e.what(), true);
                        }
                    }
                    ImGui::PopStyleColor(2);
                    ImGui::EndPopup();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (history_.empty()) return;
        // A curve over time only makes sense from two distinct dates.
        double t_min = history_time(history_.front().timestamp), t_max = t_min;
        for (const auto& h : history_) {
            t_min = std::min(t_min, history_time(h.timestamp));
            t_max = std::max(t_max, history_time(h.timestamp));
        }
        if (t_max - t_min < 1) {
            ImGui::Dummy({0, em(.6f)});
            dim_wrapped("The trend over time appears once two campaigns have been saved for this "
                        "model, at different times.");
            return;
        }
        std::map<std::string, std::vector<size_t>> groups;
        for (size_t i = 0; i < history_.size(); ++i) groups[history_[i].config].push_back(i);
        std::vector<std::string> configs;
        configs.reserve(groups.size());
        for (const auto& group : groups) configs.push_back(group.first);
        const auto names = split_labels(configs); // short legend: only what changes
        if (ImPlot::BeginPlot("##History", ImVec2(-1, std::max(em(12), ImGui::GetContentRegionAvail().y)),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
            // Prefill (pp) on the left, generation (tg) on the right: on one axis, tg would be crushed against zero.
            ImPlot::SetupAxes(nullptr, "Prefill (pp) tok/s", ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoHighlight);
            ImPlot::SetupAxis(ImAxis_Y2, "Generation (tg) tok/s",
                              ImPlotAxisFlags_Opposite | ImPlotAxisFlags_NoHighlight);
            double pp_max = 1, tg_max = 1;
            for (const auto& h : history_) {
                pp_max = std::max(pp_max, h.pp_mean);
                tg_max = std::max(tg_max, h.tg_mean);
            }
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, pp_max * 1.15, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y2, 0, tg_max * 1.15, ImPlotCond_Always);
            // X values are seconds since the first measurement: in absolute seconds (~1.8e9), ImPlot loses the
            // segments between the points. The formatter shows the local time again.
            // Date precision follows the span (otherwise several ticks show the same minute).
            static struct {
                double origin;
                const char* format;
            } axis;
            const double span = t_max - t_min;
            axis = {t_min, span < 3600 ? "%H:%M:%S" : span < 3 * 86400 ? "%d/%m %H:%M" : "%d/%m/%y"};
            ImPlot::SetupAxisFormat(
                ImAxis_X1,
                [](double v, char* buf, int size, void* data) {
                    const auto* a = static_cast<decltype(axis)*>(data);
                    const auto t = static_cast<std::time_t>(a->origin + v);
                    std::tm tm{};
#ifdef _WIN32
                    localtime_s(&tm, &t);
#else
                    localtime_r(&t, &tm);
#endif
                    return static_cast<int>(std::strftime(buf, static_cast<size_t>(size), a->format, &tm));
                },
                &axis);
            ImPlot::SetupAxisLimits(ImAxis_X1, -span * .06, span * 1.06, ImPlotCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_North,
                                ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_NoButtons);
            size_t group_index = 0;
            for (const auto& [config, indices] : groups) {
                const std::string& name = names.rows[group_index++];
                std::vector<double> x, pp, tg, pp_err, tg_err;
                for (size_t i : indices) {
                    x.push_back(history_time(history_[i].timestamp) - t_min);
                    pp.push_back(history_[i].pp_mean);
                    tg.push_back(history_[i].tg_mean);
                    pp_err.push_back(history_[i].pp_stddev);
                    tg_err.push_back(history_[i].tg_stddev);
                }
                // ImPlot draws the points and the legend; the segments are drawn with ImGui, because the
                // long diagonal PlotLine segments come out dashed on this chart.
                ImPlotSpec spec;
                spec.Marker = ImPlotMarker_Circle;
                auto plot = [&](const std::string& label, const std::vector<double>& ys,
                                const std::vector<double>& err) {
                    ImPlot::PlotScatter(label.c_str(), x.data(), ys.data(), static_cast<int>(x.size()), spec);
                    ImPlot::PlotErrorBars((label + " sd").c_str(), x.data(), ys.data(), err.data(),
                                          static_cast<int>(x.size()));
                    const ImU32 color = ImGui::GetColorU32(ImPlot::GetLastItemColor());
                    ImPlot::PushPlotClipRect();
                    for (size_t k = 1; k < x.size(); ++k)
                        ImPlot::GetPlotDrawList()->AddLine(ImPlot::PlotToPixels(x[k - 1], ys[k - 1]),
                                                           ImPlot::PlotToPixels(x[k], ys[k]), color, em(.12f));
                    ImPlot::PopPlotClipRect();
                };
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                plot(name + " pp", pp, pp_err);
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                plot(name + " tg", tg, tg_err);
            }
            if (ImPlot::IsPlotHovered()) {
                double x = ImPlot::GetPlotMousePos(ImAxis_X1, ImAxis_Y1).x + t_min;
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
        const auto path = waffle::documents_directory() / "WaffleIron" / "campaign.csv";
        const auto utf8 = waffle::utf8_from_path(path);
        std::snprintf(path_, sizeof(path_), "%s", utf8.c_str());
    }
    ~BenchView() override {
        progress_->cancel = true;
        // On shutdown, finish before the shared services are destroyed.
        if (future_.valid()) {
            if (engine_) engine_->cancel_all();
            future_.wait();
        }
    }
    const char* title() const override { return "Bench"; }
    const char* subtitle() const override {
        return "Measure throughput, latency, and VRAM on a grid of configurations, then follow them over time.";
    }
    const char* icon() const override { return icon_bench; }
    void draw(AppContext& ctx) override {
        engine_ = &ctx.engine;
        if (ctx.selected_model != seen_selection_) {
            seen_selection_ = ctx.selected_model;
            if (!future_.valid() && ctx.selected_model >= 0) model_ = ctx.selected_model;
            if (history_model_ < 0 && ctx.selected_model >= 0) {
                history_model_ = ctx.selected_model;
                refresh_ = true;
            }
        }
        try {
            if (is_ready(future_)) {
                output_ = future_.get();
                ctx.notify("Campaign finished");
            }
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float gap = em(1);
            const bool wide = avail.x >= em(48);
            if (wide) {
                if (begin_card("##Campaign", {side_width(.3f, 20, 25), avail.y})) campaign(ctx);
                end_card();
                ImGui::SameLine(0, gap);
            } else {
                if (begin_card("##Campaign")) campaign(ctx);
                end_card();
            }
            if (begin_card("##Output", {0, wide ? avail.y : std::max(em(28), ImGui::GetContentRegionAvail().y)},
                           ImGuiWindowFlags_NoScrollbar)) {
                for (const auto& error : output_.errors) {
                    ImGui::PushStyleColor(ImGuiCol_Text, col::bad);
                    ImGui::TextWrapped("Failed configuration: %s", error.c_str());
                    ImGui::PopStyleColor();
                }
                if (ImGui::BeginTabBar("bench")) {
                    if (ImGui::BeginTabItem("Results")) {
                        ImGui::Dummy({0, em(.2f)});
                        try {
                            results(ctx);
                        } catch (const std::exception& e) {
                            ctx.notify(e.what(), true);
                        }
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("History")) {
                        ImGui::Dummy({0, em(.2f)});
                        try {
                            history(ctx);
                        } catch (const std::exception& e) {
                            ctx.notify(e.what(), true);
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
            end_card();
        } catch (const std::exception& e) {
            ctx.notify(e.what(), true);
        }
    }
};
} // namespace
std::unique_ptr<View> make_bench_view() {
    return std::make_unique<BenchView>();
}
} // namespace waffle::ui
