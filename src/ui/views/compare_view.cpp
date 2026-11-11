#include "core/comparator.hpp"
#include "ui/theme.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <imgui.h>
#include <mutex>

namespace waffle::ui {
namespace {
// Neutral source text: the same reference for both models, freely
// editable.
static const char* reference_text =
    R"REF(In a small town beside a river, the municipal library prepares a yearly program devoted to discovering the local area. The program brings together walks, public readings, and workshops open to residents. The organizers begin by consulting associations, schools, and the people who use those places. They want to know which times are free, which subjects interest the public, and which obstacles might keep some people from taking part. The answers are then gathered into a simple document, presented at a meeting where anyone can suggest adjustments. Preparation mostly takes regularity and a clear split of responsibilities. One team checks the routes, another collects the documents, and a third prepares the practical information. The walks follow accessible paths that link several places in the village. At the start, a guide explains how long the walk should take and repeats the useful instructions. The group then moves at a moderate pace, with pauses to look at the landscape. Near the bridge, participants compare old photographs with the buildings as they stand now. Some facades have changed, while the width of the streets is still recognizable. The comparison shows how uses change without every landmark disappearing. Farther on, an old courtyard is now a shared garden. Residents grow vegetables and herbs in plots of different sizes. A board lists planting times, water needs, and shared tasks. The gardeners explain that results vary with the weather, the soil, and the time spent on the crops. They take notes in order to compare seasons, but they avoid drawing conclusions from a single harvest. Inside the library, an exhibition shows maps, letters, and objects lent by families. Each object carries a label with its origin and the known facts. When the date or the purpose is still uncertain, that uncertainty is written down. Visitors can fill in a card to share a memory, which is then compared with the other sources. The method sometimes means keeping several interpretations instead of picking one explanation at once. The workshops also welcome children who draw a collective map of the neighborhood. They mark the places where they like to meet and the ones they know less well. Adults help them separate a personal observation from a fact that can be checked. A route can feel long to one person and short to another, while its distance can be measured with a suitable tool. Both descriptions are useful when their meaning is stated clearly. At the end of the morning, a break lets people talk around large tables. Some mention recent changes, such as a shop opening or a change to a transit stop. Others describe older habits. The hosts leave room for people who speak little, and they rephrase a question when that helps. The aim is to collect varied points of view in a respectful setting. After the activities, the teams put the equipment away and write a short report. They count the participants, describe the difficulties they met, and note the proposals they received. That information is used to prepare the next edition. A smaller attendance does not by itself mean the theme was less interesting, because the weather, the schedule, or another event may have played a part. Comparing editions therefore means keeping a record of those circumstances. The final document is stored with the programs and with the photographs that were cleared for publication. It can be read on site and summarized in the town bulletin. The organizers prefer plain sentences so that everyone can understand the decisions and the ways to take part. They also state the questions that remain open, notably transport and the need for assistance. Over time these archives become a gradual description of local life. They show modest actions, trials, and successive corrections. Their value comes from the continuity of the observations and from the variety of people who contribute. The library becomes a place where information circulates, where memories are discussed, and where new projects can take shape.)REF";
struct CompareProgress {
    std::mutex mutex;
    std::atomic_bool cancel{false};
    std::string text[2];
    std::string label = "Setup";
    float fraction = 0;
};
class CompareView final : public View {
    int models_[2] = {-1, -1};
    int seen_selection_ = -1;
    LoadParams loads_[2];
    char prompt_[8192] = "Explain the water cycle in a few sentences.";
    char reference_[32768] = {};
    int max_tokens_ = 256;
    float temperature_ = 0;
    uint32_t seed_ = 42;
    bool perplexity_ = false;
    bool has_result_ = false;
    float metrics_h_ = 0;        // height of the measurement card, updated every frame
    bool fold_settings_ = false; // folds the settings on launch, to leave room for the results
    CompareResult result_;
    Engine* engine_ = nullptr;
    std::future<CompareResult> future_;
    std::shared_ptr<CompareProgress> progress_ = std::make_shared<CompareProgress>();
    static constexpr ImVec4 side_color(int i) { return i ? col::teal : col::accent; }
    static void metric(const char* label, double a, double b, bool higher, const char* unit, bool valid_a,
                       bool valid_b) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", label);
        const double values[2] = {a, b};
        const bool valid[2] = {valid_a, valid_b};
        for (int i = 0; i < 2; ++i) {
            ImGui::TableNextColumn();
            bool best =
                valid[0] && valid[1] && a != b && (higher ? values[i] > values[1 - i] : values[i] < values[1 - i]);
            if (best) ImGui::PushStyleColor(ImGuiCol_Text, col::good);
            if (valid[i])
                ImGui::Text(std::strcmp(unit, "tokens") ? "%.2f %s" : "%.0f %s", values[i], unit);
            else
                ImGui::TextDisabled("Not measured");
            if (best) {
                ImGui::PopStyleColor();
                ImGui::SameLine();
                badge("best", col::good);
            }
        }
    }
    void side_config(AppContext& ctx, int i) {
        ImGui::PushID(i);
        badge(i ? "B" : "A", side_color(i));
        ImGui::SameLine();
        push_font(fonts().semibold, 1.06f);
        ImGui::TextUnformatted(i ? "Model B" : "Model A");
        ImGui::PopFont();
        field_label("Model");
        model_combo("##model", ctx, models_[i]);
        int layers = 100;
        if (models_[i] >= 0 && models_[i] < static_cast<int>(ctx.library.size())) {
            const auto& m = ctx.library[static_cast<size_t>(models_[i])];
            loads_[i].model_path = m.path;
            layers = static_cast<int>(m.n_layer);
        }
        load_params_editor(loads_[i], layers);
        ImGui::PopID();
    }
    void common(AppContext& ctx) {
        card_title("Trial", "The models are loaded one after the other so they fit in VRAM.");
        field_label("Prompt");
        ImGui::InputTextMultiline("##prompt", prompt_, sizeof(prompt_), ImVec2(-FLT_MIN, em(5)));
        field_pair(
            [&](float w) {
                field_label("Tokens max", nullptr, w);
                ImGui::InputInt("##max", &max_tokens_, 0, 0);
            },
            [&](float w) {
                field_label("Seed", nullptr, w);
                ImGui::InputScalar("##seed", ImGuiDataType_U32, &seed_);
            });
        field_label("Temperature", "0: greedy decoding, reproducible.");
        ImGui::SliderFloat("##temp", &temperature_, 0, 2, "%.2f");
        checkbox("Measure perplexity", &perplexity_);
        help_marker("Each model's perplexity on a shared reference text (lower is better).");
        if (perplexity_) {
            field_label("Reference text");
            ImGui::InputTextMultiline("##reference", reference_, sizeof(reference_), ImVec2(-FLT_MIN, em(8)));
        }
        if (!loads_[0].model_path.empty() && loads_[0] == loads_[1]) {
            ImGui::PushStyleColor(ImGuiCol_Text, col::amber);
            ImGui::TextWrapped("Warning: both sides are identical.");
            ImGui::PopStyleColor();
        }
        const bool valid = models_[0] >= 0 && models_[1] >= 0 && models_[0] < static_cast<int>(ctx.library.size()) &&
                           models_[1] < static_cast<int>(ctx.library.size()) && max_tokens_ > 0 &&
                           max_tokens_ <= 100000 && (!perplexity_ || reference_[0]);
        ImGui::Dummy({0, em(.1f)});
        ImGui::BeginDisabled(!valid || future_.valid());
        if (primary_button("Run comparison", {-FLT_MIN, 0})) launch(ctx);
        ImGui::EndDisabled();
        if (!valid && !future_.valid()) {
            push_font(nullptr, .88f);
            dim_wrapped("Choose a model for A and for B.");
            ImGui::PopFont();
        }
    }
    void launch(AppContext& ctx) {
        CompareRequest req;
        req.a = {"A", loads_[0]};
        req.b = {"B", loads_[1]};
        req.prompt = prompt_;
        req.max_tokens = max_tokens_;
        req.temperature = temperature_;
        req.seed = seed_;
        if (perplexity_) req.perplexity_text = reference_;
        progress_ = std::make_shared<CompareProgress>();
        has_result_ = false;
        auto p = progress_;
        auto* engine = &ctx.engine;
        auto* telemetry = &ctx.telemetry;
        fold_settings_ = true;
        future_ = std::async(std::launch::async, [engine, telemetry, req, p] {
            return run_compare(
                *engine, telemetry, req,
                [p](int side, const std::string& piece) {
                    if (side < 0 || side > 1) return;
                    std::lock_guard lock(p->mutex);
                    p->text[side] += piece;
                },
                [p](float fraction, const std::string& label) {
                    std::lock_guard lock(p->mutex);
                    p->fraction = fraction;
                    p->label = label;
                    return !p->cancel.load();
                });
        });
    }
    void output(AppContext& ctx, int i, const std::string& text, ImVec2 size) {
        ImGui::PushID(i);
        if (begin_card("##output", size, ImGuiWindowFlags_NoScrollbar)) {
            badge(i ? "B" : "A", side_color(i));
            ImGui::SameLine();
            const int index = models_[i];
            const std::string name = has_result_ ? (i ? result_.b : result_.a).model_name
                                     : index >= 0 && index < static_cast<int>(ctx.library.size())
                                         ? model_title(ctx.library[static_cast<size_t>(index)])
                                         : std::string("No model");
            push_font(fonts().semibold, 1.0f);
            ImGui::TextUnformatted(name.c_str());
            ImGui::PopFont();
            if (has_result_) {
                const auto& side = i ? result_.b : result_.a;
                if (!side.error.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, col::bad);
                    ImGui::TextWrapped("Error: %s", side.error.c_str());
                    ImGui::PopStyleColor();
                }
            }
            ImGui::BeginChild("##text", {0, 0}, ImGuiChildFlags_FrameStyle);
            if (text.empty())
                ImGui::TextDisabled(future_.valid() ? "Waiting..." : "The model reply will appear here.");
            else
                ImGui::TextWrapped("%s", text.c_str());
            if (future_.valid() && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - em(1)) ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
        }
        end_card();
        ImGui::PopID();
    }
    void metrics() {
        card_title("Measurements");
        if (!ImGui::BeginTable("measures AB", 3,
                               ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                   ImGuiTableFlags_SizingStretchSame))
            return;
        ImGui::TableSetupColumn("Measurement");
        ImGui::TableSetupColumn("A");
        ImGui::TableSetupColumn("B");
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        const auto& a = result_.a;
        const auto& b = result_.b;
        bool va = a.error.empty(), vb = b.error.empty();
        metric("Prompt prefill", a.pp_tps, b.pp_tps, true, "tok/s", va && a.pp_tps > 0, vb && b.pp_tps > 0);
        metric("Generation", a.tg_tps, b.tg_tps, true, "tok/s", va && a.tg_tps > 0, vb && b.tg_tps > 0);
        metric("Load", a.load_ms, b.load_ms, false, "ms", va, vb);
        metric("VRAM max", static_cast<double>(a.vram_peak) / MiB, static_cast<double>(b.vram_peak) / MiB, false, "Mo",
               va && a.vram_peak > 0, vb && b.vram_peak > 0);
        metric("Perplexity", a.perplexity.value_or(0), b.perplexity.value_or(0), false, "",
               va && a.perplexity.has_value(), vb && b.perplexity.has_value());
        metric("Generated tokens", a.n_generated, b.n_generated, true, "tokens", va, vb);
        ImGui::EndTable();
    }

public:
    CompareView() { std::snprintf(reference_, sizeof(reference_), "%s", reference_text); }
    ~CompareView() override {
        progress_->cancel = true;
        // On shutdown, finish before the shared services are destroyed.
        if (future_.valid()) {
            if (engine_) engine_->cancel_all();
            future_.wait();
        }
    }
    const char* title() const override { return "Comparator"; }
    const char* subtitle() const override {
        return "Two models or two settings, one prompt: replies and measurements side by side.";
    }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        engine_ = &ctx.engine;
        if (ctx.selected_model != seen_selection_) {
            seen_selection_ = ctx.selected_model;
            if (!future_.valid() && models_[0] < 0 && ctx.selected_model >= 0) models_[0] = ctx.selected_model;
        }
        if (is_ready(future_)) {
            try {
                result_ = future_.get();
                has_result_ = true;
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
            }
        }
        std::string texts[2], label;
        float fraction;
        {
            std::lock_guard lock(progress_->mutex);
            texts[0] = progress_->text[0];
            texts[1] = progress_->text[1];
            label = progress_->label;
            fraction = progress_->fraction;
        }
        if (has_result_) {
            texts[0] = result_.a.text;
            texts[1] = result_.b.text;
        }

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float gap = em(1);
        const bool wide = avail.x >= em(50);
        if (wide) {
            if (begin_card("##Trial", {side_width(.27f, 19, 24), avail.y})) common(ctx);
            end_card();
            ImGui::SameLine(0, gap);
            ImGui::BeginChild("##Compare", {0, avail.y}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        } else {
            if (begin_card("##Trial")) common(ctx);
            end_card();
        }
        if (future_.valid() && busy_indicator(label.c_str(), fraction)) {
            progress_->cancel = true;
            ctx.engine.cancel_all();
        }
        // Settings for A and B side by side (collapsible, to leave room for the replies).
        const float row = ImGui::GetContentRegionAvail().x;
        const bool two = row >= em(30);
        const float half = two ? std::floor((row - gap) * .5f) : row;
        ImGui::SetNextItemOpen(!has_result_, ImGuiCond_Once);
        if (fold_settings_) {
            ImGui::SetNextItemOpen(false);
            fold_settings_ = false;
        }
        if (collapsing("Settings for models A and B", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginDisabled(future_.valid());
            for (int i = 0; i < 2; ++i) {
                if (i && two) ImGui::SameLine(0, gap);
                ImGui::PushID(i);
                if (begin_card("##settings", {half, 0})) side_config(ctx, i);
                end_card();
                ImGui::PopID();
            }
            ImGui::EndDisabled();
        }
        // Height of the measurement card measured on the previous frame (an estimate on the first).
        const float metrics_h =
            has_result_ ? (metrics_h_ > 0 ? metrics_h_ : em(17)) + ImGui::GetStyle().ItemSpacing.y : 0;
        const float out_h = std::max(em(12), ImGui::GetContentRegionAvail().y - metrics_h);
        for (int i = 0; i < 2; ++i) {
            if (i && two) ImGui::SameLine(0, gap);
            output(ctx, i, texts[i], {half, out_h});
        }
        if (has_result_) {
            if (begin_card("##Measurements")) metrics();
            end_card();
            metrics_h_ = ImGui::GetItemRectSize().y;
        }
        if (wide) ImGui::EndChild();
    }
};
} // namespace
std::unique_ptr<View> make_compare_view() {
    return std::make_unique<CompareView>();
}
} // namespace waffle::ui
