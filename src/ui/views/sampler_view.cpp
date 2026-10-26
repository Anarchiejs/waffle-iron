#include "core/sampler_lab.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <imgui.h>

namespace banc::ui {
namespace {
static std::string visible_piece(const std::string& piece) {
    std::string result;
    for (char c : piece) {
        if (c == ' ')
            result += "␣";
        else if (c == '\n')
            result += "↵";
        else if (c == '\r')
            result += "␍";
        else if (c == '\t')
            result += "⇥";
        else
            result += c;
    }
    return result;
}
class SamplerView final : public View {
    int model_ = -1, generate_n_ = 32, remaining_ = 0;
    LoadParams load_;
    LoadParams active_load_;
    SamplerSettings settings_;
    char prompt_[8192] = "Décris un paysage au printemps.";
    bool chat_ = true, started_ = false, dirty_ = false;
    std::shared_ptr<SamplerLab> lab_ = std::make_shared<SamplerLab>();
    std::future<LabSnapshot> future_;
    LabSnapshot snapshot_;
    bool generating_step_ = false;
    std::chrono::steady_clock::time_point step_time_;
    // Tout accès au laboratoire est effectué sur le thread du moteur.
    void submit(AppContext& ctx, int operation, std::optional<int32_t> token = {}, int node = 0) {
        auto lab = lab_;
        auto settings = settings_;
        auto load = active_load_;
        std::string prompt = prompt_;
        bool chat = chat_;
        dirty_ = false;
        generating_step_ = operation == 1 && remaining_ > 0;
        step_time_ = std::chrono::steady_clock::now();
        future_ =
            ctx.engine.submit([lab, settings, load, prompt, chat, operation, token, node](EngineSession& session) {
                session.load(load);
                if (operation == 0)
                    lab->start(session, prompt, chat);
                else if (operation == 1)
                    lab->step(session, settings, token);
                else if (operation == 2)
                    lab->go_to(session, node);
                return lab->snapshot(session, settings);
            });
    }
    void tree(AppContext& ctx, int id) {
        const auto found =
            std::find_if(snapshot_.nodes.begin(), snapshot_.nodes.end(), [id](const LabNode& n) { return n.id == id; });
        if (found == snapshot_.nodes.end()) return;
        const auto& n = *found;
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (n.children.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
        if (n.id == snapshot_.current) flags |= ImGuiTreeNodeFlags_Selected;
        ImGui::PushID(n.id);
        std::string label = n.parent < 0 ? "Consigne" : visible_piece(n.piece);
        bool expanded = ImGui::TreeNodeEx("branche", flags, "%s", label.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() && !future_.valid()) {
            remaining_ = 0;
            submit(ctx, 2, {}, n.id);
        }
        if (expanded) {
            for (int child : n.children) tree(ctx, child);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

public:
    const char* title() const override { return "Laboratoire d'échantillonnage"; }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        if (is_ready(future_)) {
            try {
                snapshot_ = future_.get();
                started_ = true;
                if (generating_step_) {
                    double seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - step_time_).count();
                    ctx.tps_history.emplace_back(ImGui::GetTime(), 1.0 / std::max(seconds, .000001));
                    if (ctx.tps_history.size() > 6000) ctx.tps_history.pop_front();
                    --remaining_;
                }
                if (snapshot_.at_eog) remaining_ = 0;
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
                remaining_ = 0;
            }
        }
        if (!future_.valid() && started_) {
            if (dirty_)
                submit(ctx, 3);
            else if (remaining_ > 0)
                submit(ctx, 1);
        }
        ImGui::BeginDisabled(future_.valid() || remaining_ > 0);
        model_combo("Modèle", ctx, model_);
        if (ImGui::CollapsingHeader("Réglages de chargement")) {
            const int layers = model_ >= 0 && model_ < static_cast<int>(ctx.library.size())
                                   ? static_cast<int>(ctx.library[static_cast<size_t>(model_)].n_layer)
                                   : 100;
            load_params_editor(load_, layers);
        }
        ImGui::InputTextMultiline("Consigne", prompt_, sizeof(prompt_), ImVec2(-1, 90));
        ImGui::Checkbox("Modèle de conversation", &chat_);
        if (ImGui::Button("Démarrer")) {
            if (model_ < 0 || model_ >= static_cast<int>(ctx.library.size()))
                ctx.notify("Sélectionnez un modèle", true);
            else {
                active_load_ = load_;
                active_load_.model_path = ctx.library[static_cast<size_t>(model_)].path;
                lab_ = std::make_shared<SamplerLab>();
                started_ = false;
                snapshot_ = {};
                submit(ctx, 0);
            }
        }
        ImGui::EndDisabled();
        bool changed = false;
        changed |= ImGui::SliderFloat("Température", &settings_.temperature, 0, 2);
        changed |= ImGui::SliderInt("Top-k", &settings_.top_k, 0, 200);
        changed |= ImGui::SliderFloat("Top-p", &settings_.top_p, 0, 1);
        changed |= ImGui::SliderFloat("Min-p", &settings_.min_p, 0, 1);
        changed |= ImGui::SliderFloat("Pénalité de répétition", &settings_.repeat_penalty, .1f, 2);
        changed |= ImGui::SliderFloat("Pénalité de fréquence", &settings_.frequency_penalty, -2, 2);
        changed |= ImGui::SliderFloat("Pénalité de présence", &settings_.presence_penalty, -2, 2);
        changed |= ImGui::SliderInt("Fenêtre des pénalités", &settings_.penalty_last_n, 0, 4096);
        changed |= ImGui::InputScalar("Graine", ImGuiDataType_U32, &settings_.seed);
        dirty_ |= changed && started_;
        if (future_.valid() || remaining_ > 0) {
            if (busy_indicator("Calcul des probabilités / génération")) {
                remaining_ = 0;
                dirty_ = false;
                ctx.engine.cancel_all();
            }
        }
        ImGui::BeginDisabled(!started_ || future_.valid() || remaining_ > 0);
        ImGui::BeginDisabled(snapshot_.at_eog);
        if (ImGui::Button("Pas suivant")) submit(ctx, 1);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("N", &generate_n_);
        ImGui::SameLine();
        if (ImGui::Button("Générer N")) remaining_ = std::clamp(generate_n_, 1, 10000);
        ImGui::EndDisabled();
        auto current = std::find_if(snapshot_.nodes.begin(), snapshot_.nodes.end(),
                                    [this](const LabNode& n) { return n.id == snapshot_.current; });
        if (current != snapshot_.nodes.end() && current->parent >= 0 && ImGui::Button("Retour au parent"))
            submit(ctx, 2, {}, current->parent);
        ImGui::EndDisabled();
        section_header("Candidats : brut (pâle), après échantillonnage (accent)");
        ImGui::BeginDisabled(!started_ || future_.valid() || remaining_ > 0 || snapshot_.at_eog);
        for (const auto& c : snapshot_.candidates) {
            ImGui::PushID(c.token);
            ImVec2 pos = ImGui::GetCursorScreenPos();
            float width = std::max(100.f, ImGui::GetContentRegionAvail().x);
            float height = ImGui::GetFrameHeight();
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(pos, ImVec2(pos.x + width * std::clamp(c.p_raw, 0.f, 1.f), pos.y + height),
                                IM_COL32(180, 190, 210, 65));
            draw->AddRectFilled(pos, ImVec2(pos.x + width * std::clamp(c.p_final, 0.f, 1.f), pos.y + height),
                                ImGui::GetColorU32(ImGuiCol_PlotHistogram));
            ImGui::InvisibleButton("choisir", ImVec2(width, height));
            if (ImGui::IsItemClicked()) submit(ctx, 1, c.token);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Token %d · logit %.3f\nBrut %.4f %% · Final %.4f %%", c.token, c.logit,
                                  100. * c.p_raw, 100. * c.p_final);
            const auto label = visible_piece(c.piece) + "  (" + std::to_string(100.f * c.p_final) + " %)";
            draw->AddText(ImVec2(pos.x + 4, pos.y + ImGui::GetStyle().FramePadding.y),
                          ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
            ImGui::PopID();
        }
        ImGui::EndDisabled();
        section_header("Texte généré (couleur = probabilité au moment du choix)");
        std::vector<const LabNode*> path;
        int id = snapshot_.current;
        for (size_t guard = 0; guard < snapshot_.nodes.size(); ++guard) {
            auto it = std::find_if(snapshot_.nodes.begin(), snapshot_.nodes.end(),
                                   [id](const LabNode& n) { return n.id == id; });
            if (it == snapshot_.nodes.end() || it->parent < 0) break;
            path.push_back(&*it);
            id = it->parent;
        }
        ImGui::BeginChild("texte coloré", ImVec2(0, 160));
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 cursor = origin;
        const float right = origin.x + std::max(1.f, ImGui::GetContentRegionAvail().x);
        const float line_height = ImGui::GetTextLineHeight();
        auto* text_draw = ImGui::GetWindowDrawList();
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            float p = std::sqrt(std::clamp((*it)->p_final, 0.f, 1.f));
            const ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(1.f - .6f * p, .45f + .5f * p, .5f, 1));
            const auto& piece = (*it)->piece;
            for (size_t i = 0; i < piece.size();) {
                unsigned char c = static_cast<unsigned char>(piece[i]);
                size_t bytes = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
                bytes = std::min(bytes, piece.size() - i);
                const char* begin = piece.data() + i;
                const char* end = begin + bytes;
                if (c == '\n') {
                    cursor.x = origin.x;
                    cursor.y += line_height;
                } else if (c != '\r') {
                    float width = ImGui::CalcTextSize(begin, end).x;
                    if (cursor.x + width > right && cursor.x > origin.x) {
                        cursor.x = origin.x;
                        cursor.y += line_height;
                    }
                    text_draw->AddText(cursor, color, begin, end);
                    if (ImGui::IsMouseHoveringRect(cursor, ImVec2(cursor.x + width, cursor.y + line_height)))
                        ImGui::SetTooltip("p = %.4f %%", 100. * (*it)->p_final);
                    cursor.x += width;
                }
                i += bytes;
            }
        }
        ImGui::Dummy(ImVec2(0, cursor.y - origin.y + line_height));
        ImGui::EndChild();
        section_header("Branches");
        for (const auto& n : snapshot_.nodes)
            if (n.parent < 0) tree(ctx, n.id);
    }
};
} // namespace
std::unique_ptr<View> make_sampler_view() {
    return std::make_unique<SamplerView>();
}
} // namespace banc::ui
