#include "core/sampler_lab.hpp"
#include "ui/theme.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
// Couleur d'un token selon sa probabilité au moment du choix : rouge (rare) vers vert (sûr).
ImU32 probability_color(float p) {
    p = std::sqrt(std::clamp(p, 0.f, 1.f));
    return ImGui::ColorConvertFloat4ToU32(ImVec4(1.f - .6f * p, .45f + .5f * p, .5f, 1));
}
class SamplerView final : public View {
    int model_ = -1, generate_n_ = 32, remaining_ = 0, seen_selection_ = -1;
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
    bool busy() const { return future_.valid() || remaining_ > 0; }
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
    void settings_panel(AppContext& ctx) {
        card_title("Consigne", "Le modèle est chargé au démarrage, puis chaque pas affiche les candidats.");
        ImGui::BeginDisabled(busy());
        field_label("Modèle");
        model_combo("##modele", ctx, model_);
        field_label("Texte de départ");
        ImGui::InputTextMultiline("##consigne", prompt_, sizeof(prompt_), ImVec2(-FLT_MIN, em(5.5f)));
        checkbox("Modèle de conversation", &chat_);
        help_marker("Applique le gabarit de conversation du modèle autour de la consigne.");
        if (collapsing("Réglages de chargement")) {
            const int layers = model_ >= 0 && model_ < static_cast<int>(ctx.library.size())
                                   ? static_cast<int>(ctx.library[static_cast<size_t>(model_)].n_layer)
                                   : 100;
            load_params_editor(load_, layers);
        }
        if (primary_button(started_ ? "Redémarrer" : "Démarrer", {-FLT_MIN, 0})) {
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

        ImGui::Dummy({0, em(.2f)});
        section_header("ÉCHANTILLONNAGE");
        bool changed = false;
        field_pair(
            [&](float w) {
                field_label("Température", nullptr, w);
                changed |= ImGui::SliderFloat("##temp", &settings_.temperature, 0, 2, "%.2f");
            },
            [&](float w) {
                field_label("Top-k", "0 = désactivé", w);
                changed |= ImGui::SliderInt("##topk", &settings_.top_k, 0, 200);
            });
        field_pair(
            [&](float w) {
                field_label("Top-p", "1 = désactivé", w);
                changed |= ImGui::SliderFloat("##topp", &settings_.top_p, 0, 1, "%.2f");
            },
            [&](float w) {
                field_label("Min-p", "0 = désactivé", w);
                changed |= ImGui::SliderFloat("##minp", &settings_.min_p, 0, 1, "%.2f");
            });
        section_header("PÉNALITÉS");
        field_pair(
            [&](float w) {
                field_label("Répétition", "1 = désactivée", w);
                changed |= ImGui::SliderFloat("##rep", &settings_.repeat_penalty, .1f, 2, "%.2f");
            },
            [&](float w) {
                field_label("Fenêtre", "Nombre de derniers tokens pris en compte.", w);
                changed |= ImGui::SliderInt("##fen", &settings_.penalty_last_n, 0, 4096);
            });
        field_pair(
            [&](float w) {
                field_label("Fréquence", nullptr, w);
                changed |= ImGui::SliderFloat("##freq", &settings_.frequency_penalty, -2, 2, "%.2f");
            },
            [&](float w) {
                field_label("Présence", nullptr, w);
                changed |= ImGui::SliderFloat("##pres", &settings_.presence_penalty, -2, 2, "%.2f");
            });
        field_label("Graine");
        changed |= ImGui::InputScalar("##graine", ImGuiDataType_U32, &settings_.seed);
        dirty_ |= changed && started_;
    }
    void controls(AppContext& ctx) {
        ImGui::BeginDisabled(!started_ || busy() || snapshot_.at_eog);
        if (primary_button("Pas suivant")) submit(ctx, 1);
        ImGui::SameLine();
        if (ImGui::Button("Générer")) remaining_ = std::clamp(generate_n_, 1, 10000);
        ImGui::SameLine(0, em(.3f));
        ImGui::SetNextItemWidth(em(4));
        ImGui::InputInt("##N", &generate_n_, 0, 0);
        ImGui::SameLine(0, em(.3f));
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("tokens");
        ImGui::EndDisabled();
        auto current = std::find_if(snapshot_.nodes.begin(), snapshot_.nodes.end(),
                                    [this](const LabNode& n) { return n.id == snapshot_.current; });
        const bool has_parent = current != snapshot_.nodes.end() && current->parent >= 0;
        ImGui::SameLine();
        ImGui::BeginDisabled(!has_parent || busy());
        if (ImGui::Button("Retour au parent") && has_parent) submit(ctx, 2, {}, current->parent);
        ImGui::EndDisabled();
        if (busy()) {
            ImGui::SameLine();
            const char* label = remaining_ > 0 ? "Génération..." : "Calcul des probabilités...";
            if (ImGui::GetContentRegionAvail().x < em(12)) ImGui::NewLine();
            if (busy_indicator(label)) {
                remaining_ = 0;
                dirty_ = false;
                ctx.engine.cancel_all();
            }
        } else if (snapshot_.at_eog) {
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            badge("fin de génération", col::amber);
        }
    }
    void generated_text() {
        card_title("Texte généré");
        // Légende du dégradé de couleur.
        {
            const char* legend = "probabilité :  faible";
            push_font(nullptr, .85f);
            const float lw = ImGui::CalcTextSize(legend).x + em(5) + ImGui::CalcTextSize("forte").x + em(1);
            const ImVec2 back = ImGui::GetCursorPos();
            ImGui::SetCursorPos({ImGui::GetWindowContentRegionMax().x - lw, ImGui::GetStyle().WindowPadding.y + em(.1f)});
            ImGui::TextDisabled("%s", legend);
            ImGui::SameLine();
            const ImVec2 p = ImGui::GetCursorScreenPos();
            auto* dl = ImGui::GetWindowDrawList();
            const float bw = em(4.5f), bh = em(.45f), y = p.y + (ImGui::GetTextLineHeight() - bh) * .5f;
            for (int i = 0; i < 24; ++i) {
                const float a = static_cast<float>(i) / 24, b = static_cast<float>(i + 1) / 24;
                dl->AddRectFilled({p.x + bw * a, y}, {p.x + bw * b, y + bh}, probability_color(a * a));
            }
            ImGui::Dummy({bw, ImGui::GetTextLineHeight()});
            ImGui::SameLine();
            ImGui::TextDisabled("forte");
            ImGui::PopFont();
            ImGui::SetCursorPos(back);
        }
        if (!started_) {
            empty_state("Aucune génération",
                        "Choisissez un modèle, écrivez une consigne puis cliquez sur Démarrer. Chaque pas montre les "
                        "tokens candidats et leur probabilité.");
            return;
        }
        std::vector<const LabNode*> path;
        int id = snapshot_.current;
        for (size_t guard = 0; guard < snapshot_.nodes.size(); ++guard) {
            auto it = std::find_if(snapshot_.nodes.begin(), snapshot_.nodes.end(),
                                   [id](const LabNode& n) { return n.id == id; });
            if (it == snapshot_.nodes.end() || it->parent < 0) break;
            path.push_back(&*it);
            id = it->parent;
        }
        if (path.empty()) {
            dim_wrapped("La consigne est prête : cliquez sur Pas suivant ou sur un candidat.");
            return;
        }
        ImGui::BeginChild("texte coloré", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 cursor = origin;
        const float right = origin.x + std::max(1.f, ImGui::GetContentRegionAvail().x - em(.5f));
        const float line_height = ImGui::GetTextLineHeightWithSpacing();
        auto* text_draw = ImGui::GetWindowDrawList();
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const ImU32 color = probability_color((*it)->p_final);
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
                    if (ImGui::IsWindowHovered() &&
                        ImGui::IsMouseHoveringRect(cursor, ImVec2(cursor.x + width, cursor.y + line_height)))
                        ImGui::SetTooltip("p = %.4f %%", 100. * (*it)->p_final);
                    cursor.x += width;
                }
                i += bytes;
            }
        }
        ImGui::Dummy(ImVec2(0, cursor.y - origin.y + line_height));
        if (generating_step_) ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
    }
    void candidates(AppContext& ctx) {
        card_title("Candidats", "Barre pâle : probabilité brute. Barre colorée : après échantillonnage. Cliquez pour "
                                "choisir.");
        if (snapshot_.candidates.empty()) {
            dim_wrapped(started_ ? "Aucun candidat à ce nœud." : "Les candidats apparaissent après le démarrage.");
            return;
        }
        ImGui::BeginDisabled(!started_ || busy() || snapshot_.at_eog);
        const float width = ImGui::GetContentRegionAvail().x;
        const float height = ImGui::GetFrameHeight();
        int rank = 0;
        for (const auto& c : snapshot_.candidates) {
            ImGui::PushID(c.token);
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            auto* draw = ImGui::GetWindowDrawList();
            const float r = em(.3f);
            // Trois colonnes : rang, barre (avec le texte du token), pourcentage.
            const float bar_x = pos.x + em(1.8f), bar_w = std::max(em(4), width - em(1.8f) - em(5.2f));
            draw->AddRectFilled({bar_x, pos.y}, {bar_x + bar_w, pos.y + height},
                                ImGui::GetColorU32(ImVec4(1, 1, 1, .03f)), r);
            draw->AddRectFilled({bar_x, pos.y}, {bar_x + bar_w * std::clamp(c.p_raw, 0.f, 1.f), pos.y + height},
                                IM_COL32(180, 190, 210, 45), r);
            draw->AddRectFilled({bar_x, pos.y}, {bar_x + bar_w * std::clamp(c.p_final, 0.f, 1.f), pos.y + height},
                                ImGui::GetColorU32(ImVec4(.34f, .37f, .68f, .9f)), r);
            ImGui::InvisibleButton("choisir", ImVec2(width, height));
            const bool hovered = ImGui::IsItemHovered();
            if (hovered) draw->AddRect(pos, {pos.x + width, pos.y + height}, ImGui::GetColorU32(col::accent), r);
            if (ImGui::IsItemClicked()) submit(ctx, 1, c.token);
            if (hovered)
                ImGui::SetTooltip("Token %d · logit %.3f\nBrut %.4f %% · Final %.4f %%", c.token, c.logit,
                                  100. * c.p_raw, 100. * c.p_final);
            const float ty = pos.y + (height - ImGui::GetTextLineHeight()) * .5f;
            char number[8];
            std::snprintf(number, sizeof number, "%d", ++rank);
            const float nw = ImGui::CalcTextSize(number).x;
            draw->AddText({bar_x - em(.5f) - nw, ty}, ImGui::GetColorU32(col::dim), number);
            const auto label = visible_piece(c.piece);
            draw->PushClipRect({bar_x, pos.y}, {bar_x + bar_w, pos.y + height}, true);
            draw->AddText(fonts().mono, em(1), {bar_x + em(.6f), ty}, ImGui::GetColorU32(ImGuiCol_Text),
                          label.c_str());
            draw->PopClipRect();
            char percent[24];
            std::snprintf(percent, sizeof percent, "%.2f %%", 100.f * c.p_final);
            const float pw = ImGui::CalcTextSize(percent).x;
            draw->AddText({pos.x + width - pw - em(.6f), ty}, ImGui::GetColorU32(col::text), percent);
            ImGui::PopID();
        }
        ImGui::EndDisabled();
    }

public:
    const char* title() const override { return "Laboratoire d'échantillonnage"; }
    const char* nav_title() const override { return "Échantillonnage"; }
    const char* subtitle() const override {
        return "Avancez token par token : probabilités avant et après échantillonnage, et branches alternatives.";
    }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        // Suit le modèle choisi dans la bibliothèque.
        if (ctx.selected_model != seen_selection_) {
            seen_selection_ = ctx.selected_model;
            if (!busy() && ctx.selected_model >= 0) model_ = ctx.selected_model;
        }
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

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float gap = em(1);
        const bool wide = avail.x >= em(48);
        if (wide) {
            if (begin_card("##Réglages", {side_width(.3f, 20, 25), avail.y})) settings_panel(ctx);
            end_card();
            ImGui::SameLine(0, gap);
            ImGui::BeginChild("##Atelier", {0, avail.y}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        } else {
            if (begin_card("##Réglages")) settings_panel(ctx);
            end_card();
        }
        controls(ctx);
        ImGui::Dummy({0, em(.1f)});
        const ImVec2 area = ImGui::GetContentRegionAvail();
        const float sy = ImGui::GetStyle().ItemSpacing.y;
        const float text_h = wide ? std::max(em(9), area.y * .38f) : em(14);
        if (begin_card("##Texte", {0, text_h}, ImGuiWindowFlags_NoScrollbar)) generated_text();
        end_card();
        const float rest = wide ? std::max(em(14), ImGui::GetContentRegionAvail().y) : em(20);
        const float row_w = ImGui::GetContentRegionAvail().x;
        if (row_w >= em(40)) {
            const float branches_w = std::clamp(row_w * .36f, em(14), em(22));
            if (begin_card("##Candidats", {row_w - branches_w - gap, rest})) candidates(ctx);
            end_card();
            ImGui::SameLine(0, gap);
            if (begin_card("##Branches", {branches_w, rest})) branches(ctx);
            end_card();
        } else {
            if (begin_card("##Candidats", {0, rest})) candidates(ctx);
            end_card();
            if (begin_card("##Branches", {0, em(14) - sy})) branches(ctx);
            end_card();
        }
        if (wide) ImGui::EndChild();
    }
    void branches(AppContext& ctx) {
        card_title("Branches", "Cliquez sur un nœud pour y revenir.");
        if (snapshot_.nodes.empty()) {
            dim_wrapped("L'arbre des choix apparaît ici.");
            return;
        }
        for (const auto& n : snapshot_.nodes)
            if (n.parent < 0) tree(ctx, n.id);
    }
};
} // namespace
std::unique_ptr<View> make_sampler_view() {
    return std::make_unique<SamplerView>();
}
} // namespace banc::ui
