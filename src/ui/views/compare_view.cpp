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

namespace banc::ui {
namespace {
// Texte original neutre : même référence pour les deux modèles, librement
// éditable.
static const char* reference_text =
    R"REF(Dans une petite commune située près d'une rivière, la bibliothèque municipale prépare chaque année un programme consacré à la découverte du territoire. Ce programme rassemble des promenades, des lectures publiques et des ateliers ouverts aux habitants. Les organisateurs commencent par consulter les associations, les établissements scolaires et les personnes qui fréquentent les lieux. Ils cherchent à connaître les horaires disponibles, les sujets qui intéressent le public et les obstacles qui pourraient empêcher certaines personnes de participer. Les réponses sont ensuite réunies dans un document simple, présenté lors d'une réunion où chacun peut proposer des ajustements. La préparation demande surtout de la régularité et une répartition claire des responsabilités. Une équipe vérifie les parcours, une autre rassemble les documents et une troisième prépare les informations pratiques. Les promenades suivent des chemins accessibles qui relient plusieurs endroits du village. Au départ, un guide explique la durée prévue et rappelle les consignes utiles. Le groupe avance ensuite à un rythme modéré, avec des pauses pour observer le paysage. Près du pont, les participants comparent des photographies anciennes avec les bâtiments actuels. Certaines façades ont changé, tandis que les dimensions des rues restent reconnaissables. Cette comparaison permet de comprendre comment les usages évoluent sans que tous les repères disparaissent. Plus loin, une ancienne cour accueille désormais un jardin partagé. Les habitants y cultivent des légumes et des plantes aromatiques dans des parcelles de tailles différentes. Un tableau indique les périodes de plantation, les besoins en eau et les tâches communes. Les jardiniers expliquent que les résultats varient selon la météo, la qualité du sol et le temps consacré aux cultures. Ils prennent des notes pour comparer les saisons, mais évitent de tirer des conclusions à partir d'une seule récolte. Dans la bibliothèque, une exposition présente des cartes, des lettres et des objets prêtés par des familles. Chaque objet porte une étiquette qui précise son origine et les informations connues. Lorsque la date ou la fonction demeure incertaine, cette incertitude est indiquée. Les visiteurs peuvent compléter une fiche pour partager leurs souvenirs, qui seront ensuite comparés avec les autres sources disponibles. Cette méthode demande parfois de conserver plusieurs interprétations plutôt que de choisir immédiatement une explication. Les ateliers accueillent aussi des enfants qui réalisent une carte collective du quartier. Ils dessinent les lieux où ils aiment se retrouver et ceux qu'ils connaissent moins bien. Les adultes les aident à distinguer une observation personnelle d'une information vérifiable. Un trajet peut paraître long à une personne et court à une autre, tandis que sa distance peut être mesurée avec un outil adapté. Les deux types de description ont leur utilité lorsque leur sens est clairement expliqué. À la fin de la matinée, une pause permet aux participants de discuter autour de grandes tables. Certains évoquent des changements récents, comme l'ouverture d'un commerce ou la modification d'un arrêt de transport. D'autres racontent des habitudes plus anciennes. Les animateurs veillent à laisser une place aux personnes qui parlent peu et à reformuler les questions lorsque cela facilite la discussion. Le but consiste à recueillir des points de vue variés dans un cadre respectueux. Après les activités, les équipes rangent le matériel et rédigent un compte rendu. Elles comptent les participants, décrivent les difficultés rencontrées et notent les propositions reçues. Ces informations servent à préparer la prochaine édition. Une fréquentation plus faible ne signifie pas nécessairement que le thème était moins intéressant, car les conditions météorologiques, les horaires ou un autre événement peuvent avoir joué un rôle. Pour comparer les éditions, il faut donc garder une trace de ces circonstances. Le document final est conservé avec les programmes et les photographies dont la diffusion a été autorisée. Il reste consultable sur place et peut être résumé dans le bulletin communal. Les organisateurs privilégient des phrases simples pour que chacun puisse comprendre les décisions prises et les possibilités de participation. Ils précisent également les questions qui restent ouvertes, notamment les moyens de transport et les besoins d'accompagnement. Au fil du temps, ces archives forment une description progressive de la vie locale. Elles montrent des actions modestes, des essais et des corrections successives. Leur intérêt vient de la continuité des observations et de la diversité des personnes qui y contribuent. La bibliothèque devient ainsi un lieu où les informations circulent, où les souvenirs sont discutés et où de nouveaux projets peuvent prendre forme.)REF";
struct CompareProgress {
    std::mutex mutex;
    std::atomic_bool cancel{false};
    std::string text[2];
    std::string label = "Préparation";
    float fraction = 0;
};
class CompareView final : public View {
    int models_[2] = {-1, -1};
    int seen_selection_ = -1;
    LoadParams loads_[2];
    char prompt_[8192] = "Explique le cycle de l'eau en quelques phrases.";
    char reference_[32768] = {};
    int max_tokens_ = 256;
    float temperature_ = 0;
    uint32_t seed_ = 42;
    bool perplexity_ = false;
    bool has_result_ = false;
    float metrics_h_ = 0; // hauteur de la carte des mesures, mise à jour à chaque image
    bool fold_settings_ = false; // replie les réglages au lancement, pour laisser la place aux résultats
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
                ImGui::TextDisabled("Non mesuré");
            if (best) {
                ImGui::PopStyleColor();
                ImGui::SameLine();
                badge("meilleur", col::good);
            }
        }
    }
    void side_config(AppContext& ctx, int i) {
        ImGui::PushID(i);
        badge(i ? "B" : "A", side_color(i));
        ImGui::SameLine();
        push_font(fonts().semibold, 1.06f);
        ImGui::TextUnformatted(i ? "Modèle B" : "Modèle A");
        ImGui::PopFont();
        field_label("Modèle");
        model_combo("##modele", ctx, models_[i]);
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
        card_title("Essai", "Les modèles sont chargés l'un après l'autre pour tenir dans la VRAM.");
        field_label("Consigne");
        ImGui::InputTextMultiline("##consigne", prompt_, sizeof(prompt_), ImVec2(-FLT_MIN, em(5)));
        field_pair(
            [&](float w) {
                field_label("Tokens max", nullptr, w);
                ImGui::InputInt("##max", &max_tokens_, 0, 0);
            },
            [&](float w) {
                field_label("Graine", nullptr, w);
                ImGui::InputScalar("##graine", ImGuiDataType_U32, &seed_);
            });
        field_label("Température", "0 : décodage glouton, reproductible.");
        ImGui::SliderFloat("##temp", &temperature_, 0, 2, "%.2f");
        checkbox("Mesurer la perplexité", &perplexity_);
        help_marker("Perplexité de chaque modèle sur un texte de référence commun (plus bas = mieux).");
        if (perplexity_) {
            field_label("Texte de référence");
            ImGui::InputTextMultiline("##reference", reference_, sizeof(reference_), ImVec2(-FLT_MIN, em(8)));
        }
        if (!loads_[0].model_path.empty() && loads_[0] == loads_[1]) {
            ImGui::PushStyleColor(ImGuiCol_Text, col::amber);
            ImGui::TextWrapped("Attention : les deux côtés sont identiques.");
            ImGui::PopStyleColor();
        }
        const bool valid = models_[0] >= 0 && models_[1] >= 0 && models_[0] < static_cast<int>(ctx.library.size()) &&
                           models_[1] < static_cast<int>(ctx.library.size()) && max_tokens_ > 0 &&
                           max_tokens_ <= 100000 && (!perplexity_ || reference_[0]);
        ImGui::Dummy({0, em(.1f)});
        ImGui::BeginDisabled(!valid || future_.valid());
        if (primary_button("Lancer la comparaison", {-FLT_MIN, 0})) launch(ctx);
        ImGui::EndDisabled();
        if (!valid && !future_.valid()) {
            push_font(nullptr, .88f);
            dim_wrapped("Choisissez un modèle pour A et pour B.");
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
        if (begin_card("##sortie", size, ImGuiWindowFlags_NoScrollbar)) {
            badge(i ? "B" : "A", side_color(i));
            ImGui::SameLine();
            const int index = models_[i];
            const std::string name = has_result_ ? (i ? result_.b : result_.a).model_name
                                     : index >= 0 && index < static_cast<int>(ctx.library.size())
                                         ? model_title(ctx.library[static_cast<size_t>(index)])
                                         : std::string("Aucun modèle");
            push_font(fonts().semibold, 1.0f);
            ImGui::TextUnformatted(name.c_str());
            ImGui::PopFont();
            if (has_result_) {
                const auto& side = i ? result_.b : result_.a;
                if (!side.error.empty()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, col::bad);
                    ImGui::TextWrapped("Erreur : %s", side.error.c_str());
                    ImGui::PopStyleColor();
                }
            }
            ImGui::BeginChild("##texte", {0, 0}, ImGuiChildFlags_FrameStyle);
            if (text.empty())
                ImGui::TextDisabled(future_.valid() ? "En attente..." : "La réponse du modèle s'affichera ici.");
            else
                ImGui::TextWrapped("%s", text.c_str());
            if (future_.valid() && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - em(1)) ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
        }
        end_card();
        ImGui::PopID();
    }
    void metrics() {
        card_title("Mesures");
        if (!ImGui::BeginTable("mesures AB", 3,
                               ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchSame))
            return;
        ImGui::TableSetupColumn("Mesure");
        ImGui::TableSetupColumn("A");
        ImGui::TableSetupColumn("B");
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TableHeadersRow();
        ImGui::PopStyleColor();
        const auto& a = result_.a;
        const auto& b = result_.b;
        bool va = a.error.empty(), vb = b.error.empty();
        metric("Lecture consigne", a.pp_tps, b.pp_tps, true, "tok/s", va && a.pp_tps > 0, vb && b.pp_tps > 0);
        metric("Génération", a.tg_tps, b.tg_tps, true, "tok/s", va && a.tg_tps > 0, vb && b.tg_tps > 0);
        metric("Chargement", a.load_ms, b.load_ms, false, "ms", va, vb);
        metric("VRAM max", static_cast<double>(a.vram_peak) / MiB, static_cast<double>(b.vram_peak) / MiB, false,
               "Mo", va && a.vram_peak > 0, vb && b.vram_peak > 0);
        metric("Perplexité", a.perplexity.value_or(0), b.perplexity.value_or(0), false, "",
               va && a.perplexity.has_value(), vb && b.perplexity.has_value());
        metric("Tokens générés", a.n_generated, b.n_generated, true, "tokens", va, vb);
        ImGui::EndTable();
    }

public:
    CompareView() { std::snprintf(reference_, sizeof(reference_), "%s", reference_text); }
    ~CompareView() override {
        progress_->cancel = true;
        // À l'arrêt, terminer avant la destruction des services partagés.
        if (future_.valid()) {
            if (engine_) engine_->cancel_all();
            future_.wait();
        }
    }
    const char* title() const override { return "Comparateur"; }
    const char* subtitle() const override {
        return "Deux modèles ou deux réglages, une même consigne : réponses et mesures côte à côte.";
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
            if (begin_card("##Essai", {side_width(.27f, 19, 24), avail.y})) common(ctx);
            end_card();
            ImGui::SameLine(0, gap);
            ImGui::BeginChild("##Duel", {0, avail.y}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        } else {
            if (begin_card("##Essai")) common(ctx);
            end_card();
        }
        if (future_.valid() && busy_indicator(label.c_str(), fraction)) {
            progress_->cancel = true;
            ctx.engine.cancel_all();
        }
        // Réglages A et B côte à côte (repliables pour laisser la place aux réponses).
        const float row = ImGui::GetContentRegionAvail().x;
        const bool two = row >= em(30);
        const float half = two ? std::floor((row - gap) * .5f) : row;
        ImGui::SetNextItemOpen(!has_result_, ImGuiCond_Once);
        if (fold_settings_) {
            ImGui::SetNextItemOpen(false);
            fold_settings_ = false;
        }
        if (collapsing("Réglages des modèles A et B", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginDisabled(future_.valid());
            for (int i = 0; i < 2; ++i) {
                if (i && two) ImGui::SameLine(0, gap);
                ImGui::PushID(i);
                if (begin_card("##réglages", {half, 0})) side_config(ctx, i);
                end_card();
                ImGui::PopID();
            }
            ImGui::EndDisabled();
        }
        // Hauteur de la carte des mesures mesurée à l'image précédente (estimation à la première).
        const float metrics_h = has_result_ ? (metrics_h_ > 0 ? metrics_h_ : em(17)) + ImGui::GetStyle().ItemSpacing.y : 0;
        const float out_h = std::max(em(12), ImGui::GetContentRegionAvail().y - metrics_h);
        for (int i = 0; i < 2; ++i) {
            if (i && two) ImGui::SameLine(0, gap);
            output(ctx, i, texts[i], {half, out_h});
        }
        if (has_result_) {
            if (begin_card("##Mesures")) metrics();
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
} // namespace banc::ui
