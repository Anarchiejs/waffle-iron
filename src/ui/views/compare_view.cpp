#include "core/comparator.hpp"
#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include <atomic>
#include <cstdio>
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
    LoadParams loads_[2];
    char prompt_[8192] = "Explique le cycle de l'eau en quelques phrases.";
    char reference_[32768] = {};
    int max_tokens_ = 256;
    float temperature_ = 0;
    uint32_t seed_ = 42;
    bool perplexity_ = false;
    bool has_result_ = false;
    CompareResult result_;
    Engine* engine_ = nullptr;
    std::future<CompareResult> future_;
    std::shared_ptr<CompareProgress> progress_ = std::make_shared<CompareProgress>();
    static void metric(const char* label, double a, double b, bool higher, const char* unit, bool valid_a,
                       bool valid_b) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label);
        const double values[2] = {a, b};
        const bool valid[2] = {valid_a, valid_b};
        for (int i = 0; i < 2; ++i) {
            ImGui::TableNextColumn();
            bool best =
                valid[0] && valid[1] && a != b && (higher ? values[i] > values[1 - i] : values[i] < values[1 - i]);
            if (best) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.35f, .9f, .65f, 1));
            if (valid[i])
                ImGui::Text("%.2f %s%s", values[i], unit, best ? " *" : "");
            else
                ImGui::TextUnformatted("Non mesuré");
            if (best) ImGui::PopStyleColor();
        }
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
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        engine_ = &ctx.engine;
        if (is_ready(future_)) {
            try {
                result_ = future_.get();
                has_result_ = true;
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
            }
        }
        ImGui::TextWrapped("Les modèles sont chargés l'un après l'autre pour "
                           "respecter les 8 Go de VRAM.");
        ImGui::BeginDisabled(future_.valid());
        if (ImGui::BeginTable("réglages AB", 2, ImGuiTableFlags_Resizable)) {
            for (int i = 0; i < 2; ++i) {
                ImGui::TableNextColumn();
                ImGui::PushID(i);
                section_header(i ? "B" : "A");
                model_combo("Modèle", ctx, models_[i]);
                int layers = 100;
                if (models_[i] >= 0 && models_[i] < static_cast<int>(ctx.library.size())) {
                    const auto& m = ctx.library[static_cast<size_t>(models_[i])];
                    loads_[i].model_path = m.path;
                    layers = static_cast<int>(m.n_layer);
                }
                load_params_editor(loads_[i], layers);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::InputTextMultiline("Consigne", prompt_, sizeof(prompt_), ImVec2(-1, 90));
        ImGui::InputInt("Tokens max", &max_tokens_);
        ImGui::SliderFloat("Température", &temperature_, 0, 2);
        ImGui::InputScalar("Graine", ImGuiDataType_U32, &seed_);
        ImGui::Checkbox("Mesurer la perplexité", &perplexity_);
        if (perplexity_)
            ImGui::InputTextMultiline("Texte de référence", reference_, sizeof(reference_), ImVec2(-1, 180));
        if (!loads_[0].model_path.empty() && loads_[0] == loads_[1])
            ImGui::TextColored(ImVec4(1, .75f, .3f, 1), "Attention : les deux côtés sont identiques.");
        bool valid = models_[0] >= 0 && models_[1] >= 0 && models_[0] < static_cast<int>(ctx.library.size()) &&
                     models_[1] < static_cast<int>(ctx.library.size()) && max_tokens_ > 0 && max_tokens_ <= 100000 &&
                     (!perplexity_ || reference_[0]);
        ImGui::BeginDisabled(!valid);
        if (ImGui::Button("Lancer")) {
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
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        std::string texts[2], label;
        float fraction;
        {
            std::lock_guard lock(progress_->mutex);
            texts[0] = progress_->text[0];
            texts[1] = progress_->text[1];
            label = progress_->label;
            fraction = progress_->fraction;
        }
        if (future_.valid() && busy_indicator(label.c_str(), fraction)) {
            progress_->cancel = true;
            ctx.engine.cancel_all();
        }
        if (has_result_) {
            texts[0] = result_.a.text;
            texts[1] = result_.b.text;
        }
        if (ImGui::BeginTable("textes AB", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("A");
            ImGui::TableSetupColumn("B");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            for (int i = 0; i < 2; ++i) {
                ImGui::TableNextColumn();
                ImGui::PushID(i);
                ImGui::BeginChild("sortie", ImVec2(0, 240));
                ImGui::TextWrapped("%s", texts[i].c_str());
                ImGui::EndChild();
                if (has_result_) {
                    const auto& side = i ? result_.b : result_.a;
                    ImGui::TextUnformatted(side.model_name.c_str());
                    if (!side.error.empty()) ImGui::TextWrapped("Erreur : %s", side.error.c_str());
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (has_result_ && ImGui::BeginTable("mesures AB", 3, ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("Mesure (* : meilleur)");
            ImGui::TableSetupColumn("A");
            ImGui::TableSetupColumn("B");
            ImGui::TableHeadersRow();
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
    }
};
} // namespace
std::unique_ptr<View> make_compare_view() {
    return std::make_unique<CompareView>();
}
} // namespace banc::ui
