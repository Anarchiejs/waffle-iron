#include "ui/view.hpp"
#include "ui/widgets.hpp"
#include "core/gguf_reader.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <future>
#include <numeric>
#include <sstream>

namespace banc::ui {
namespace {
std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return {bytes.begin(), bytes.end()};
}
struct ScanResult {
    std::vector<ModelInfo> models;
    std::vector<std::string> errors;
};
class LibraryView final : public View {
public:
    bool rescan = false;
    const char* title() const override { return "Bibliothèque"; }
    const char* icon() const override { return ""; }
    void draw(AppContext& ctx) override {
        if (!initialized_) {
            for (const auto& r : ctx.store.list_models()) {
                ModelInfo m;
                m.path = std::filesystem::u8path(r.path);
                m.file_name = r.file_name;
                m.name = r.name;
                m.architecture = r.architecture;
                m.file_type = r.file_type;
                m.file_size = r.file_size;
                m.n_params = r.n_params;
                m.context_length = r.context_length;
                m.n_layer = r.n_layer;
                if (is_language_model(m)) ctx.library.push_back(std::move(m));
            }
            initialized_ = true;
            rescan = true;
        }
        if (is_ready(scan_)) {
            try {
                auto result = scan_.get();
                for (auto& m : result.models) {
                    auto it = std::find_if(ctx.library.begin(), ctx.library.end(),
                                           [&](const auto& old) { return old.path == m.path; });
                    if (it != ctx.library.end())
                        *it = std::move(m);
                    else
                        ctx.library.push_back(std::move(m));
                }
                ctx.notify("Scan terminé : " + std::to_string(result.models.size()) + " modèles");
                for (const auto& error : result.errors) ctx.notify(error, true);
            } catch (const std::exception& e) {
                ctx.notify(e.what(), true);
            }
        }
        if (rescan && !scan_.valid()) {
            auto roots = default_model_roots();
            roots.insert(roots.end(), extra_roots_.begin(), extra_roots_.end());
            auto* store = &ctx.store;
            scan_ = std::async(std::launch::async, [roots = std::move(roots), store] {
                ScanResult result;
                for (const auto& path : scan_model_dirs(roots)) {
                    try {
                        auto m = read_gguf(path, false);
                        if (!is_language_model(m)) continue;
                        store->upsert_model(m);
                        result.models.push_back(std::move(m));
                    } catch (const std::exception& e) {
                        result.errors.push_back(path_utf8(path.filename()) + " : " + e.what());
                    }
                }
                return result;
            });
            rescan = false;
        }
        ImGui::InputTextWithHint("##Filtre", "Filtrer les modèles...", filter_.data(), filter_.size());
        ImGui::SameLine();
        if (ImGui::Button("Rescanner")) rescan = true;
        if (scan_.valid()) ImGui::TextDisabled("Scan des modèles en arrière-plan...");
        if (ImGui::Button("Ajouter un dossier...")) ImGui::OpenPopup("Dossier de modèles");
        if (ImGui::BeginPopup("Dossier de modèles")) {
            ImGui::InputText("Chemin UTF-8", folder_.data(), folder_.size());
            if (ImGui::Button("Ajouter") && folder_[0]) {
                auto p = std::filesystem::u8path(folder_.data());
                if (std::filesystem::is_directory(p)) {
                    extra_roots_.push_back(p);
                    rescan = true;
                    ImGui::CloseCurrentPopup();
                } else
                    ctx.notify("Ce dossier n'existe pas", true);
            }
            ImGui::EndPopup();
        }
        if (ImGui::BeginTable("Modèles", 6,
                              ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_ScrollY,
                              {0, 240})) {
            const char* columns[] = {"Nom", "Archi", "Quantif", "Taille", "Paramètres", "Contexte"};
            // Le nom prend la place libre ; les colonnes numériques gardent leur largeur naturelle.
            static const float weights[] = {3.0f, 1.4f, 1.0f, 1.0f, 1.0f, 1.0f};
            for (int i = 0; i < 6; ++i)
                ImGui::TableSetupColumn(columns[i], ImGuiTableColumnFlags_WidthStretch, weights[i], i);
            ImGui::TableHeadersRow();
            std::vector<size_t> order(ctx.library.size());
            std::iota(order.begin(), order.end(), 0);
            if (auto* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount) {
                const auto sort = specs->Specs[0];
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    const auto& x = ctx.library[a];
                    const auto& y = ctx.library[b];
                    int c = 0;
                    switch (sort.ColumnUserID) {
                        case 0:
                            c = x.name.compare(y.name);
                            break;
                        case 1:
                            c = x.architecture.compare(y.architecture);
                            break;
                        case 2:
                            c = x.file_type.compare(y.file_type);
                            break;
                        case 3:
                            c = (x.file_size > y.file_size) - (x.file_size < y.file_size);
                            break;
                        case 4:
                            c = (x.n_params > y.n_params) - (x.n_params < y.n_params);
                            break;
                        case 5:
                            c = (x.context_length > y.context_length) - (x.context_length < y.context_length);
                            break;
                    }
                    return sort.SortDirection == ImGuiSortDirection_Ascending ? c < 0 : c > 0;
                });
            }
            for (auto i : order) {
                const auto& m = ctx.library[i];
                if (filter_[0] &&
                    (m.name + " " + m.architecture + " " + m.file_type).find(filter_.data()) == std::string::npos)
                    continue;
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (ImGui::Selectable(m.name.c_str(), ctx.selected_model == static_cast<int>(i),
                                      ImGuiSelectableFlags_SpanAllColumns))
                    ctx.selected_model = static_cast<int>(i);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(m.architecture.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(m.file_type.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(format_bytes(m.file_size).c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(format_count(m.n_params).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%u", m.context_length);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (const auto* m = ctx.selected()) {
            section_header(m->name.c_str());
            ImGui::TextWrapped("%s", path_utf8(m->path).c_str());
            ImGui::Text("%s · %s · %u couches · contexte %u", m->architecture.c_str(), m->file_type.c_str(), m->n_layer,
                        m->context_length);
            uint64_t total = 0;
            for (const auto& [type, bytes] : m->bytes_by_type) total += bytes;
            for (const auto& [type, bytes] : m->bytes_by_type) {
                auto label = type + " : " + format_bytes(bytes);
                ImGui::ProgressBar(total ? static_cast<float>(static_cast<double>(bytes) / total) : 0, {-1, 0},
                                   label.c_str());
            }
            if (edit_path_ != path_utf8(m->path)) {
                edit_path_ = path_utf8(m->path);
                tags_.fill(0);
                notes_.fill(0);
                if (auto record = ctx.store.find_model(edit_path_)) {
                    std::string tags;
                    for (const auto& t : record->tags) {
                        if (!tags.empty()) tags += ", ";
                        tags += t;
                    }
                    std::snprintf(tags_.data(), tags_.size(), "%s", tags.c_str());
                    std::snprintf(notes_.data(), notes_.size(), "%s", record->notes.c_str());
                }
            }
            ImGui::InputText("Étiquettes (virgules)", tags_.data(), tags_.size());
            ImGui::InputTextMultiline("Notes", notes_.data(), notes_.size(), {-1, 80});
            if (ImGui::Button("Enregistrer")) {
                std::vector<std::string> tags;
                std::istringstream stream(tags_.data());
                std::string tag;
                while (std::getline(stream, tag, ',')) {
                    auto a = tag.find_first_not_of(" \t");
                    if (a != std::string::npos) tags.push_back(tag.substr(a, tag.find_last_not_of(" \t") - a + 1));
                }
                ctx.store.set_model_tags(edit_path_, tags);
                ctx.store.set_model_notes(edit_path_, notes_.data());
                ctx.notify("Fiche enregistrée");
            }
            if (ImGui::CollapsingHeader("Métadonnées")) {
                ImGui::InputText("Filtre des clés", metadata_filter_.data(), metadata_filter_.size());
                for (const auto& [key, value] : m->metadata)
                    if (!metadata_filter_[0] || key.find(metadata_filter_.data()) != std::string::npos)
                        ImGui::TextWrapped("%s : %s", key.c_str(), value.c_str());
            }
        }
    }
private:
    bool initialized_ = false;
    std::future<ScanResult> scan_;
    std::vector<std::filesystem::path> extra_roots_;
    std::array<char, 256> filter_{}, metadata_filter_{};
    std::array<char, 2048> folder_{}, tags_{};
    std::array<char, 16384> notes_{};
    std::string edit_path_;
};
} // namespace
std::unique_ptr<View> make_library_view() {
    return std::make_unique<LibraryView>();
}
void request_library_scan(View& view) {
    if (auto* library = dynamic_cast<LibraryView*>(&view)) library->rescan = true;
}
} // namespace banc::ui
