#include "ui/icons.hpp"
#include "ui/view.hpp"
#include "ui/theme.hpp"
#include "ui/widgets.hpp"
#include "core/gguf_reader.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <future>
#include <numeric>
#include <sstream>

namespace waffle::ui {
namespace {
std::string path_utf8(const std::filesystem::path& path) {
    const auto bytes = path.u8string();
    return {bytes.begin(), bytes.end()};
}
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
struct ScanResult {
    std::vector<ModelInfo> models;
    std::vector<std::string> errors;
};
class LibraryView final : public View {
public:
    bool rescan = false;
    const char* title() const override { return "Library"; }
    const char* subtitle() const override { return "Your GGUF models: metadata, weight layout, tags, and notes."; }
    const char* icon() const override { return icon_library; }
    // Indexes the known folders from the first frame, even when this page is not shown.
    void tick(AppContext& ctx) override {
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
                // The database may still hold older records: a missing file, or an Ollama draft model.
                std::error_code ec;
                const auto blob = find_ollama_blob(m.path);
                if (is_language_model(m) && std::filesystem::exists(m.path, ec) && (!blob || blob->model))
                    ctx.library.push_back(std::move(m));
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
                const auto n = result.models.size();
                if (n == 0)
                    ctx.notify("Scan finished: no models");
                else if (n == 1)
                    ctx.notify("Scan finished: 1 model");
                else
                    ctx.notify("Scan finished: " + std::to_string(n) + " models");
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
    }
    void draw(AppContext& ctx) override {
        toolbar(ctx);
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float gap = em(1);
        if (avail.x >= em(48)) {
            // Master and detail side by side, each at full height.
            const float detail_w = std::clamp(avail.x * .34f, em(19), em(27));
            if (begin_card("##List", {avail.x - detail_w - gap, avail.y})) list(ctx);
            end_card();
            ImGui::SameLine(0, gap);
            if (begin_card("##Record", {detail_w, avail.y})) detail(ctx);
            end_card();
        } else {
            const float list_h = std::max(em(13), avail.y * .45f);
            if (begin_card("##List", {0, list_h})) list(ctx);
            end_card();
            if (begin_card("##Record", {0, std::max(em(16), ImGui::GetContentRegionAvail().y)})) detail(ctx);
            end_card();
        }
    }

private:
    void toolbar(AppContext& ctx) {
        const auto& style = ImGui::GetStyle();
        const float add_w = ImGui::CalcTextSize("Add a folder").x + style.FramePadding.x * 2;
        const float scan_w = ImGui::CalcTextSize("Scanning...").x + style.FramePadding.x * 2;
        const float avail = ImGui::GetContentRegionAvail().x;
        const float buttons = add_w + scan_w + style.ItemSpacing.x;
        ImGui::SetNextItemWidth(std::max(em(10), std::min(em(28), avail - buttons - style.ItemSpacing.x * 2)));
        ImGui::InputTextWithHint("##Filter", "Search: name, file, architecture, quantization...", filter_.data(),
                                 filter_.size());
        ImGui::SameLine();
        char count[48];
        std::snprintf(count, sizeof count, ctx.library.size() == 1 ? "%zu model" : "%zu models", ctx.library.size());
        const float right = avail - buttons;
        if (ImGui::GetCursorPosX() + ImGui::CalcTextSize(count).x + em(1) < right) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", count);
            ImGui::SameLine();
        }
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), right));
        if (ImGui::Button("Add a folder", {add_w, 0})) ImGui::OpenPopup("Model folder");
        ImGui::SameLine();
        ImGui::BeginDisabled(scan_.valid());
        if (primary_button(scan_.valid() ? "Scanning..." : "Rescan", {scan_w, 0})) rescan = true;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Ctrl+R");
        ImGui::SetNextWindowSize({em(30), 0});
        if (ImGui::BeginPopup("Model folder")) {
            card_title("Add a model folder", "Full path of a folder that contains .gguf files.");
            ImGui::SetNextItemWidth(-FLT_MIN);
#ifdef _WIN32
            ImGui::InputTextWithHint("##Path", "%USERPROFILE%\\models", folder_.data(), folder_.size());
#else
            ImGui::InputTextWithHint("##Path", "~/models", folder_.data(), folder_.size());
#endif
            if (primary_button("Add and scan") && folder_[0]) {
                auto p = std::filesystem::u8path(folder_.data());
                if (std::filesystem::is_directory(p)) {
                    extra_roots_.push_back(p);
                    rescan = true;
                    ImGui::CloseCurrentPopup();
                } else
                    ctx.notify("This folder does not exist", true);
            }
            ImGui::EndPopup();
        }
        ImGui::Dummy({0, em(.2f)});
    }
    void empty_library() {
        card_title("No model found", "On startup, Waffle Iron indexes the .gguf files in these folders:");
        const auto roots = candidate_model_roots();
#ifdef _WIN32
        if (roots.empty()) dim_wrapped("No known location: the USERPROFILE variable is missing.");
#else
        if (roots.empty()) dim_wrapped("No known location: the HOME variable is missing.");
#endif
        for (const auto& root : roots) {
            std::error_code ec;
            const bool present = std::filesystem::is_directory(root, ec);
            badge(present ? "present" : "absent", present ? col::good : col::faint);
            ImGui::SameLine();
            push_font(fonts().mono, .88f);
            ImGui::TextWrapped("%s", path_utf8(root).c_str());
            ImGui::PopFont();
        }
        ImGui::Dummy({0, em(.4f)});
        dim_wrapped("Put your .gguf files in the profile models folder, or add another folder. "
                    "Ollama blobs and LM Studio models are included when they are present.");
    }
    void list(AppContext& ctx) {
        if (ctx.library.empty()) {
            if (scan_.valid())
                empty_state("Looking for models...", "Known folders are scanned in the background.");
            else
                empty_library();
            return;
        }
        const float line = ImGui::GetTextLineHeight();
        const float row_h = line * 2 + em(.6f);
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {em(.6f), em(.3f)});
        if (ImGui::BeginTable("Models", 6,
                              ImGuiTableFlags_Sortable | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerH |
                                  ImGuiTableFlags_Hideable,
                              {0, ImGui::GetContentRegionAvail().y})) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoHide, 0, 0);
            ImGui::TableSetupColumn("Arch", ImGuiTableColumnFlags_WidthFixed, em(6.5f), 1);
            ImGui::TableSetupColumn("Quant", 0, 0, 2);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_PreferSortDescending, 0, 3);
            ImGui::TableSetupColumn("Parameters", ImGuiTableColumnFlags_PreferSortDescending, 0, 4);
            ImGui::TableSetupColumn("Context", ImGuiTableColumnFlags_PreferSortDescending, 0, 5);
            // Narrow table: hide secondary columns first so the name stays readable.
            const float table_w = ImGui::GetContentRegionAvail().x;
            ImGui::TableSetColumnEnabled(1, table_w >= em(36));
            ImGui::TableSetColumnEnabled(4, table_w >= em(30));
            ImGui::TableSetColumnEnabled(5, table_w >= em(25));
            ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
            ImGui::TableHeadersRow();
            ImGui::PopStyleColor();
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
                            c = model_title(x).compare(model_title(y));
                            if (c == 0) c = x.file_name.compare(y.file_name);
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
            const std::string needle = lower(filter_.data());
            int shown = 0;
            for (auto i : order) {
                const auto& m = ctx.library[i];
                if (!needle.empty() &&
                    lower(m.name + " " + m.file_name + " " + m.architecture + " " + m.file_type).find(needle) ==
                        std::string::npos)
                    continue;
                ++shown;
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow(0, row_h);
                ImGui::TableNextColumn();
                const ImVec2 pos = ImGui::GetCursorScreenPos();
                const bool selected = ctx.selected_model == static_cast<int>(i);
                if (ImGui::Selectable("##line", selected,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                                      {0, row_h - em(.6f)}))
                    ctx.selected_model = static_cast<int>(i);
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    ctx.requested_page = 1;
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                    ImGui::SetTooltip("%s\n%s", model_title(m).c_str(), m.file_name.c_str());
                // Name in semibold, file underneath; ellipsized to the column width.
                const float max_x = pos.x + ImGui::GetColumnWidth();
                auto* dl = ImGui::GetWindowDrawList();
                text_ellipsis(dl, fonts().semibold, em(1), pos, max_x, ImGui::GetColorU32(col::text), model_title(m));
                text_ellipsis(dl, fonts().regular, em(.84f), {pos.x, pos.y + line + 1}, max_x,
                              ImGui::GetColorU32(col::dim), m.file_name);
                const float middle = (row_h - em(.6f) - line) * .5f;
                auto cell = [&] {
                    ImGui::TableNextColumn();
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + middle);
                };
                cell();
                ImGui::TextUnformatted(m.architecture.c_str());
                cell();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.08f));
                badge(m.file_type.c_str(), col::accent);
                cell();
                ImGui::TextUnformatted(format_bytes(m.file_size).c_str());
                cell();
                ImGui::TextUnformatted(format_count(m.n_params).c_str());
                cell();
                ImGui::TextUnformatted(format_count(m.context_length).c_str());
                ImGui::PopID();
            }
            if (shown == 0) {
                ImGui::TableNextRow(0, row_h);
                ImGui::TableNextColumn();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + em(.6f));
                ImGui::TextDisabled("No model matches the search.");
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }
    void detail(AppContext& ctx) {
        const auto* m = ctx.selected();
        if (!m) {
            empty_state("No model selected",
                        "Choose a model in the list to show its record. A double-click opens it in the "
                        "VRAM planner.");
            return;
        }
        push_font(fonts().semibold, 1.25f);
        ImGui::TextWrapped("%s", model_title(*m).c_str());
        ImGui::PopFont();
        push_font(fonts().mono, .8f);
        ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
        ImGui::TextWrapped("%s", path_utf8(m->path).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::Dummy({0, em(.15f)});
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const float half = (ImGui::GetContentRegionAvail().x - gap) * .5f;
        if (primary_button("Plan VRAM", {half, 0})) ctx.requested_page = 1;
        ImGui::SameLine(0, gap);
        if (ImGui::Button("Open in sampler", {half, 0})) ctx.requested_page = 3;
        ImGui::Dummy({0, em(.25f)});
        char layers[32];
        std::snprintf(layers, sizeof layers, "%u", m->n_layer);
        tile_grid({{"Architecture", m->architecture, "", col::accent},
                   {"Quantization", m->file_type, "", col::mauve},
                   {"Size", format_bytes(m->file_size), "", col::teal},
                   {"Parameters", format_count(m->n_params), "", col::amber},
                   {"Layers", layers, "", col::accent},
                   {"Max context", format_count(m->context_length), "tokens", col::teal}},
                  em(6.8f));
        if (!m->bytes_by_type.empty()) {
            section_header("WEIGHT LAYOUT");
            uint64_t total = 0;
            for (const auto& [type, bytes] : m->bytes_by_type) total += bytes;
            static const ImVec4 colors[] = {col::accent, col::teal, col::mauve, col::amber};
            int k = 0;
            for (const auto& [type, bytes] : m->bytes_by_type) {
                const float f = total ? static_cast<float>(static_cast<double>(bytes) / total) : 0;
                ImGui::TextUnformatted(type.c_str());
                char value[64];
                std::snprintf(value, sizeof value, "%s  ·  %.0f %%", format_bytes(bytes).c_str(), 100.0 * f);
                ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize(value).x);
                ImGui::TextDisabled("%s", value);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.25f));
                meter(f, colors[k++ % 4]);
            }
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
        section_header("TAGS AND NOTES");
        field_label("Tags", "Comma-separated.");
        ImGui::InputTextWithHint("##Tags", "chat, fast, code...", tags_.data(), tags_.size());
        field_label("Notes");
        ImGui::InputTextMultiline("##Notes", notes_.data(), notes_.size(), {-FLT_MIN, em(5.5f)});
        if (primary_button("Save record")) {
            std::vector<std::string> tags;
            std::istringstream stream(tags_.data());
            std::string tag;
            while (std::getline(stream, tag, ',')) {
                auto a = tag.find_first_not_of(" \t");
                if (a != std::string::npos) tags.push_back(tag.substr(a, tag.find_last_not_of(" \t") - a + 1));
            }
            ctx.store.set_model_tags(edit_path_, tags);
            ctx.store.set_model_notes(edit_path_, notes_.data());
            ctx.notify("Record saved");
        }
        ImGui::Dummy({0, em(.2f)});
        if (!m->metadata.empty() && collapsing("GGUF metadata")) {
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputTextWithHint("##Keys", "Filter keys...", metadata_filter_.data(), metadata_filter_.size());
            if (ImGui::BeginTable("Metadata", 2,
                                  ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                      ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Key", 0, 1.0f);
                ImGui::TableSetupColumn("Value", 0, 1.3f);
                const std::string needle = lower(metadata_filter_.data());
                for (const auto& [key, value] : m->metadata) {
                    if (!needle.empty() && lower(key).find(needle) == std::string::npos) continue;
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    push_font(fonts().mono, .8f);
                    ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
                    ImGui::TextWrapped("%s", key.c_str());
                    ImGui::PopStyleColor();
                    ImGui::PopFont();
                    ImGui::TableNextColumn();
                    const std::string shown = value.size() > 400 ? value.substr(0, 400) + "..." : value;
                    ImGui::TextWrapped("%s", shown.c_str());
                }
                ImGui::EndTable();
            }
        }
    }
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
} // namespace waffle::ui
