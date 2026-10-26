#include "ui/widgets.hpp"

#include "core/vram_planner.hpp"
#include "ui/app_context.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace banc::ui {

void AppContext::notify(std::string text, bool error) {
    toasts.push_back({std::move(text), error, ImGui::GetTime() + (error ? 8.0 : 4.0)});
    while (toasts.size() > 5) toasts.pop_front();
}

std::string format_bytes(uint64_t bytes) {
    char buf[32];
    const double b = static_cast<double>(bytes);
    if (bytes >= GiB)
        std::snprintf(buf, sizeof buf, "%.2f Go", b / GiB);
    else if (bytes >= MiB)
        std::snprintf(buf, sizeof buf, "%.0f Mo", b / MiB);
    else if (bytes >= KiB)
        std::snprintf(buf, sizeof buf, "%.0f Ko", b / KiB);
    else
        std::snprintf(buf, sizeof buf, "%llu o", static_cast<unsigned long long>(bytes));
    return buf;
}

std::string format_count(uint64_t n) {
    char buf[32];
    const double v = static_cast<double>(n);
    if (n >= 1'000'000'000ull)
        std::snprintf(buf, sizeof buf, "%.2f G", v / 1e9);
    else if (n >= 1'000'000ull)
        std::snprintf(buf, sizeof buf, "%.0f M", v / 1e6);
    else if (n >= 1'000ull)
        std::snprintf(buf, sizeof buf, "%.1f k", v / 1e3);
    else
        std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(n));
    return buf;
}

std::string format_duration_ms(double ms) {
    char buf[32];
    if (ms >= 1000.0)
        std::snprintf(buf, sizeof buf, "%.1f s", ms / 1000.0);
    else
        std::snprintf(buf, sizeof buf, "%.0f ms", ms);
    return buf;
}

bool model_combo(const char* label, AppContext& ctx, int& index) {
    const char* preview = (index >= 0 && index < static_cast<int>(ctx.library.size()))
                              ? ctx.library[static_cast<size_t>(index)].name.c_str()
                              : "(choisir un modèle)";
    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < static_cast<int>(ctx.library.size()); ++i) {
            const auto& m = ctx.library[static_cast<size_t>(i)];
            ImGui::PushID(i);
            std::string item = m.name + "  ·  " + m.file_type + "  ·  " + format_bytes(m.file_size);
            if (ImGui::Selectable(item.c_str(), i == index)) {
                index = i;
                changed = true;
            }
            if (i == index) ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    return changed;
}

namespace {

bool string_combo(const char* label, std::string& value, const std::vector<std::string>& items) {
    bool changed = false;
    if (ImGui::BeginCombo(label, value.c_str())) {
        for (const auto& item : items) {
            if (ImGui::Selectable(item.c_str(), item == value)) {
                value = item;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

} // namespace

bool load_params_editor(LoadParams& p, int max_layers) {
    bool changed = false;
    int ngl = p.n_gpu_layers < 0 ? max_layers : std::min(p.n_gpu_layers, max_layers);
    if (ImGui::SliderInt("Couches sur GPU", &ngl, 0, std::max(max_layers, 1))) {
        p.n_gpu_layers = ngl >= max_layers ? -1 : ngl;
        changed = true;
    }
    help_marker("Nombre de couches du modèle placées en VRAM. Le maximum place tout le modèle sur la carte.");

    static const int ctx_steps[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    int ctx_idx = 0;
    for (int i = 0; i < IM_ARRAYSIZE(ctx_steps); ++i)
        if (static_cast<uint32_t>(ctx_steps[i]) <= p.n_ctx) ctx_idx = i;
    char ctx_label[32];
    std::snprintf(ctx_label, sizeof ctx_label, "%u tokens", p.n_ctx);
    if (ImGui::SliderInt("Contexte", &ctx_idx, 0, IM_ARRAYSIZE(ctx_steps) - 1, ctx_label)) {
        p.n_ctx = static_cast<uint32_t>(ctx_steps[ctx_idx]);
        changed = true;
    }

    changed |= string_combo("Cache K", p.type_k, kv_cache_types());
    changed |= string_combo("Cache V", p.type_v, kv_cache_types());
    help_marker(
        "Quantifier le cache KV (q8_0, q4_0) divise sa taille ; un cache V quantifié exige la flash attention.");

    const char* fa_items[] = {"Auto", "Non", "Oui"};
    int fa = p.flash_attn + 1;
    if (ImGui::Combo("Flash attention", &fa, fa_items, IM_ARRAYSIZE(fa_items))) {
        p.flash_attn = fa - 1;
        changed = true;
    }

    changed |= ImGui::Checkbox("Cache SWA complet", &p.swa_full);
    help_marker("Modèles à fenêtre glissante (Gemma...) : par défaut llama.cpp alloue le cache de ces couches sur "
                "tout le contexte. Décoché, il se limite à la fenêtre : beaucoup moins de VRAM en contexte long.");

    static const int ub_steps[] = {64, 128, 256, 512, 1024, 2048};
    int ub_idx = 3;
    for (int i = 0; i < IM_ARRAYSIZE(ub_steps); ++i)
        if (static_cast<uint32_t>(ub_steps[i]) == p.n_ubatch) ub_idx = i;
    char ub_label[16];
    std::snprintf(ub_label, sizeof ub_label, "%u", p.n_ubatch);
    if (ImGui::SliderInt("Micro-lot (ubatch)", &ub_idx, 0, IM_ARRAYSIZE(ub_steps) - 1, ub_label)) {
        p.n_ubatch = static_cast<uint32_t>(ub_steps[ub_idx]);
        changed = true;
    }
    return changed;
}

bool busy_indicator(const char* label, float fraction) {
    if (fraction >= 0.0f) {
        ImGui::ProgressBar(fraction, ImVec2(-90.0f, 0.0f), label);
    } else {
        // Barre indéterminée : un segment qui va et vient.
        const float t = static_cast<float>(ImGui::GetTime());
        const float phase = 0.5f + 0.5f * std::sin(t * 3.0f);
        ImGui::ProgressBar(phase, ImVec2(-90.0f, 0.0f), label);
    }
    ImGui::SameLine();
    return ImGui::Button("Annuler", ImVec2(80.0f, 0.0f));
}

void section_header(const char* text) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::Separator();
}

void help_marker(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace banc::ui
