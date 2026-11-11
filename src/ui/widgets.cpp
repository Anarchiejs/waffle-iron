#include "ui/widgets.hpp"

#include "core/vram_planner.hpp"
#include "ui/app_context.hpp"
#include "ui/theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace waffle::ui {

void AppContext::notify(std::string text, bool error) {
    toasts.push_back({std::move(text), error, ImGui::GetTime() + (error ? 8.0 : 4.0)});
    while (toasts.size() > 5) toasts.pop_front();
}

std::string format_bytes(uint64_t bytes) {
    char buf[32];
    const double b = static_cast<double>(bytes);
    if (bytes >= GiB)
        std::snprintf(buf, sizeof buf, "%.2f GiB", b / GiB);
    else if (bytes >= MiB)
        std::snprintf(buf, sizeof buf, "%.0f MiB", b / MiB);
    else if (bytes >= KiB)
        std::snprintf(buf, sizeof buf, "%.0f KiB", b / KiB);
    else
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
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

float em(float k) {
    return ImGui::GetFontSize() * k;
}

void push_font(ImFont* font, float scale) {
    ImGui::PushFont(font, ImGui::GetStyle().FontSizeBase * scale);
}

std::string model_title(const ModelInfo& m) {
    return m.name.empty() ? m.file_name : m.name;
}

std::string model_details(const ModelInfo& m) {
    return m.file_name + "  ·  " + m.file_type + "  ·  " + format_bytes(m.file_size);
}

void text_ellipsis(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, float max_x, ImU32 color,
                   const std::string& text) {
    if (!font) font = ImGui::GetFont();
    auto width = [&](const char* b, const char* e) { return font->CalcTextSizeA(size, FLT_MAX, 0, b, e).x; };
    const char* begin = text.c_str();
    const char* end = begin + text.size();
    if (pos.x + width(begin, end) <= max_x) {
        dl->AddText(font, size, pos, color, begin, end);
        return;
    }
    static const char ellipsis[] = "…"; // ellipsis (U+2026)
    const float room = max_x - pos.x - width(ellipsis, ellipsis + 3);
    while (end > begin && width(begin, end) > room) {
        --end;
        while (end > begin && (static_cast<unsigned char>(*end) & 0xC0) == 0x80) --end; // UTF-8 boundary
    }
    const std::string cut = std::string(begin, end) + ellipsis;
    dl->AddText(font, size, pos, color, cut.c_str());
}

bool model_combo(const char* id, AppContext& ctx, int& index, float width) {
    const bool valid = index >= 0 && index < static_cast<int>(ctx.library.size());
    std::string preview = "Choose a model...";
    if (valid) {
        const auto& m = ctx.library[static_cast<size_t>(index)];
        preview = model_title(m) + "  ·  " + m.file_type;
    }
    ImGui::SetNextItemWidth(width);
    bool changed = false;
    ImGui::SetNextWindowSizeConstraints({em(24), 0}, {FLT_MAX, em(24)});
    if (ImGui::BeginCombo(id, preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (ctx.library.empty()) ImGui::TextDisabled("The library is empty.");
        for (int i = 0; i < static_cast<int>(ctx.library.size()); ++i) {
            const auto& m = ctx.library[static_cast<size_t>(i)];
            ImGui::PushID(i);
            const float h = ImGui::GetTextLineHeight() * 2 + ImGui::GetStyle().ItemSpacing.y * .3f;
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            if (ImGui::Selectable("##model", i == index, 0, {0, h})) {
                index = i;
                changed = true;
            }
            if (i == index) ImGui::SetItemDefaultFocus();
            auto* dl = ImGui::GetWindowDrawList();
            const auto title = model_title(m);
            const auto details = model_details(m);
            const float max_x = pos.x + ImGui::GetContentRegionAvail().x;
            text_ellipsis(dl, nullptr, ImGui::GetFontSize(), pos, max_x, ImGui::GetColorU32(ImGuiCol_Text), title);
            text_ellipsis(dl, nullptr, ImGui::GetFontSize(), {pos.x, pos.y + ImGui::GetTextLineHeight() + 1}, max_x,
                          ImGui::GetColorU32(col::dim), details);
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    return changed;
}

namespace {

bool string_combo(const char* id, std::string& value, const std::vector<std::string>& items) {
    bool changed = false;
    if (ImGui::BeginCombo(id, value.c_str())) {
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
    const int top = std::max(max_layers, 1);
    int ngl = p.n_gpu_layers < 0 ? top : std::min(p.n_gpu_layers, top);
    field_label("GPU layers",
                "How many model layers are placed in VRAM. The maximum puts the whole model on the GPU.");
    char ngl_label[48];
    if (ngl >= top)
        std::snprintf(ngl_label, sizeof ngl_label, "All (%d)", top);
    else
        std::snprintf(ngl_label, sizeof ngl_label, "%d / %d", ngl, top);
    if (ImGui::SliderInt("##ngl", &ngl, 0, top, ngl_label)) {
        p.n_gpu_layers = ngl >= top ? -1 : ngl;
        changed = true;
    }

    static const int ctx_steps[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    int ctx_idx = 0;
    for (int i = 0; i < IM_ARRAYSIZE(ctx_steps); ++i)
        if (static_cast<uint32_t>(ctx_steps[i]) <= p.n_ctx) ctx_idx = i;
    char ctx_label[32];
    std::snprintf(ctx_label, sizeof ctx_label, "%u tokens", p.n_ctx);
    field_label("Context");
    if (ImGui::SliderInt("##ctx", &ctx_idx, 0, IM_ARRAYSIZE(ctx_steps) - 1, ctx_label)) {
        p.n_ctx = static_cast<uint32_t>(ctx_steps[ctx_idx]);
        changed = true;
    }

    field_pair(
        [&](float w) {
            field_label("Cache K", nullptr, w);
            changed |= string_combo("##k", p.type_k, kv_cache_types());
        },
        [&](float w) {
            field_label("Cache V",
                        "Quantizing the KV cache (q8_0, q4_0) shrinks it; a quantized V cache requires flash "
                        "attention.",
                        w);
            changed |= string_combo("##v", p.type_v, kv_cache_types());
        });
    field_pair(
        [&](float w) {
            field_label("Flash attention", nullptr, w);
            const char* fa_items[] = {"Auto", "No", "Yes"};
            int fa = p.flash_attn + 1;
            if (ImGui::Combo("##fa", &fa, fa_items, IM_ARRAYSIZE(fa_items))) {
                p.flash_attn = fa - 1;
                changed = true;
            }
        },
        [&](float w) {
            static const int ub_steps[] = {64, 128, 256, 512, 1024, 2048};
            int ub_idx = 3;
            for (int i = 0; i < IM_ARRAYSIZE(ub_steps); ++i)
                if (static_cast<uint32_t>(ub_steps[i]) == p.n_ubatch) ub_idx = i;
            char ub_label[16];
            std::snprintf(ub_label, sizeof ub_label, "%u", p.n_ubatch);
            field_label("Micro-batch (ubatch)", nullptr, w);
            if (ImGui::SliderInt("##ub", &ub_idx, 0, IM_ARRAYSIZE(ub_steps) - 1, ub_label)) {
                p.n_ubatch = static_cast<uint32_t>(ub_steps[ub_idx]);
                changed = true;
            }
        });
    changed |= checkbox("Full SWA cache", &p.swa_full);
    help_marker("Sliding-window models (Gemma...): llama.cpp allocates those layers' cache for "
                "the whole context by default. Unchecked, it is limited to the window and uses much less VRAM at long context.");
    return changed;
}

bool busy_indicator(const char* label, float fraction) {
    const float button = ImGui::CalcTextSize("Cancel").x + ImGui::GetStyle().FramePadding.x * 2;
    if (fraction < 0.0f) {
        // Indeterminate bar: a segment that travels back and forth.
        const float t = static_cast<float>(ImGui::GetTime());
        fraction = 0.5f + 0.5f * std::sin(t * 3.0f);
    }
    ImGui::ProgressBar(std::clamp(fraction, 0.0f, 1.0f),
                       ImVec2(-(button + ImGui::GetStyle().ItemSpacing.x), 0.0f), label);
    ImGui::SameLine();
    return ImGui::Button("Cancel", ImVec2(button, 0.0f));
}

bool begin_card(const char* id, ImVec2 size, ImGuiWindowFlags flags) {
    ImGuiChildFlags child = ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding;
    if (size.y == 0) {
        child |= ImGuiChildFlags_AutoResizeY;
        flags |= ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, col::card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.1f), em(.95f)});
    const bool open = ImGui::BeginChild(id, size, child, flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    return open;
}

void end_card() {
    ImGui::EndChild();
}

void card_title(const char* title, const char* subtitle) {
    push_font(fonts().semibold, 1.06f);
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    if (subtitle) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - em(.3f));
        dim_wrapped(subtitle);
    }
    ImGui::Dummy({0, em(.15f)});
}

void section_header(const char* text) {
    ImGui::Dummy({0, em(.2f)});
    push_font(fonts().semibold, .82f);
    ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void field_label(const char* text, const char* help, float width) {
    push_font(nullptr, .9f);
    ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
    if (help) help_marker(help);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y * .55f);
    ImGui::SetNextItemWidth(width);
}

void field_pair(const std::function<void(float)>& left, const std::function<void(float)>& right) {
    const float avail = ImGui::GetContentRegionAvail().x;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    if (avail < em(15)) {
        left(avail);
        right(avail);
        return;
    }
    // Explicit positions instead of SameLine: SameLine would resume the left frame's baseline and
    // push the right label downward.
    const float w = std::floor((avail - gap) * .5f);
    const ImVec2 start = ImGui::GetCursorPos();
    ImGui::BeginGroup();
    left(w);
    ImGui::EndGroup();
    const float bottom_left = ImGui::GetCursorPosY();
    ImGui::SetCursorPos({start.x + w + gap, start.y});
    ImGui::BeginGroup();
    right(w);
    ImGui::EndGroup();
    const float bottom = std::max(bottom_left, ImGui::GetCursorPosY());
    ImGui::SetCursorPos({start.x, bottom});
}

bool checkbox(const char* label, bool* value) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {em(.22f), em(.22f)});
    const bool changed = ImGui::Checkbox(label, value);
    ImGui::PopStyleVar();
    return changed;
}

bool collapsing(const char* label, ImGuiTreeNodeFlags flags) {
    ImGui::PushStyleColor(ImGuiCol_Header, col::card_hi);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(.165f, .180f, .225f, 1));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(.20f, .22f, .28f, 1));
    const bool open = ImGui::CollapsingHeader(label, flags);
    ImGui::PopStyleColor(3);
    return open;
}

void help_marker(const char* text) {
    ImGui::SameLine(0, em(.35f));
    ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
    push_font(nullptr, .9f);
    ImGui::TextUnformatted("(?)");
    ImGui::PopFont();
    ImGui::PopStyleColor();
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(em(28));
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void dim_wrapped(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

bool primary_button(const char* label, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_Button, col::accent_bg);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.34f, 0.37f, 0.68f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.39f, 0.43f, 0.76f, 1));
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    return clicked;
}

void badge(const char* text, const ImVec4& color) {
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 pad{em(.45f), em(.08f)};
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, {pos.x + size.x + pad.x * 2, pos.y + size.y + pad.y * 2},
                      ImGui::GetColorU32(ImVec4(color.x, color.y, color.z, .16f)), em(.35f));
    dl->AddText({pos.x + pad.x, pos.y + pad.y}, ImGui::GetColorU32(color), text);
    ImGui::Dummy({size.x + pad.x * 2, size.y + pad.y * 2});
}

void meter(float fraction, const ImVec4& color, float width, float height) {
    if (width < 0) width = ImGui::GetContentRegionAvail().x;
    if (height <= 0) height = em(.42f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    const float r = height * .5f;
    dl->AddRectFilled(pos, {pos.x + width, pos.y + height}, ImGui::GetColorU32(ImVec4(1, 1, 1, .07f)), r);
    const float f = std::clamp(fraction, 0.0f, 1.0f);
    if (f > 0) dl->AddRectFilled(pos, {pos.x + std::max(height, width * f), pos.y + height}, ImGui::GetColorU32(color), r);
    ImGui::Dummy({width, height});
}

void empty_state(const char* title, const char* text) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float width = std::min(avail.x, em(26));
    const float x = ImGui::GetCursorPosX() + (avail.x - width) * .5f;
    const float block = em(4.5f);
    if (avail.y > block) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - block) * .4f);
    push_font(fonts().semibold, 1.15f);
    const float tw = ImGui::CalcTextSize(title).x;
    ImGui::SetCursorPosX(x + std::max(0.0f, (width - tw) * .5f));
    ImGui::TextUnformatted(title);
    ImGui::PopFont();
    ImGui::SetCursorPosX(x);
    ImGui::PushTextWrapPos(x + width);
    ImGui::PushStyleColor(ImGuiCol_Text, col::dim);
    // Center a short text line by line: one line when it fits, otherwise a wrapped paragraph.
    const float lw = ImGui::CalcTextSize(text).x;
    if (lw < width) ImGui::SetCursorPosX(x + (width - lw) * .5f);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
}

void tile_grid(const std::vector<Tile>& tiles, float min_width) {
    if (tiles.empty()) return;
    const float avail = ImGui::GetContentRegionAvail().x;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    int cols = std::max(1, static_cast<int>((avail + gap) / (min_width + gap)));
    cols = std::min(cols, static_cast<int>(tiles.size()));
    // Balanced rows: 6 tiles in 4 columns become 3 + 3 rather than 4 + 2.
    const int rows = (static_cast<int>(tiles.size()) + cols - 1) / cols;
    cols = (static_cast<int>(tiles.size()) + rows - 1) / rows;
    const float w = std::floor((avail - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols));
    const float h = em(3.9f);
    auto* dl = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        if (i % static_cast<size_t>(cols)) ImGui::SameLine(0, gap);
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(col::card_hi), em(.55f));
        dl->AddRectFilled({pos.x, pos.y + em(.7f)}, {pos.x + em(.18f), pos.y + h - em(.7f)},
                          ImGui::GetColorU32(t.accent), em(.1f));
        const float px = pos.x + em(.85f);
        const float clip = pos.x + w - em(.4f);
        const ImVec4 clip_rect{pos.x, pos.y, clip, pos.y + h};
        dl->AddText(fonts().semibold, em(.82f), {px, pos.y + em(.55f)}, ImGui::GetColorU32(col::dim), t.label.c_str(),
                    nullptr, 0, &clip_rect);
        dl->AddText(fonts().semibold, em(1.32f), {px, pos.y + em(1.45f)}, ImGui::GetColorU32(col::text),
                    t.value.c_str(), nullptr, 0, &clip_rect);
        if (!t.sub.empty()) {
            const float vw = fonts().semibold->CalcTextSizeA(em(1.32f), FLT_MAX, 0, t.value.c_str()).x;
            dl->AddText(nullptr, em(.85f), {px + vw + em(.45f), pos.y + em(1.85f)}, ImGui::GetColorU32(col::dim),
                        t.sub.c_str(), nullptr, 0, &clip_rect);
        }
        ImGui::Dummy({w, h});
        if (ImGui::IsItemHovered() && !t.sub.empty())
            ImGui::SetTooltip("%s : %s %s", t.label.c_str(), t.value.c_str(), t.sub.c_str());
    }
}

float side_width(float ratio, float min_em, float max_em) {
    return std::clamp(ImGui::GetContentRegionAvail().x * ratio, em(min_em), em(max_em));
}

} // namespace waffle::ui
