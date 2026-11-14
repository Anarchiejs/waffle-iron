#include "ui/widgets.hpp"

#include "core/util.hpp"
#include "core/vram_planner.hpp"
#include "ui/app_context.hpp"
#include "ui/theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace waffle::ui {

void AppContext::notify(std::string text, bool error) {
    toasts.push_back({std::move(text), error, ImGui::GetTime() + (error ? 8.0 : 4.0)});
    while (toasts.size() > 5) toasts.pop_front();
}

std::string format_bytes(uint64_t bytes) {
    return waffle::format_bytes(bytes);
}

std::string format_count(uint64_t n) {
    return waffle::format_count(n);
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
    field_label("GPU layers", "How many model layers are placed in VRAM. The maximum puts the whole model on the GPU.");
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
                "the whole context by default. Unchecked, it is limited to the window and uses much less VRAM at long "
                "context.");
    return changed;
}

bool busy_indicator(const char* label, float fraction) {
    const float button = ImGui::CalcTextSize("Cancel").x + ImGui::GetStyle().FramePadding.x * 2;
    if (fraction < 0.0f) {
        // Indeterminate bar: a segment that travels back and forth.
        const auto t = static_cast<float>(ImGui::GetTime());
        fraction = 0.5f + 0.5f * std::sin(t * 3.0f);
    }
    // Syrup wash with a bright leading edge; the label stays readable on top.
    const float w = std::max(em(4), ImGui::GetContentRegionAvail().x - button - ImGui::GetStyle().ItemSpacing.x);
    const float h = ImGui::GetFrameHeight();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    const float r = ImGui::GetStyle().FrameRounding;
    const float f = std::clamp(fraction, 0.0f, 1.0f);
    dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(col::field), r);
    if (f > 0) {
        dl->AddRectFilled(pos, {pos.x + w * f, pos.y + h}, ImGui::GetColorU32(alpha(col::accent, .26f)), r);
        dl->AddRectFilled({pos.x + w * f - em(.12f), pos.y}, {pos.x + w * f, pos.y + h},
                          ImGui::GetColorU32(col::accent));
    }
    dl->AddRect(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(col::border_hi), r);
    if (label && label[0]) {
        const ImVec4 clip{pos.x, pos.y, pos.x + w - em(.5f), pos.y + h};
        dl->AddText(nullptr, 0, {pos.x + em(.7f), pos.y + (h - ImGui::GetTextLineHeight()) * .5f},
                    ImGui::GetColorU32(col::text), label, nullptr, 0, &clip);
    }
    ImGui::Dummy({w, h});
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {em(1.3f), em(1.15f)});
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
    ImGui::Dummy({0, em(.3f)});
    push_font(fonts().semibold, .74f);
    ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
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
    ImGui::PushStyleColor(ImGuiCol_Header, col::field);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(.165f, .147f, .122f, 1));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(.188f, .167f, .137f, 1));
    const bool open = ImGui::CollapsingHeader(label, flags);
    ImGui::PopStyleColor(3);
    return open;
}

void help_marker(const char* text) {
    // A small outlined "?" disc, centered on the previous item (often a smaller label).
    const float line = ImGui::GetItemRectSize().y > 0 ? ImGui::GetItemRectSize().y : ImGui::GetTextLineHeight();
    ImGui::SameLine(0, em(.4f));
    const float d = em(.95f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 c{pos.x + d * .5f, pos.y + line * .5f};
    ImGui::Dummy({d, line});
    const bool hovered = ImGui::IsItemHovered();
    auto* dl = ImGui::GetWindowDrawList();
    dl->AddCircle(c, d * .5f, ImGui::GetColorU32(hovered ? col::accent : col::border_hi), 0, 1.2f);
    const float fs = em(.68f);
    const ImVec2 q = fonts().semibold->CalcTextSizeA(fs, FLT_MAX, 0, "?");
    dl->AddText(fonts().semibold, fs, {c.x - q.x * .5f, c.y - q.y * .5f},
                ImGui::GetColorU32(hovered ? col::accent : col::dim), "?");
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
    ImGui::PushStyleColor(ImGuiCol_Button, col::accent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col::accent_hi);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(.85f, .58f, .19f, 1));
    ImGui::PushStyleColor(ImGuiCol_Border, col::accent);
    ImGui::PushStyleColor(ImGuiCol_Text, col::accent_ink);
    push_font(fonts().semibold);
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopFont();
    ImGui::PopStyleColor(5);
    return clicked;
}

bool danger_button(const char* label, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, alpha(col::bad, .14f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, alpha(col::bad, .24f));
    ImGui::PushStyleColor(ImGuiCol_Border, alpha(col::bad, .45f));
    ImGui::PushStyleColor(ImGuiCol_Text, col::bad);
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(5);
    return clicked;
}

void badge(const char* text, const ImVec4& color, bool mono) {
    ImFont* font = mono ? fonts().mono : ImGui::GetFont();
    const float fs = mono ? em(.85f) : ImGui::GetFontSize();
    const ImVec2 size = font->CalcTextSizeA(fs, FLT_MAX, 0, text);
    const float line = ImGui::GetTextLineHeight();
    const ImVec2 pad{em(.45f), em(.1f)};
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    const float h = line + pad.y * 2;
    dl->AddRectFilled(pos, {pos.x + size.x + pad.x * 2, pos.y + h}, ImGui::GetColorU32(alpha(color, .14f)), em(.32f));
    dl->AddText(font, fs, {pos.x + pad.x, pos.y + (h - size.y) * .5f}, ImGui::GetColorU32(color), text);
    ImGui::Dummy({size.x + pad.x * 2, h});
}

void status_pill(const char* text, const ImVec4& color) {
    push_font(fonts().semibold, .86f);
    const ImVec2 size = ImGui::CalcTextSize(text);
    const float dot = em(.5f);
    const ImVec2 pad{em(.8f), em(.38f)};
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 end{pos.x + pad.x * 2 + dot + em(.5f) + size.x, pos.y + size.y + pad.y * 2};
    auto* dl = ImGui::GetWindowDrawList();
    const float r = (end.y - pos.y) * .5f;
    dl->AddRectFilled(pos, end, ImGui::GetColorU32(alpha(color, .10f)), r);
    dl->AddRect(pos, end, ImGui::GetColorU32(alpha(color, .38f)), r);
    dl->AddCircleFilled({pos.x + pad.x + dot * .5f, (pos.y + end.y) * .5f}, dot * .5f, ImGui::GetColorU32(color));
    dl->AddText({pos.x + pad.x + dot + em(.5f), pos.y + pad.y}, ImGui::GetColorU32(color), text);
    ImGui::Dummy({end.x - pos.x, end.y - pos.y});
    ImGui::PopFont();
}

void meter(float fraction, const ImVec4& color, float width, float height) {
    if (width < 0) width = ImGui::GetContentRegionAvail().x;
    if (height <= 0) height = em(.42f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    const float r = height * .5f;
    dl->AddRectFilled(pos, {pos.x + width, pos.y + height}, ImGui::GetColorU32(col::cell_off), r);
    const float f = std::clamp(fraction, 0.0f, 1.0f);
    if (f > 0)
        dl->AddRectFilled(pos, {pos.x + std::max(height, width * f), pos.y + height}, ImGui::GetColorU32(color), r);
    ImGui::Dummy({width, height});
}

void empty_state(const char* title, const char* text) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float width = std::min(avail.x, em(26));
    const float x = ImGui::GetCursorPosX() + (avail.x - width) * .5f;
    const float block = em(6.3f);
    if (avail.y > block) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - block) * .4f);
    {
        // A small 3 x 3 grid with the middle cell lit: the empty waffle.
        const float cell = em(.62f), gap = em(.2f), side = cell * 3 + gap * 2;
        ImGui::SetCursorPosX(x + (width - side) * .5f);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        auto* dl = ImGui::GetWindowDrawList();
        for (int i = 0; i < 9; ++i) {
            const ImVec2 a{p.x + static_cast<float>(i % 3) * (cell + gap),
                           p.y + static_cast<float>(std::div(i, 3).quot) * (cell + gap)};
            dl->AddRectFilled(a, {a.x + cell, a.y + cell}, ImGui::GetColorU32(i == 4 ? col::accent : col::border_hi),
                              em(.12f));
        }
        ImGui::Dummy({side, side + em(.5f)});
    }
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
    const float h = em(3.95f);
    auto* dl = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        if (i % static_cast<size_t>(cols)) ImGui::SameLine(0, gap);
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        // Field-colored tile: a small accent square before the label, the value in monospace.
        dl->AddRectFilled(pos, {pos.x + w, pos.y + h}, ImGui::GetColorU32(col::field), em(.65f));
        const float px = pos.x + em(.85f);
        const float clip = pos.x + w - em(.4f);
        const ImVec4 clip_rect{pos.x, pos.y, clip, pos.y + h};
        const float sq = em(.42f);
        dl->AddRectFilled({px, pos.y + em(.78f)}, {px + sq, pos.y + em(.78f) + sq}, ImGui::GetColorU32(t.accent),
                          em(.08f));
        dl->AddText(fonts().regular, em(.82f), {px + sq + em(.45f), pos.y + em(.55f)}, ImGui::GetColorU32(col::dim),
                    t.label.c_str(), nullptr, 0, &clip_rect);
        dl->AddText(fonts().mono, em(1.25f), {px, pos.y + em(1.55f)}, ImGui::GetColorU32(col::text), t.value.c_str(),
                    nullptr, 0, &clip_rect);
        if (!t.sub.empty()) {
            const float vw = fonts().mono->CalcTextSizeA(em(1.25f), FLT_MAX, 0, t.value.c_str()).x;
            dl->AddText(nullptr, em(.82f), {px + vw + em(.45f), pos.y + em(1.9f)}, ImGui::GetColorU32(col::dim),
                        t.sub.c_str(), nullptr, 0, &clip_rect);
        }
        ImGui::Dummy({w, h});
        if (ImGui::IsItemHovered() && !t.sub.empty())
            ImGui::SetTooltip("%s : %s %s", t.label.c_str(), t.value.c_str(), t.sub.c_str());
    }
}

void cell_meter(float fraction, const ImVec4& color, int cells, float width, float height) {
    if (width < 0) width = ImGui::GetContentRegionAvail().x;
    if (height <= 0) height = em(.62f);
    cells = std::max(cells, 1);
    const float gap = std::max(1.0f, em(.18f));
    const float cw = (width - gap * static_cast<float>(cells - 1)) / static_cast<float>(cells);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    const float lit = std::clamp(fraction, 0.0f, 1.0f) * static_cast<float>(cells);
    for (int i = 0; i < cells; ++i) {
        const float x = pos.x + static_cast<float>(i) * (cw + gap);
        dl->AddRectFilled({x, pos.y}, {x + cw, pos.y + height}, ImGui::GetColorU32(col::cell_off), em(.12f));
        // The last lit cell is filled in proportion, so small changes stay visible.
        const float part = std::clamp(lit - static_cast<float>(i), 0.0f, 1.0f);
        if (part > 0)
            dl->AddRectFilled({x, pos.y}, {x + cw * part, pos.y + height}, ImGui::GetColorU32(color), em(.12f));
    }
    ImGui::Dummy({width, height});
}

ImVec4 load_color(float fraction) {
    return fraction > .9f ? col::bad : fraction > .75f ? col::warn : col::accent;
}

void draw_mark(ImDrawList* dl, ImVec2 pos, float size) {
    dl->AddRectFilled(pos, {pos.x + size, pos.y + size}, ImGui::GetColorU32(col::accent), size * .28f);
    const float cell = size * 6.f / 36.f, step = size * 8.f / 36.f, start = size * 7.f / 36.f;
    const ImU32 ink = ImGui::GetColorU32(alpha(col::accent_ink, .82f));
    for (int i = 0; i < 9; ++i) {
        const ImVec2 a{pos.x + start + static_cast<float>(i % 3) * step,
                       pos.y + start + static_cast<float>(std::div(i, 3).quot) * step};
        dl->AddRectFilled(a, {a.x + cell, a.y + cell}, ink, cell * .25f);
    }
}

int waffle(const char* id, const std::vector<WaffleSlice>& slices, int cells, int columns, double total,
           float max_cell) {
    cells = std::max(cells, 1);
    columns = std::clamp(columns, 1, cells);
    double sum = 0;
    for (const auto& slice : slices) sum += std::max(0.0, slice.amount);
    if (total <= 0) total = sum;
    if (total <= 0) total = 1;
    // Cells per slice, by largest remainder, so the shown cells add up exactly.
    const double unit = total / cells;
    std::vector<int> counts(slices.size(), 0);
    std::vector<std::pair<double, size_t>> remainders;
    int used = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        const double exact = std::max(0.0, slices[i].amount) / unit;
        counts[i] = static_cast<int>(std::floor(exact));
        used += counts[i];
        remainders.emplace_back(exact - std::floor(exact), i);
    }
    const int target = static_cast<int>(std::lround(sum / unit));
    std::sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (const auto& [rest, i] : remainders) {
        if (used >= target) break;
        if (rest <= 0) continue;
        ++counts[i];
        ++used;
    }
    // A slice that is not zero keeps at least one cell, so it stays findable. The cell is taken from the
    // largest slice, so the total does not grow.
    for (size_t i = 0; i < slices.size(); ++i)
        if (slices[i].amount > 0 && counts[i] == 0) {
            counts[i] = 1;
            ++used;
            const auto largest = std::max_element(counts.begin(), counts.end());
            if (used > target && *largest > 1) {
                --*largest;
                --used;
            }
        }
    const int shown = std::max(cells, used);
    const int rows = (shown + columns - 1) / columns;
    const float avail = ImGui::GetContentRegionAvail().x;
    const float gap = std::max(1.0f, em(.2f));
    float cell = (avail - gap * static_cast<float>(columns - 1)) / static_cast<float>(columns);
    if (max_cell > 0) cell = std::min(cell, max_cell);
    cell = std::max(cell, em(.35f));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size{cell * static_cast<float>(columns) + gap * static_cast<float>(columns - 1),
                      cell * static_cast<float>(rows) + gap * static_cast<float>(rows - 1)};
    ImGui::PushID(id);
    ImGui::InvisibleButton("##waffle", {std::max(size.x, 1.0f), std::max(size.y, 1.0f)});
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 mouse = ImGui::GetMousePos();
    ImGui::PopID();
    auto* dl = ImGui::GetWindowDrawList();
    const float r = std::max(1.0f, cell * .16f);
    int hovered_slice = -1;
    int k = 0;
    auto cell_rect = [&](int n) {
        const ImVec2 a{pos.x + static_cast<float>(n % columns) * (cell + gap),
                       pos.y + static_cast<float>(std::div(n, columns).quot) * (cell + gap)};
        return std::pair{a, ImVec2{a.x + cell, a.y + cell}};
    };
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& slice = slices[i];
        for (int j = 0; j < counts[i]; ++j, ++k) {
            const auto [a, b] = cell_rect(k);
            const bool over = k >= cells; // beyond the capacity: a raspberry cell with the slice color inside
            if (over) {
                dl->AddRectFilled(a, b, ImGui::GetColorU32(col::bad), r);
                const float in = cell * .24f;
                dl->AddRectFilled({a.x + in, a.y + in}, {b.x - in, b.y - in}, ImGui::GetColorU32(slice.color), r * .5f);
            } else
                dl->AddRectFilled(a, b, ImGui::GetColorU32(slice.color), r);
            if (slice.hatched) {
                dl->PushClipRect(a, b, true);
                const ImU32 stripe = ImGui::GetColorU32(ImVec4(0, 0, 0, .32f));
                for (float t = -cell; t < cell; t += cell * .42f)
                    dl->AddLine({a.x + t, b.y}, {a.x + t + cell, a.y}, stripe, std::max(1.0f, cell * .12f));
                dl->PopClipRect();
            }
            if (hovered && mouse.x >= a.x && mouse.x < b.x + gap && mouse.y >= a.y && mouse.y < b.y + gap)
                hovered_slice = static_cast<int>(i);
        }
    }
    for (; k < shown; ++k) {
        const auto [a, b] = cell_rect(k);
        dl->AddRectFilled(a, b, ImGui::GetColorU32(col::cell_off), r);
    }
    if (hovered_slice >= 0) {
        const auto& slice = slices[static_cast<size_t>(hovered_slice)];
        ImGui::SetTooltip("%s  %s", slice.label.c_str(), slice.value.c_str());
    }
    return hovered_slice;
}

void waffle_legend(const std::vector<WaffleSlice>& slices) {
    const float avail = ImGui::GetContentRegionAvail().x;
    const float start = ImGui::GetCursorPosX();
    const float sq = em(.62f);
    bool first = true;
    for (const auto& slice : slices) {
        const float label_w = ImGui::CalcTextSize(slice.label.c_str()).x;
        const float value_w = fonts().mono->CalcTextSizeA(em(.88f), FLT_MAX, 0, slice.value.c_str()).x;
        const float w = sq + em(.5f) + label_w + em(.45f) + value_w;
        if (!first) {
            ImGui::SameLine(0, em(1.2f));
            if (ImGui::GetCursorPosX() - start + w > avail) ImGui::NewLine();
        }
        first = false;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float line = ImGui::GetTextLineHeight();
        auto* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled({p.x, p.y + (line - sq) * .5f}, {p.x + sq, p.y + (line + sq) * .5f},
                          ImGui::GetColorU32(slice.color), em(.12f));
        dl->AddText({p.x + sq + em(.5f), p.y}, ImGui::GetColorU32(col::soft), slice.label.c_str());
        dl->AddText(fonts().mono, em(.88f), {p.x + sq + em(.95f) + label_w, p.y + em(.06f)},
                    ImGui::GetColorU32(col::dim), slice.value.c_str());
        ImGui::Dummy({w, line});
    }
}

bool segmented(const char* id, const char* const* labels, int count, int& index) {
    ImGui::PushID(id);
    const float pad = em(.25f);
    const float h = ImGui::GetFrameHeight();
    float total = pad;
    std::vector<float> widths;
    for (int i = 0; i < count; ++i) {
        widths.push_back(std::max(em(4), ImGui::CalcTextSize(labels[i]).x + em(1.6f)));
        total += widths.back() + (i ? em(.15f) : 0);
    }
    total += pad;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    auto* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, {pos.x + total, pos.y + h + pad * 2}, ImGui::GetColorU32(col::card), em(.6f));
    dl->AddRect(pos, {pos.x + total, pos.y + h + pad * 2}, ImGui::GetColorU32(col::border), em(.6f));
    bool changed = false;
    float x = pos.x + pad;
    for (int i = 0; i < count; ++i) {
        ImGui::SetCursorScreenPos({x, pos.y + pad});
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##seg", {widths[static_cast<size_t>(i)], h}) && index != i) {
            index = i;
            changed = true;
        }
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        const bool on = index == i;
        const ImVec2 a{x, pos.y + pad}, b{x + widths[static_cast<size_t>(i)], pos.y + pad + h};
        if (on)
            dl->AddRectFilled(a, b, ImGui::GetColorU32(col::accent), em(.45f));
        else if (hovered)
            dl->AddRectFilled(a, b, ImGui::GetColorU32(alpha(col::accent, .08f)), em(.45f));
        ImFont* font = on ? fonts().semibold : fonts().regular;
        const ImVec2 ts = font->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, 0, labels[i]);
        dl->AddText(font, ImGui::GetFontSize(), {a.x + (b.x - a.x - ts.x) * .5f, a.y + (h - ts.y) * .5f},
                    ImGui::GetColorU32(on ? col::accent_ink : col::soft), labels[i]);
        x = b.x + em(.15f);
    }
    ImGui::SetCursorScreenPos(pos);
    ImGui::Dummy({total, h + pad * 2});
    ImGui::PopID();
    return changed;
}

int flow_text(const std::vector<TextSpan>& spans, ImFont* font) {
    if (!font) font = ImGui::GetFont();
    const float fs = ImGui::GetFontSize();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 cursor = origin;
    const float right = origin.x + std::max(1.f, ImGui::GetContentRegionAvail().x - em(.3f));
    const float line_height = ImGui::GetTextLineHeightWithSpacing() + em(.12f);
    auto* dl = ImGui::GetWindowDrawList();
    int hovered = -1;
    const bool window_hovered = ImGui::IsWindowHovered();
    for (size_t s = 0; s < spans.size(); ++s) {
        const auto& span = spans[s];
        const ImU32 color = ImGui::GetColorU32(span.color);
        const ImU32 back = ImGui::GetColorU32(span.background);
        const std::string& piece = span.text;
        for (size_t i = 0; i < piece.size();) {
            const auto c = static_cast<unsigned char>(piece[i]);
            size_t bytes = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
            bytes = std::min(bytes, piece.size() - i);
            const char* begin = piece.data() + i;
            const char* end = begin + bytes;
            if (c == '\n') {
                cursor.x = origin.x;
                cursor.y += line_height;
            } else if (c != '\r') {
                const float width = font->CalcTextSizeA(fs, FLT_MAX, 0, begin, end).x;
                // Wrap before a whole word when it would overflow; a word wider than the line breaks anywhere.
                const bool space = c == ' ' || c == '\t';
                const bool word_start =
                    !space && (i == 0 || piece[i - 1] == ' ' || piece[i - 1] == '\t' || piece[i - 1] == '\n');
                float advance = width;
                if (word_start) {
                    size_t j = i;
                    while (j < piece.size() && piece[j] != ' ' && piece[j] != '\t' && piece[j] != '\n' &&
                           piece[j] != '\r')
                        ++j;
                    advance = font->CalcTextSizeA(fs, FLT_MAX, 0, begin, piece.data() + j).x;
                    if (advance > right - origin.x) advance = width;
                }
                if (cursor.x + advance > right && cursor.x > origin.x && !space) {
                    cursor.x = origin.x;
                    cursor.y += line_height;
                }
                const ImVec2 b{cursor.x + width, cursor.y + ImGui::GetTextLineHeight() + em(.1f)};
                if (span.background.w > 0) dl->AddRectFilled({cursor.x, cursor.y - em(.05f)}, b, back);
                dl->AddText(font, fs, cursor, color, begin, end);
                if (span.strike) {
                    const float y = cursor.y + ImGui::GetTextLineHeight() * .55f;
                    dl->AddLine({cursor.x, y}, {cursor.x + width, y}, color, std::max(1.0f, em(.07f)));
                }
                if (window_hovered && ImGui::IsMouseHoveringRect(cursor, {b.x, cursor.y + line_height}))
                    hovered = static_cast<int>(s);
                cursor.x += width;
            }
            i += bytes;
        }
    }
    ImGui::Dummy({std::max(1.0f, right - origin.x), cursor.y - origin.y + line_height});
    return hovered;
}

float side_width(float ratio, float min_em, float max_em) {
    return std::clamp(ImGui::GetContentRegionAvail().x * ratio, em(min_em), em(max_em));
}

} // namespace waffle::ui
