#include "ui/app.hpp"
#include "ui/widgets.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
namespace banc::ui {
void request_library_scan(View& view);
App::App(bool icons) : store_(Store::default_path()), ctx_{engine_, telemetry_, store_}, icons_(icons) {
    views_ = {make_library_view(), make_vram_view(),    make_bench_view(),
              make_sampler_view(), make_compare_view(), make_monitor_view()};
}
App::~App() {
    // Attendre le moteur avant de détruire les vues et leurs références au contexte.
    engine_.cancel_all();
    engine_.submit([](EngineSession&) {}).wait();
}
void App::frame() {
    const double now = ImGui::GetTime();
    if (now - last_sample_ >= .1) {
        last_sample_ = now;
        if (auto sample = telemetry_.sample()) ctx_.gpu_history.push_back(*sample);
        while (
            ctx_.gpu_history.size() > 6000 ||
            (!ctx_.gpu_history.empty() && ctx_.gpu_history.back().t_seconds - ctx_.gpu_history.front().t_seconds > 600))
            ctx_.gpu_history.pop_front();
        while (ctx_.tps_history.size() > 6000 ||
               (!ctx_.tps_history.empty() && now - ctx_.tps_history.front().first > 600))
            ctx_.tps_history.pop_front();
    }
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Fichier")) {
            if (ImGui::MenuItem("Rescanner les modèles")) {
                if (views_[0]) request_library_scan(*views_[0]);
                if (views_[0]) views_[0]->open = true;
            }
            if (ImGui::MenuItem("Quitter")) quit_ = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Affichage")) {
            for (auto& v : views_)
                if (v) ImGui::MenuItem(v->title(), nullptr, &v->open);
            if (ImGui::MenuItem("Réinitialiser la disposition")) reset_layout_ = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Aide")) {
            ImGui::MenuItem("À propos", nullptr, &about_);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    auto* vp = ImGui::GetMainViewport();
    const float status_height = ImGui::GetFrameHeight() + 8;
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize({vp->WorkSize.x, vp->WorkSize.y - status_height});
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##Espace", nullptr,
                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                     ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleVar(3);
    const auto dock = ImGui::GetID("BancDock");
    if (reset_layout_ || !ImGui::DockBuilderGetNode(dock)) {
        ImGui::DockBuilderRemoveNode(dock);
        ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dock, {vp->WorkSize.x, vp->WorkSize.y - status_height});
        ImGuiID center = dock;
        const auto left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, .27f, nullptr, &center);
        const auto bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, .28f, nullptr, &center);
        for (size_t i = 0; i < views_.size(); ++i)
            if (views_[i]) ImGui::DockBuilderDockWindow(views_[i]->title(), i == 0 ? left : i == 5 ? bottom : center);
        ImGui::DockBuilderFinish(dock);
        reset_layout_ = false;
    }
    ImGui::DockSpace(dock);
    ImGui::End();
    for (auto& v : views_)
        if (v && v->open) {
            if (ImGui::Begin(v->title(), &v->open)) {
                if (icons_ && v->icon()[0]) {
                    ImGui::TextUnformatted(v->icon());
                    ImGui::SameLine();
                }
                try {
                    v->draw(ctx_);
                } catch (const std::exception& e) {
                    ctx_.notify(e.what(), true);
                } catch (...) {
                    ctx_.notify("Erreur inattendue dans la vue", true);
                }
            }
            ImGui::End();
        }
    ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - status_height});
    ImGui::SetNextWindowSize({vp->WorkSize.x, status_height});
    ImGui::Begin("##État", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings);
    auto status = engine_.status();
    ImGui::TextUnformatted(status.empty() ? (engine_.state() == EngineState::Idle ? "Prêt" : "Travail en cours...")
                                          : status.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled(" | %s", telemetry_.device_name().c_str());
    if (!ctx_.gpu_history.empty()) {
        const auto& g = ctx_.gpu_history.back();
        ImGui::SameLine();
        ImGui::Text(" | VRAM %s / %s", format_bytes(g.vram_used).c_str(), format_bytes(g.vram_total).c_str());
    }
    ImGui::End();
    if (about_) {
        ImGui::Begin("À propos", &about_, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Banc Local %s\nllama.cpp %s", BANC_VERSION, BANC_LLAMA_VERSION);
        ImGui::End();
    }
    while (!ctx_.toasts.empty() && ctx_.toasts.front().expires_at <= now) ctx_.toasts.pop_front();
    float y = vp->WorkPos.y + vp->WorkSize.y - status_height - 12;
    int id = 0;
    for (auto it = ctx_.toasts.rbegin(); it != ctx_.toasts.rend(); ++it) {
        if (it->expires_at <= now) continue;
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 12, y}, ImGuiCond_Always, {1, 1});
        const float alpha = static_cast<float>(std::clamp(it->expires_at - now, 0.0, 1.0));
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
        std::string name = "##Toast" + std::to_string(id++);
        ImGui::Begin(name.c_str(), nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);
        ImGui::PushTextWrapPos(360);
        ImGui::TextColored(it->error ? ImVec4(1, .5f, .5f, 1) : ImVec4(.7f, .8f, 1, 1), "%s", it->text.c_str());
        ImGui::PopTextWrapPos();
        y -= ImGui::GetWindowHeight() + 8;
        ImGui::End();
        ImGui::PopStyleVar();
    }
}
} // namespace banc::ui
