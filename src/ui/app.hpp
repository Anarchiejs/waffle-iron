#pragma once
#include "ui/view.hpp"
#include <imgui.h>
#include <array>
namespace waffle::ui {
class App {
public:
    explicit App(bool icons = false);
    ~App();
    void frame();
    bool quitting() const { return quit_; }
private:
    void sidebar(float width, float height, bool compact);
    void status_bar(float height);
    void toasts(const ImVec2& corner);
    Engine engine_;
    Telemetry telemetry_;
    Store store_;
    AppContext ctx_;
    std::array<std::unique_ptr<View>, 6> views_;
    int page_ = 0;
    bool quit_ = false, about_ = false, icons_ = false;
    double last_sample_ = -1;
};
} // namespace waffle::ui
