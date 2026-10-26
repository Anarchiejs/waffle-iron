#pragma once
#include "ui/view.hpp"
#include <array>
namespace banc::ui {
class App {
public:
    explicit App(bool icons = false);
    ~App();
    void frame();
    bool quitting() const { return quit_; }
private:
    Engine engine_;
    Telemetry telemetry_;
    Store store_;
    AppContext ctx_;
    std::array<std::unique_ptr<View>, 6> views_;
    bool quit_ = false, reset_layout_ = false, about_ = false, icons_ = false;
    double last_sample_ = -1;
};
} // namespace banc::ui
