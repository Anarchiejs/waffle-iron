#include "core/bench.hpp"
#include "core/engine.hpp"
#include "core/gguf_reader.hpp"
#include "core/telemetry.hpp"
#include "core/types.hpp"
#include "core/vram_planner.hpp"
#include "store/store.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

std::string utf8_from_path(const std::filesystem::path& path) {
    const auto u8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

std::filesystem::path path_from_utf8(std::string_view utf8) {
    const std::u8string u8(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size());
    return std::filesystem::path(u8);
}

std::filesystem::path normalize_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path : canonical;
}

std::string now_iso() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    if (std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) return "1970-01-01T00:00:00Z";
    return buf;
}

size_t display_width(const std::string& text) {
    size_t width = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) ++width;
    }
    return width;
}

void print_table(const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
    std::vector<size_t> widths(headers.size());
    for (size_t column = 0; column < headers.size(); ++column) widths[column] = display_width(headers[column]);
    for (const auto& row : rows) {
        for (size_t column = 0; column < row.size() && column < widths.size(); ++column)
            widths[column] = std::max(widths[column], display_width(row[column]));
    }
    auto emit = [&](const std::vector<std::string>& row) {
        for (size_t column = 0; column < headers.size(); ++column) {
            if (column) std::cout << "  ";
            const std::string cell = column < row.size() ? row[column] : "";
            std::cout << cell;
            if (column + 1 != headers.size()) std::cout << std::string(widths[column] - display_width(cell), ' ');
        }
        std::cout << '\n';
    };
    emit(headers);
    std::vector<std::string> rule;
    rule.reserve(headers.size());
    for (size_t width : widths) rule.emplace_back(width, '-');
    emit(rule);
    for (const auto& row : rows) emit(row);
}

std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << ' ' << units[unit];
    return out.str();
}

std::string format_count(uint64_t count) {
    const char* units[] = {"", " k", " M", " G", " T"};
    double value = static_cast<double>(count);
    int unit = 0;
    while (value >= 1000.0 && unit < 4) {
        value /= 1000.0;
        ++unit;
    }
    std::ostringstream out;
    if (unit == 0)
        out << count;
    else
        out << std::fixed << std::setprecision(value >= 100 ? 0 : 1) << value << units[unit];
    return out.str();
}

std::vector<std::string> split_comma(const std::string& text, const std::string& flag) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : text) {
        if (c == ',') {
            if (!current.empty()) parts.push_back(current);
            current.clear();
        } else if (c != ' ') {
            current.push_back(c);
        }
    }
    if (!current.empty()) parts.push_back(current);
    if (parts.empty()) throw waffle::Error("Empty list for " + flag);
    return parts;
}

long long parse_integer(const std::string& text, const std::string& flag) {
    if (text.empty()) throw waffle::Error("Expected an integer for " + flag);
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        throw waffle::Error("Expected an integer for " + flag + ": " + text);
    }
    if (used != text.size()) throw waffle::Error("Expected an integer for " + flag + ": " + text);
    return value;
}

double parse_double(const std::string& text, const std::string& flag) {
    if (text.empty()) throw waffle::Error("Expected a number for " + flag);
    size_t used = 0;
    double value = 0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw waffle::Error("Expected a number for " + flag + ": " + text);
    }
    if (used != text.size()) throw waffle::Error("Expected a number for " + flag + ": " + text);
    return value;
}

uint32_t parse_u32_at_least(const std::string& text, const std::string& flag, uint32_t minimum) {
    const long long value = parse_integer(text, flag);
    if (value < static_cast<long long>(minimum) || value > static_cast<long long>(UINT32_MAX))
        throw waffle::Error(flag + " out of range: " + text);
    return static_cast<uint32_t>(value);
}

std::vector<std::string> command_line() {
#ifdef _WIN32
    int argc = 0;
    LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wide) return {};
    std::vector<std::string> args;
    args.reserve(argc > 0 ? static_cast<size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
        if (bytes <= 1) {
            args.emplace_back();
            continue;
        }
        std::string utf8(static_cast<size_t>(bytes - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, utf8.data(), bytes, nullptr, nullptr);
        args.push_back(std::move(utf8));
    }
    LocalFree(wide);
    return args;
#else
    // Filled in by main on other platforms.
    return {};
#endif
}

const char* kGeneralHelp = R"HELP(waffle — local GGUF model workbench

Usage: waffle [--db PATH] COMMAND [options]

Commands:
  scan [folders...]   Record GGUF files found (default model folders otherwise)
  list                Library: name, quantization, size, parameters, context, path
  info MODEL          Main metadata and bytes by type
  plan MODEL          VRAM estimate and how many GPU layers fit
  bench MODEL         Speed measurement, settings grid, save
  history MODEL       Measurement history for this model
  campaigns           Saved campaigns
  export ID           Export a campaign (--md by default, or --csv)
  gpu                 One GPU telemetry sample

  --db PATH           SQLite database (default: %LOCALAPPDATA%\WaffleIron\waffle.db)
  --help              This help

MODEL is a file path or a unique substring of the name in the database.
waffle COMMAND --help   Help for that command.
)HELP";

const char* kScanHelp = "waffle scan [folders...]\n\nWalks the folders (or the default model folders) and records each "
                        "GGUF.\n";
const char* kListHelp = "waffle list\n\nShows the library stored in the database.\n";
const char* kInfoHelp = "waffle info MODEL\n\nReads the file and prints the main metadata, then bytes by type.\n";
const char* kPlanHelp = R"HELP(waffle plan MODEL [options]

  --ngl N            GPU layers (-1 = all)
  --ctx N            Context size
  --ctk TYPE         K cache type (f16, q8_0, ...)
  --ctv TYPE         V cache type
  --fa 0|1           Flash attention
  --swa-full 0|1     SWA cache sized to the context (1, llama.cpp default) or to the window only (0)
  --budget-gib X     VRAM budget in GiB (default: free VRAM from telemetry)
  --verify           Load the model on the GPU and measure the gap versus the estimate
)HELP";
const char* kBenchHelp = R"HELP(waffle bench MODEL [options]

Each setting takes a comma-separated list (Cartesian product).

  --ngl LIST         GPU layers, e.g. 0,20,99
  --ctx LIST         Context sizes
  --ctk LIST         K and V cache types
  --fa LIST          Flash attention, 0 or 1
  --pp LIST          Prompt tokens
  --tg LIST          Generated tokens
  --reps LIST        Measured repetitions
  --name NAME        Saved campaign name
)HELP";
const char* kHistoryHelp = "waffle history MODEL\n\nOne point per result whose load path is this model, oldest "
                           "first.\n";
const char* kCampaignsHelp = "waffle campaigns\n\nLists saved campaigns.\n";
const char* kExportHelp = R"HELP(waffle export ID [options]

  --md               Markdown (default)
  --csv              CSV
  -o FILE            Write a file instead of standard output
)HELP";
const char* kGpuHelp = "waffle gpu\n\nPrints one NVML sample, or why telemetry is unavailable.\n";

struct Invocation {
    std::optional<std::filesystem::path> db;
    std::string command;
    std::vector<std::string> args;
    bool general_help = false;
    bool command_help = false;
};

bool is_help(const std::string& arg) {
    return arg == "--help" || arg == "-h";
}

Invocation parse_invocation(const std::vector<std::string>& args) {
    Invocation invocation;
    std::vector<std::string> tokens;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--db") {
            if (i + 1 >= args.size()) throw waffle::Error("Missing value for --db");
            invocation.db = path_from_utf8(args[++i]);
        } else if (arg.rfind("--db=", 0) == 0) {
            if (arg.size() == 5) throw waffle::Error("Missing value for --db");
            invocation.db = path_from_utf8(arg.substr(5));
        } else {
            tokens.push_back(arg);
        }
    }
    if (tokens.empty() || is_help(tokens[0])) {
        invocation.general_help = true;
        if (tokens.empty()) throw waffle::Error("Missing command. Try: waffle --help");
        return invocation;
    }
    invocation.command = tokens[0];
    invocation.args.assign(tokens.begin() + 1, tokens.end());
    for (const auto& arg : invocation.args) {
        if (is_help(arg)) invocation.command_help = true;
    }
    return invocation;
}

struct FlagParser {
    explicit FlagParser(std::vector<std::string> args) : args_(std::move(args)) {}

    bool take_flag(const std::string& name) {
        for (size_t i = 0; i < args_.size(); ++i) {
            if (args_[i] == name) {
                args_.erase(args_.begin() + static_cast<std::ptrdiff_t>(i));
                return true;
            }
        }
        return false;
    }

    std::optional<std::string> take_value(const std::string& name) {
        for (size_t i = 0; i < args_.size(); ++i) {
            const std::string& arg = args_[i];
            std::optional<std::string> value;
            size_t erase_count = 1;
            if (arg == name) {
                if (i + 1 >= args_.size()) throw waffle::Error("Missing value for " + name);
                value = args_[i + 1];
                erase_count = 2;
            } else if (arg.rfind(name + "=", 0) == 0) {
                value = arg.substr(name.size() + 1);
                if (value->empty()) throw waffle::Error("Missing value for " + name);
            }
            if (value) {
                args_.erase(args_.begin() + static_cast<std::ptrdiff_t>(i),
                            args_.begin() + static_cast<std::ptrdiff_t>(i + erase_count));
                return value;
            }
        }
        return std::nullopt;
    }

    std::vector<std::string> positionals() {
        for (const auto& arg : args_) {
            if (!arg.empty() && arg[0] == '-') throw waffle::Error("Unknown option: " + arg);
        }
        return args_;
    }

private:
    std::vector<std::string> args_;
};

struct ResolvedModel {
    std::filesystem::path path;
    std::string key;
    std::optional<waffle::ModelRecord> record;
};

ResolvedModel resolve_model(waffle::Store& store, const std::string& spec) {
    const auto as_path = path_from_utf8(spec);
    std::error_code ec;
    if (std::filesystem::is_regular_file(as_path, ec)) {
        const auto path = normalize_path(as_path);
        const std::string key = utf8_from_path(path);
        return ResolvedModel{path, key, store.find_model(key)};
    }

    std::vector<waffle::ModelRecord> hits;
    for (const auto& model : store.list_models()) {
        if (model.name.find(spec) != std::string::npos || model.file_name.find(spec) != std::string::npos ||
            model.path.find(spec) != std::string::npos)
            hits.push_back(model);
    }
    if (hits.empty()) throw waffle::Error("Model not found: " + spec);
    if (hits.size() > 1) {
        std::string message = "Several models match \"" + spec + "\":";
        const size_t shown = std::min<size_t>(hits.size(), 8);
        for (size_t i = 0; i < shown; ++i) message += "\n  " + hits[i].name + "  " + hits[i].path;
        if (hits.size() > shown) message += "\n  ...";
        throw waffle::Error(message);
    }
    return ResolvedModel{path_from_utf8(hits[0].path), hits[0].path, hits[0]};
}

int cmd_scan(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    std::vector<std::filesystem::path> roots;
    if (positionals.empty()) {
        roots = waffle::default_model_roots();
    } else {
        for (const auto& dir : positionals) roots.push_back(path_from_utf8(dir));
    }
    const auto files = waffle::scan_model_dirs(roots);
    int recorded = 0;
    for (const auto& file : files) {
        try {
            const auto path = normalize_path(file);
            auto info = waffle::read_gguf(path, false);
            if (!waffle::is_language_model(info)) continue;
            info.path = path;
            store.upsert_model(info);
            ++recorded;
        } catch (const waffle::Error& error) {
            std::cerr << error.what() << '\n';
        }
    }
    if (recorded <= 1)
        std::cout << recorded << " model recorded\n";
    else
        std::cout << recorded << " models recorded\n";
    return 0;
}

int cmd_list(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    if (!parser.positionals().empty()) throw waffle::Error("waffle list takes no arguments");
    std::vector<std::vector<std::string>> rows;
    for (const auto& model : store.list_models()) {
        // Record stored before Ollama draft models were recognized: hide it.
        if (const auto blob = waffle::find_ollama_blob(std::filesystem::u8path(model.path)); blob && !blob->model)
            continue;
        rows.push_back({model.name, model.file_type, format_bytes(model.file_size), format_count(model.n_params),
                        std::to_string(model.context_length), model.path});
    }
    if (rows.empty()) {
        std::cout << "No model recorded\n";
        return 0;
    }
    print_table({"Name", "Quant", "Size", "Parameters", "Context", "Path"}, rows);
    return 0;
}

int cmd_info(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw waffle::Error("Usage: waffle info MODEL");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw waffle::Error("Model file not found: " + utf8_from_path(model.path));
    const waffle::ModelInfo info = waffle::read_gguf(model.path, true);
    std::cout << "Name            " << info.name << '\n'
              << "File            " << info.file_name << '\n'
              << "Path            " << utf8_from_path(info.path) << '\n'
              << "Architecture    " << info.architecture << '\n'
              << "Quant           " << info.file_type << '\n'
              << "Size            " << format_bytes(info.file_size) << '\n'
              << "Parameters      " << format_count(info.n_params) << '\n'
              << "Context         " << info.context_length << '\n'
              << "Layers          " << info.n_layer << '\n'
              << "Embedding       " << info.n_embd << '\n'
              << "Heads           " << info.n_head << " (KV " << info.n_head_kv << ")\n"
              << "Head dim K/V    " << info.head_dim_k << " / " << info.head_dim_v << '\n';
    if (info.sliding_window) std::cout << "Window          " << info.sliding_window << '\n';
    if (info.n_expert) std::cout << "Experts         " << info.n_expert << " (active " << info.n_expert_used << ")\n";
    if (model.record && !model.record->notes.empty()) std::cout << "Notes           " << model.record->notes << '\n';
    if (model.record && !model.record->tags.empty()) {
        std::cout << "Tags            ";
        for (size_t i = 0; i < model.record->tags.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << model.record->tags[i];
        }
        std::cout << '\n';
    }
    std::cout << "Bytes by type\n";
    if (info.bytes_by_type.empty()) {
        std::cout << "  (none)\n";
    } else {
        for (const auto& [type, bytes] : info.bytes_by_type)
            std::cout << "  " << type << "  " << format_bytes(bytes) << '\n';
    }
    return 0;
}

void print_estimate(const waffle::VramEstimate& estimate) {
    std::cout << "GPU weights     " << format_bytes(estimate.weights_gpu) << '\n'
              << "CPU weights     " << format_bytes(estimate.weights_cpu) << '\n'
              << "KV cache GPU    " << format_bytes(estimate.kv_cache_gpu) << '\n'
              << "KV cache CPU    " << format_bytes(estimate.kv_cache_cpu) << '\n'
              << "GPU compute     " << format_bytes(estimate.compute_gpu) << '\n'
              << "CPU compute     " << format_bytes(estimate.compute_cpu) << '\n'
              << "GPU reserve     " << format_bytes(estimate.overhead_gpu) << '\n'
              << "GPU total       " << format_bytes(estimate.total_gpu()) << '\n'
              << "CPU total       " << format_bytes(estimate.total_cpu()) << '\n';
    for (const auto& note : estimate.notes) std::cout << "Note            " << note << '\n';
}

// Load the model with the plan settings and compare measured VRAM with the estimate.
void verify_plan(waffle::Store& store, const std::filesystem::path& path, const waffle::VramPlanInput& input,
                 const waffle::VramEstimate& estimate) {
    waffle::Telemetry telemetry;
    const auto before = telemetry.sample();
    if (!before) throw waffle::Error("Cannot verify: " + telemetry.error());

    waffle::LoadParams params;
    params.model_path = path;
    params.n_gpu_layers = input.n_gpu_layers;
    params.n_ctx = input.n_ctx;
    params.n_batch = input.n_batch;
    params.n_ubatch = input.n_ubatch;
    params.type_k = input.type_k;
    params.type_v = input.type_v;
    params.flash_attn = input.flash_attn ? 1 : 0;
    params.n_seq_max = input.n_seq;
    params.swa_full = input.swa_full;

    waffle::Engine engine;
    // Keep the lines where llama.cpp reports its allocations: they explain a gap.
    std::vector<std::string> allocations;
    std::mutex allocations_mutex;
    engine.set_log_sink([&](int, std::string_view text) {
        if (text.find("buffer size") == std::string_view::npos && text.find("cells") == std::string_view::npos) return;
        std::string line(text);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        std::lock_guard lock(allocations_mutex);
        allocations.push_back(std::move(line));
    });
    uint64_t measured = 0;
    engine
        .submit([&](waffle::EngineSession& session) {
            session.load(params);
            // A first decode allocates the final compute buffers.
            const auto tokens = session.tokenize("Hello", true);
            session.decode(tokens, 0);
            if (const auto after = telemetry.sample())
                measured = after->vram_used > before->vram_used ? after->vram_used - before->vram_used : 0;
            session.unload();
        })
        .get();

    const uint64_t estimated = estimate.total_gpu();
    const double gap = estimated ? (static_cast<double>(measured) - static_cast<double>(estimated)) /
                                       static_cast<double>(estimated) * 100.0
                                 : 0.0;
    std::ostringstream gap_text;
    gap_text << std::showpos << std::fixed << std::setprecision(1) << gap << " %";
    std::cout << "Measured GPU    " << format_bytes(measured) << '\n'
              << "Gap             " << gap_text.str() << " (measured - estimate)\n";
    if (!allocations.empty()) {
        std::cout << "Allocations reported by llama.cpp:\n";
        for (const auto& line : allocations) std::cout << "  " << line << '\n';
    }

    waffle::VramCheck check;
    check.model_path = utf8_from_path(path);
    check.timestamp = now_iso();
    check.config = "ngl=" + std::to_string(input.n_gpu_layers) + " ctx=" + std::to_string(input.n_ctx) +
                   " kv=" + input.type_k + "/" + input.type_v + " fa=" + (input.flash_attn ? "on" : "off") +
                   (input.swa_full ? "" : " swa=window");
    check.estimated_gpu = estimated;
    check.measured_gpu = measured;
    store.save_vram_check(check);
}

int cmd_plan(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto ngl = parser.take_value("--ngl");
    const auto ctx = parser.take_value("--ctx");
    const auto ctk = parser.take_value("--ctk");
    const auto ctv = parser.take_value("--ctv");
    const auto fa = parser.take_value("--fa");
    const auto swa_full = parser.take_value("--swa-full");
    const auto budget_text = parser.take_value("--budget-gib");
    const bool verify = parser.take_flag("--verify");
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw waffle::Error("Usage: waffle plan MODEL [options]");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw waffle::Error("Model file not found: " + utf8_from_path(model.path));

    waffle::VramPlanInput input;
    if (ngl) input.n_gpu_layers = static_cast<int>(parse_integer(*ngl, "--ngl"));
    if (ctx) input.n_ctx = parse_u32_at_least(*ctx, "--ctx", 1);
    if (ctk) input.type_k = *ctk;
    if (ctv) input.type_v = *ctv;
    if (fa) {
        const long long value = parse_integer(*fa, "--fa");
        if (value != 0 && value != 1) throw waffle::Error("--fa expects 0 or 1");
        input.flash_attn = value == 1;
    }
    if (swa_full) {
        const long long value = parse_integer(*swa_full, "--swa-full");
        if (value != 0 && value != 1) throw waffle::Error("--swa-full expects 0 or 1");
        input.swa_full = value == 1;
    }

    const waffle::ModelInfo info = waffle::read_gguf(model.path, true);
    const waffle::VramEstimate estimate = waffle::estimate_vram(info, input);
    print_estimate(estimate);

    std::optional<uint64_t> budget;
    if (budget_text) {
        const double gib = parse_double(*budget_text, "--budget-gib");
        if (!(gib > 0)) throw waffle::Error("--budget-gib must be positive");
        budget = static_cast<uint64_t>(gib * static_cast<double>(waffle::GiB));
        std::cout << "Budget          " << format_bytes(*budget) << '\n';
    } else {
        waffle::Telemetry telemetry;
        if (const auto sample = telemetry.sample()) {
            // VRAM already taken by other programs is not available for the model.
            budget = sample->vram_total - std::min(sample->vram_used, sample->vram_total);
            std::cout << "Budget          " << format_bytes(*budget) << " (free VRAM)\n";
        } else {
            std::cout << "Budget          unavailable: " << telemetry.error() << '\n';
        }
    }
    if (budget) {
        const int layers = waffle::max_gpu_layers_for(info, input, *budget);
        std::cout << "Max GPU layers  " << layers << '\n';
    }
    if (verify) verify_plan(store, model.path, input, estimate);
    return 0;
}

bool show_progress(float fraction, const std::string& label) {
    const float clamped = std::clamp(fraction, 0.f, 1.f);
    constexpr int width = 24;
    const int filled = std::clamp(static_cast<int>(std::lround(clamped * width)), 0, width);
    std::cerr << '\r' << '[' << std::string(static_cast<size_t>(filled), '#')
              << std::string(static_cast<size_t>(width - filled), '-') << "] " << std::setw(3)
              << static_cast<int>(std::lround(clamped * 100.0)) << "% " << label << "   " << std::flush;
    return true;
}

int cmd_bench(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto ngl_text = parser.take_value("--ngl");
    const auto ctx_text = parser.take_value("--ctx");
    const auto ctk_text = parser.take_value("--ctk");
    const auto fa_text = parser.take_value("--fa");
    const auto pp_text = parser.take_value("--pp");
    const auto tg_text = parser.take_value("--tg");
    const auto reps_text = parser.take_value("--reps");
    const auto name_text = parser.take_value("--name");
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw waffle::Error("Usage: waffle bench MODEL [options]");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw waffle::Error("Model file not found: " + utf8_from_path(model.path));

    waffle::CampaignGrid grid;
    grid.base.load.model_path = model.path;
    if (ngl_text) {
        for (const auto& part : split_comma(*ngl_text, "--ngl"))
            grid.n_gpu_layers.push_back(static_cast<int>(parse_integer(part, "--ngl")));
        grid.base.load.n_gpu_layers = grid.n_gpu_layers.front();
    }
    if (ctx_text) {
        for (const auto& part : split_comma(*ctx_text, "--ctx"))
            grid.n_ctx.push_back(parse_u32_at_least(part, "--ctx", 1));
        grid.base.load.n_ctx = grid.n_ctx.front();
    }
    if (ctk_text) {
        grid.kv_type = split_comma(*ctk_text, "--ctk");
        grid.base.load.type_k = grid.kv_type.front();
        grid.base.load.type_v = grid.kv_type.front();
    }
    if (fa_text) {
        for (const auto& part : split_comma(*fa_text, "--fa")) {
            const long long value = parse_integer(part, "--fa");
            if (value != 0 && value != 1) throw waffle::Error("--fa expects 0 or 1");
            grid.flash_attn.push_back(static_cast<int>(value));
        }
        grid.base.load.flash_attn = grid.flash_attn.front();
    }

    std::vector<uint32_t> prompts{grid.base.n_prompt};
    std::vector<uint32_t> generated{grid.base.n_gen};
    std::vector<int> repetitions{grid.base.repetitions};
    if (pp_text) {
        prompts.clear();
        for (const auto& part : split_comma(*pp_text, "--pp")) prompts.push_back(parse_u32_at_least(part, "--pp", 1));
    }
    if (tg_text) {
        generated.clear();
        for (const auto& part : split_comma(*tg_text, "--tg")) generated.push_back(parse_u32_at_least(part, "--tg", 1));
    }
    if (reps_text) {
        repetitions.clear();
        for (const auto& part : split_comma(*reps_text, "--reps")) {
            const long long value = parse_integer(part, "--reps");
            if (value < 1) throw waffle::Error("--reps must be at least 1");
            repetitions.push_back(static_cast<int>(value));
        }
    }

    waffle::Engine engine;
    waffle::Telemetry telemetry;
    waffle::Telemetry* telemetry_ptr = telemetry.available() ? &telemetry : nullptr;
    std::vector<waffle::BenchResult> results;
    std::vector<std::string> errors;
    bool printed = false;
    for (uint32_t n_prompt : prompts) {
        for (uint32_t n_gen : generated) {
            for (int reps : repetitions) {
                waffle::CampaignGrid one = grid;
                one.base.n_prompt = n_prompt;
                one.base.n_gen = n_gen;
                one.base.repetitions = reps;
                auto batch = waffle::run_campaign(engine, telemetry_ptr, one, show_progress, &errors);
                printed = true;
                results.insert(results.end(), std::make_move_iterator(batch.begin()),
                               std::make_move_iterator(batch.end()));
            }
        }
    }
    if (printed) std::cerr << '\n';
    for (const auto& error : errors) std::cerr << error << '\n';
    if (results.empty()) throw waffle::Error("No measurement recorded");

    std::string markdown = waffle::to_markdown(results);
    if (markdown.empty() || markdown.back() != '\n') markdown.push_back('\n');
    std::cout << markdown;

    const std::string name = name_text && !name_text->empty() ? *name_text : "waffle " + now_iso();
    const int64_t id = store.save_campaign(name, results);
    std::cerr << "Campaign " << id << " saved (" << name << ")\n";
    return errors.empty() ? 0 : 1;
}

int cmd_history(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw waffle::Error("Usage: waffle history MODEL");
    const auto model = resolve_model(store, positionals[0]);
    const auto points = store.history(model.key);
    if (points.empty()) {
        std::cout << "No measurements for this model\n";
        return 0;
    }
    std::vector<std::vector<std::string>> rows;
    for (const auto& point : points) {
        std::ostringstream pp;
        std::ostringstream tg;
        std::ostringstream ttft;
        pp << std::fixed << std::setprecision(1) << point.pp_mean << " ± " << point.pp_stddev;
        tg << std::fixed << std::setprecision(1) << point.tg_mean << " ± " << point.tg_stddev;
        ttft << std::fixed << std::setprecision(1) << point.ttft_mean;
        rows.push_back({point.timestamp, point.llama_version, point.config, pp.str(), tg.str(), ttft.str()});
    }
    print_table({"Date", "llama.cpp", "Config", "pp t/s", "tg t/s", "ttft ms"}, rows);
    return 0;
}

int cmd_campaigns(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    if (!parser.positionals().empty()) throw waffle::Error("waffle campaigns takes no arguments");
    const auto campaigns = store.list_campaigns();
    if (campaigns.empty()) {
        std::cout << "No campaign\n";
        return 0;
    }
    std::vector<std::vector<std::string>> rows;
    for (const auto& campaign : campaigns) {
        rows.push_back({std::to_string(campaign.id), campaign.created_at, std::to_string(campaign.n_results),
                        campaign.model_names, campaign.name});
    }
    print_table({"Id", "Date", "Results", "Models", "Name"}, rows);
    return 0;
}

int cmd_export(waffle::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const bool csv = parser.take_flag("--csv");
    const bool markdown = parser.take_flag("--md");
    if (csv && markdown) throw waffle::Error("Choose --csv or --md");
    const auto out = parser.take_value("-o");
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw waffle::Error("Usage: waffle export ID [--csv|--md] [-o FILE]");
    const long long id_value = parse_integer(positionals[0], "ID");
    if (id_value < 1) throw waffle::Error("Invalid campaign id");
    const auto results = store.load_campaign(id_value);
    std::string text = csv ? waffle::to_csv(results) : waffle::to_markdown(results);
    if (!text.empty() && text.back() != '\n') text.push_back('\n');
    if (!out) {
        std::cout << text;
        return 0;
    }
    const auto path = path_from_utf8(*out);
    if (!path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) throw waffle::Error("Cannot create the folder: " + utf8_from_path(path.parent_path()));
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) throw waffle::Error("Cannot write: " + utf8_from_path(path));
    file << text;
    if (!file) throw waffle::Error("Cannot write: " + utf8_from_path(path));
    return 0;
}

int cmd_gpu() {
    waffle::Telemetry telemetry;
    if (!telemetry.available()) {
        std::cout << "Telemetry unavailable: " << telemetry.error() << '\n';
        return 0;
    }
    const auto sample = telemetry.sample();
    if (!sample) {
        std::cout << "Telemetry unavailable: " << telemetry.error() << '\n';
        return 0;
    }
    std::cout << "Device          " << telemetry.device_name() << '\n'
              << "Utilization     " << sample->gpu_util_pct << " %\n"
              << "Memory          " << sample->mem_util_pct << " %\n"
              << "VRAM            " << format_bytes(sample->vram_used) << " / " << format_bytes(sample->vram_total)
              << '\n'
              << "Temperature     " << sample->temperature_c << " °C\n"
              << "Power           " << std::fixed << std::setprecision(1) << sample->power_w << " W\n"
              << "SM clock        " << sample->sm_clock_mhz << " MHz\n";
    return 0;
}

int dispatch(const Invocation& invocation) {
    if (invocation.general_help) {
        std::cout << kGeneralHelp;
        return 0;
    }
    const std::string& command = invocation.command;
    auto help_of = [&]() -> const char* {
        if (command == "scan") return kScanHelp;
        if (command == "list") return kListHelp;
        if (command == "info") return kInfoHelp;
        if (command == "plan") return kPlanHelp;
        if (command == "bench") return kBenchHelp;
        if (command == "history") return kHistoryHelp;
        if (command == "campaigns") return kCampaignsHelp;
        if (command == "export") return kExportHelp;
        if (command == "gpu") return kGpuHelp;
        return nullptr;
    };
    if (invocation.command_help) {
        if (const char* text = help_of()) {
            std::cout << text;
            return 0;
        }
        throw waffle::Error("Unknown command: " + command);
    }
    if (command == "gpu") {
        FlagParser parser(invocation.args);
        if (!parser.positionals().empty()) throw waffle::Error("waffle gpu takes no arguments");
        return cmd_gpu();
    }
    if (!help_of()) throw waffle::Error("Unknown command: " + command + ". Try: waffle --help");

    waffle::Store store(invocation.db ? *invocation.db : waffle::Store::default_path());
    if (command == "scan") return cmd_scan(store, invocation.args);
    if (command == "list") return cmd_list(store, invocation.args);
    if (command == "info") return cmd_info(store, invocation.args);
    if (command == "plan") return cmd_plan(store, invocation.args);
    if (command == "bench") return cmd_bench(store, invocation.args);
    if (command == "history") return cmd_history(store, invocation.args);
    if (command == "campaigns") return cmd_campaigns(store, invocation.args);
    if (command == "export") return cmd_export(store, invocation.args);
    throw waffle::Error("Unknown command: " + command);
}

} // namespace

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        const std::vector<std::string> args = command_line();
#else
        std::vector<std::string> args;
        args.reserve(argc > 0 ? static_cast<size_t>(argc - 1) : 0);
        for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
#endif
        (void)argc;
        (void)argv;
        if (args.empty()) {
            std::cout << kGeneralHelp;
            return 0;
        }
        return dispatch(parse_invocation(args));
    } catch (const waffle::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
