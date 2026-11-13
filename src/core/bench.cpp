#include "core/bench.hpp"
#include "core/engine.hpp"
#include "core/gguf_reader.hpp"
#include "core/json.hpp"
#include "core/telemetry.hpp"
#include "core/util.hpp"

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

namespace waffle {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double median_of(const std::vector<double>& sorted) {
    const size_t n = sorted.size();
    return n % 2 ? sorted[n / 2] : std::midpoint(sorted[n / 2 - 1], sorted[n / 2]);
}
std::string utc_now() {
    return now_iso();
}
std::string csv_cell(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + '"';
}
std::string markdown_cell(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '\r') continue;
        if (c == '\n') {
            out += "<br>";
            continue;
        }
        if (c == '|' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}
void report(Engine& engine, const ProgressFn& progress, float fraction, const std::string& label) {
    if (progress && !progress(fraction, label)) {
        engine.cancel_all();
        throw Cancelled();
    }
}
} // namespace

Stats compute_stats(std::span<const double> values) {
    Stats out;
    if (values.empty()) return out;
    for (double x : values)
        if (!std::isfinite(x)) throw Error("Statistics require finite values");
    out.n = static_cast<int>(values.size());
    // Welford avoids subtracting two large, nearly equal squares.
    double m2 = 0;
    size_t count = 0;
    for (double x : values) {
        const double delta = x - out.mean;
        out.mean += delta / static_cast<double>(++count);
        m2 += delta * (x - out.mean);
    }
    out.stddev = count > 1 ? std::sqrt(std::max(0.0, m2 / static_cast<double>(count - 1))) : 0;
    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    out.min = sorted.front();
    out.max = sorted.back();
    out.median = median_of(sorted);
    std::vector<double> deviations;
    for (double x : values) deviations.push_back(std::abs(x - out.median));
    std::sort(deviations.begin(), deviations.end());
    const double mad = median_of(deviations);
    if (mad > 0) {
        for (size_t i = 0; i < values.size(); ++i)
            if (0.6745 * std::abs(values[i] - out.median) / mad > 3.5) out.outliers.push_back(static_cast<int>(i));
    }
    return out;
}

BenchSample run_bench_once(EngineSession& session, const BenchConfig& cfg) {
    const auto& info = session.info();
    if (!cfg.n_prompt || !cfg.n_gen || uint64_t(cfg.n_prompt) + cfg.n_gen > info.n_ctx)
        throw Error("The bench needs a non-empty prompt and generation that fit in the context");
    session.throw_if_stopped();
    std::vector<llama_token> ordinary;
    for (int i = 0; i < session.n_vocab(); ++i) {
        const auto attr = llama_vocab_get_attr(session.vocab(), i);
        if ((attr & (LLAMA_TOKEN_ATTR_NORMAL | LLAMA_TOKEN_ATTR_BYTE)) &&
            !(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED)) &&
            !session.is_eog(i))
            ordinary.push_back(i);
    }
    if (ordinary.empty()) throw Error("Vocabulary has no ordinary token");
    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<size_t> pick(0, ordinary.size() - 1);
    std::vector<llama_token> prompt(cfg.n_prompt);
    for (auto& token : prompt) token = ordinary[pick(rng)];
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(),
                                                                          llama_sampler_free);
    session.clear_memory();
    llama_perf_context_reset(session.ctx());
    BenchSample sample;
    const auto start = Clock::now();
    session.decode(prompt, 0);
    llama_synchronize(session.ctx());
    sample.pp_tps = cfg.n_prompt / seconds(start);
    const auto generation = Clock::now();
    for (uint32_t i = 0; i < cfg.n_gen; ++i) {
        session.throw_if_stopped();
        const llama_token token = llama_sampler_sample(sampler.get(), session.ctx(), -1);
        llama_sampler_accept(sampler.get(), token);
        session.decode(std::span(&token, 1), static_cast<int32_t>(cfg.n_prompt + i));
        llama_synchronize(session.ctx());
        if (i == 0) sample.ttft_ms = seconds(start) * 1000;
    }
    sample.tg_tps = cfg.n_gen / seconds(generation);
    return sample;
}

BenchResult run_bench(Engine& engine, Telemetry* telemetry, const BenchConfig& cfg, const ProgressFn& progress) {
    if (cfg.repetitions < 1 || cfg.warmup < 0) throw Error("Invalid repetition or warmup count");
    std::optional<uint64_t> baseline;
    if (telemetry) {
        if (auto sample = telemetry->sample()) baseline = sample->vram_used;
    }
    BenchResult result;
    result.config = cfg;
    result.timestamp = utc_now();
    result.llama_version = WAFFLE_LLAMA_VERSION;
    report(engine, progress, 0, "Loading model");
    engine
        .submit([&](EngineSession& s) {
            engine.set_status("Loading model");
            const auto& info = s.load(cfg.load);
            result.load_ms = info.load_ms;
            // Name and quantization read from the GGUF header; the llama.cpp description is the fallback.
            result.model_name = info.description;
            result.model_file_type = info.description;
            try {
                const ModelInfo gguf = read_gguf(cfg.load.model_path, false);
                result.model_name = gguf.name;
                result.model_file_type = gguf.file_type;
            } catch (const Error&) {
                result.model_name = info.description;
                result.model_file_type = info.description;
            }
            result.gpu_name = telemetry ? telemetry->device_name() : s.device_memory().name;
        })
        .get();
    const int64_t total = int64_t(cfg.warmup) + cfg.repetitions;
    for (int64_t i = 0; i < total; ++i) {
        const bool warmup = i < cfg.warmup;
        const auto label = warmup ? "Warmup" : "Run " + std::to_string(i - cfg.warmup + 1);
        report(engine, progress, static_cast<float>(i) / static_cast<float>(total), label);
        auto sample = engine
                          .submit([&](EngineSession& s) {
                              engine.set_status(label);
                              std::unique_ptr<VramPeakWatcher> watcher;
                              if (telemetry)
                                  watcher = std::make_unique<VramPeakWatcher>(*telemetry, std::chrono::milliseconds(50),
                                                                              baseline);
                              auto measured = run_bench_once(s, cfg);
                              if (watcher) measured.vram_peak = watcher->stop();
                              return measured;
                          })
                          .get();
        if (!warmup) {
            result.vram_peak = std::max(result.vram_peak, sample.vram_peak);
            result.samples.push_back(sample);
        }
    }
    report(engine, progress, 1, "Runs finished");
    std::vector<double> pp, tg, ttft;
    for (const auto& s : result.samples) {
        pp.push_back(s.pp_tps);
        tg.push_back(s.tg_tps);
        ttft.push_back(s.ttft_ms);
    }
    result.pp = compute_stats(pp);
    result.tg = compute_stats(tg);
    result.ttft = compute_stats(ttft);
    return result;
}

std::vector<BenchConfig> expand_grid(const CampaignGrid& grid) {
    std::vector<BenchConfig> result{grid.base};
    auto expand = [&](const auto& values, auto assign) {
        if (values.empty()) return;
        std::vector<BenchConfig> next;
        for (const auto& config : result)
            for (const auto& value : values) {
                auto copy = config;
                assign(copy, value);
                next.push_back(std::move(copy));
            }
        result = std::move(next);
    };
    expand(grid.n_gpu_layers, [](auto& c, int v) { c.load.n_gpu_layers = v; });
    expand(grid.n_ctx, [](auto& c, uint32_t v) { c.load.n_ctx = v; });
    expand(grid.kv_type, [](auto& c, const auto& v) { c.load.type_k = c.load.type_v = v; });
    expand(grid.flash_attn, [](auto& c, int v) { c.load.flash_attn = v; });
    expand(grid.n_ubatch, [](auto& c, uint32_t v) { c.load.n_ubatch = v; });
    expand(grid.models, [](auto& c, const auto& v) { c.load.model_path = v; });
    return result;
}

std::vector<BenchResult> run_campaign(Engine& engine, Telemetry* telemetry, const CampaignGrid& grid,
                                      const ProgressFn& progress, std::vector<std::string>* errors) {
    const auto configs = expand_grid(grid);
    std::vector<BenchResult> results;
    for (size_t i = 0; i < configs.size(); ++i) {
        try {
            results.push_back(run_bench(engine, telemetry, configs[i], [&](float f, const std::string& label) {
                return !progress || progress((static_cast<float>(i) + f) / static_cast<float>(configs.size()), label);
            }));
        } catch (const Cancelled&) {
            throw;
        } catch (const std::exception& e) {
            if (errors) errors->push_back(config_label(configs[i]) + ": " + e.what());
        }
    }
    return results;
}

std::string config_label(const BenchConfig& cfg) {
    const auto& p = cfg.load;
    return "ngl=" + std::to_string(p.n_gpu_layers) + " ctx=" + std::to_string(p.n_ctx) + " kv=" + p.type_k +
           (p.type_k == p.type_v ? "" : "/" + p.type_v) + " fa=" +
           (p.flash_attn < 0 ? "auto"
            : p.flash_attn   ? "on"
                             : "off") +
           " ub=" + std::to_string(p.n_ubatch) + (p.swa_full ? " swa=full" : " swa=window") +
           (cfg.seed == 42 ? "" : " seed=" + std::to_string(cfg.seed));
}

std::string to_csv(const std::vector<BenchResult>& results) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12);
    // Stable schema: identity, settings, sample count, means/standard deviations (tokens/s, ms), VRAM (bytes),
    // provenance.
    out << "model,model_file_type,n_gpu_layers,n_ctx,type_k,type_v,flash_attn,n_ubatch,n_batch,n_threads,n_prompt,n_"
           "gen,repetitions,warmup,samples,pp_mean,pp_stddev,tg_mean,tg_stddev,ttft_mean_ms,ttft_stddev_ms,load_ms,"
           "vram_peak_bytes,gpu,llama_version,timestamp,swa_full,seed\n";
    for (const auto& r : results) {
        const auto& c = r.config;
        const auto& p = c.load;
        out << csv_cell(r.model_name) << ',' << csv_cell(r.model_file_type) << ',' << p.n_gpu_layers << ',' << p.n_ctx
            << ',' << csv_cell(p.type_k) << ',' << csv_cell(p.type_v) << ',' << p.flash_attn << ',' << p.n_ubatch << ','
            << p.n_batch << ',' << p.n_threads << ',' << c.n_prompt << ',' << c.n_gen << ',' << c.repetitions << ','
            << c.warmup << ',' << r.samples.size() << ',' << r.pp.mean << ',' << r.pp.stddev << ',' << r.tg.mean << ','
            << r.tg.stddev << ',' << r.ttft.mean << ',' << r.ttft.stddev << ',' << r.load_ms << ',' << r.vram_peak
            << ',' << csv_cell(r.gpu_name) << ',' << csv_cell(r.llama_version) << ',' << csv_cell(r.timestamp) << ','
            << (p.swa_full ? 1 : 0) << ',' << c.seed << '\n';
    }
    return out.str();
}

std::string to_markdown(const std::vector<BenchResult>& results) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(2);
    out << "| Model | Configuration | n | pp (tokens/s) | tg (tokens/s) | TTFT (ms) | Load (ms) | VRAM peak (GiB) "
           "|\n"
           "|---|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : results)
        out << "| " << markdown_cell(r.model_name) << " | " << markdown_cell(config_label(r.config)) << " | "
            << r.samples.size() << " | " << r.pp.mean << " ± " << r.pp.stddev << " | " << r.tg.mean << " ± "
            << r.tg.stddev << " | " << r.ttft.mean << " ± " << r.ttft.stddev << " | " << r.load_ms << " | "
            << static_cast<double>(r.vram_peak) / GiB << " |\n";
    return out.str();
}

namespace {

Json stats_json(const Stats& stats) {
    std::map<std::string, Json> fields;
    fields.emplace("n", Json::number_value(stats.n));
    fields.emplace("mean", Json::number_value(stats.mean));
    fields.emplace("stddev", Json::number_value(stats.stddev));
    fields.emplace("median", Json::number_value(stats.median));
    fields.emplace("min", Json::number_value(stats.min));
    fields.emplace("max", Json::number_value(stats.max));
    std::vector<Json> outliers;
    for (int index : stats.outliers) outliers.push_back(Json::number_value(index));
    fields.emplace("outliers", Json::array_value(std::move(outliers)));
    return Json::object_value(std::move(fields));
}

Stats stats_from(const Json* value) {
    Stats stats;
    if (!value) return stats;
    stats.n = static_cast<int>(value->find("n") ? value->find("n")->as_int() : 0);
    stats.mean = value->find("mean") ? value->find("mean")->as_number() : 0;
    stats.stddev = value->find("stddev") ? value->find("stddev")->as_number() : 0;
    stats.median = value->find("median") ? value->find("median")->as_number() : 0;
    stats.min = value->find("min") ? value->find("min")->as_number() : 0;
    stats.max = value->find("max") ? value->find("max")->as_number() : 0;
    if (const Json* outliers = value->find("outliers"); outliers && outliers->kind == Json::Kind::Array) {
        for (const auto& item : outliers->items) stats.outliers.push_back(static_cast<int>(item.as_int()));
    }
    return stats;
}

const Json& require_object(const Json& value, const char* what) {
    if (value.kind != Json::Kind::Object) throw Error(std::string("JSON: expected ") + what);
    return value;
}

} // namespace

std::string results_to_json(const std::vector<BenchResult>& results) {
    std::vector<Json> rows;
    for (const auto& result : results) {
        const auto& load = result.config.load;
        std::map<std::string, Json> fields;
        fields.emplace("model_name", Json::string_value(result.model_name));
        fields.emplace("model_file_type", Json::string_value(result.model_file_type));
        fields.emplace("model_path", Json::string_value(utf8_from_path(load.model_path)));
        fields.emplace("n_gpu_layers", Json::number_value(load.n_gpu_layers));
        fields.emplace("n_ctx", Json::number_value(load.n_ctx));
        fields.emplace("n_batch", Json::number_value(load.n_batch));
        fields.emplace("n_ubatch", Json::number_value(load.n_ubatch));
        fields.emplace("type_k", Json::string_value(load.type_k));
        fields.emplace("type_v", Json::string_value(load.type_v));
        fields.emplace("flash_attn", Json::number_value(load.flash_attn));
        fields.emplace("n_threads", Json::number_value(load.n_threads));
        fields.emplace("n_seq_max", Json::number_value(load.n_seq_max));
        fields.emplace("swa_full", Json::boolean_value(load.swa_full));
        fields.emplace("n_prompt", Json::number_value(result.config.n_prompt));
        fields.emplace("n_gen", Json::number_value(result.config.n_gen));
        fields.emplace("repetitions", Json::number_value(result.config.repetitions));
        fields.emplace("warmup", Json::number_value(result.config.warmup));
        fields.emplace("seed", Json::number_value(result.config.seed));
        fields.emplace("load_ms", Json::number_value(result.load_ms));
        fields.emplace("vram_peak", Json::number_value(static_cast<double>(result.vram_peak)));
        fields.emplace("gpu_name", Json::string_value(result.gpu_name));
        fields.emplace("llama_version", Json::string_value(result.llama_version));
        fields.emplace("timestamp", Json::string_value(result.timestamp));
        fields.emplace("pp", stats_json(result.pp));
        fields.emplace("tg", stats_json(result.tg));
        fields.emplace("ttft", stats_json(result.ttft));
        std::vector<Json> samples;
        for (const auto& sample : result.samples) {
            std::map<std::string, Json> one;
            one.emplace("pp_tps", Json::number_value(sample.pp_tps));
            one.emplace("tg_tps", Json::number_value(sample.tg_tps));
            one.emplace("ttft_ms", Json::number_value(sample.ttft_ms));
            one.emplace("vram_peak", Json::number_value(static_cast<double>(sample.vram_peak)));
            samples.push_back(Json::object_value(std::move(one)));
        }
        fields.emplace("samples", Json::array_value(std::move(samples)));
        rows.push_back(Json::object_value(std::move(fields)));
    }
    std::map<std::string, Json> root;
    root.emplace("version", Json::number_value(1));
    root.emplace("results", Json::array_value(std::move(rows)));
    return Json::object_value(std::move(root)).dump();
}

std::vector<BenchResult> results_from_json(std::string_view text) {
    const Json root = require_object(parse_json(text), "an object");
    const Json* rows = root.find("results");
    if (!rows || rows->kind != Json::Kind::Array) throw Error("JSON: expected a results array");
    std::vector<BenchResult> loaded;
    for (const auto& row : rows->items) {
        require_object(row, "a result object");
        BenchResult result;
        result.model_name = row.find("model_name") ? row.find("model_name")->as_string() : "";
        result.model_file_type = row.find("model_file_type") ? row.find("model_file_type")->as_string() : "";
        auto& load = result.config.load;
        load.model_path = path_from_utf8(row.find("model_path") ? row.find("model_path")->as_string() : "");
        load.n_gpu_layers = row.find("n_gpu_layers") ? static_cast<int>(row.find("n_gpu_layers")->as_int()) : -1;
        load.n_ctx = row.find("n_ctx") ? static_cast<uint32_t>(row.find("n_ctx")->as_int()) : 0;
        load.n_batch = row.find("n_batch") ? static_cast<uint32_t>(row.find("n_batch")->as_int()) : 0;
        load.n_ubatch = row.find("n_ubatch") ? static_cast<uint32_t>(row.find("n_ubatch")->as_int()) : 0;
        load.type_k = row.find("type_k") ? row.find("type_k")->as_string("f16") : "f16";
        load.type_v = row.find("type_v") ? row.find("type_v")->as_string("f16") : "f16";
        load.flash_attn = row.find("flash_attn") ? static_cast<int>(row.find("flash_attn")->as_int(-1)) : -1;
        load.n_threads = row.find("n_threads") ? static_cast<int>(row.find("n_threads")->as_int()) : 0;
        load.n_seq_max = row.find("n_seq_max") ? static_cast<uint32_t>(row.find("n_seq_max")->as_int(1)) : 1;
        load.swa_full = row.find("swa_full") ? row.find("swa_full")->as_bool(true) : true;
        result.config.n_prompt = row.find("n_prompt") ? static_cast<uint32_t>(row.find("n_prompt")->as_int()) : 0;
        result.config.n_gen = row.find("n_gen") ? static_cast<uint32_t>(row.find("n_gen")->as_int()) : 0;
        result.config.repetitions = row.find("repetitions") ? static_cast<int>(row.find("repetitions")->as_int()) : 0;
        result.config.warmup = row.find("warmup") ? static_cast<int>(row.find("warmup")->as_int()) : 0;
        result.config.seed = row.find("seed") ? static_cast<uint32_t>(row.find("seed")->as_int(42)) : 42;
        result.load_ms = row.find("load_ms") ? row.find("load_ms")->as_number() : 0;
        result.vram_peak = row.find("vram_peak") ? static_cast<uint64_t>(row.find("vram_peak")->as_int()) : 0;
        result.gpu_name = row.find("gpu_name") ? row.find("gpu_name")->as_string() : "";
        result.llama_version = row.find("llama_version") ? row.find("llama_version")->as_string() : "";
        result.timestamp = row.find("timestamp") ? row.find("timestamp")->as_string() : "";
        result.pp = stats_from(row.find("pp"));
        result.tg = stats_from(row.find("tg"));
        result.ttft = stats_from(row.find("ttft"));
        if (const Json* samples = row.find("samples"); samples && samples->kind == Json::Kind::Array) {
            for (const auto& sample : samples->items) {
                BenchSample one;
                one.pp_tps = sample.find("pp_tps") ? sample.find("pp_tps")->as_number() : 0;
                one.tg_tps = sample.find("tg_tps") ? sample.find("tg_tps")->as_number() : 0;
                one.ttft_ms = sample.find("ttft_ms") ? sample.find("ttft_ms")->as_number() : 0;
                one.vram_peak =
                    sample.find("vram_peak") ? static_cast<uint64_t>(sample.find("vram_peak")->as_int()) : 0;
                result.samples.push_back(one);
            }
        }
        loaded.push_back(std::move(result));
    }
    return loaded;
}

} // namespace waffle
