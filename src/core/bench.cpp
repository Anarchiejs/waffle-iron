#include "core/bench.hpp"
#include "core/engine.hpp"
#include "core/gguf_reader.hpp"
#include "core/telemetry.hpp"

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

namespace banc {
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
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
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
        if (!std::isfinite(x)) throw Error("Les statistiques exigent des valeurs finies");
    out.n = static_cast<int>(values.size());
    // Welford évite la soustraction de deux grands carrés presque égaux.
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
        throw Error("Le banc exige une consigne et une génération non vides tenant dans le contexte");
    session.throw_if_stopped();
    std::vector<llama_token> ordinary;
    for (int i = 0; i < session.n_vocab(); ++i) {
        const auto attr = llama_vocab_get_attr(session.vocab(), i);
        if ((attr & (LLAMA_TOKEN_ATTR_NORMAL | LLAMA_TOKEN_ATTR_BYTE)) &&
            !(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED)) &&
            !session.is_eog(i))
            ordinary.push_back(i);
    }
    if (ordinary.empty()) throw Error("Le vocabulaire ne contient aucun token ordinaire");
    std::mt19937 rng(42);
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
    if (cfg.repetitions < 1 || cfg.warmup < 0) throw Error("Nombre de répétitions ou de préchauffages invalide");
    BenchResult result;
    result.config = cfg;
    result.timestamp = utc_now();
    result.llama_version = BANC_LLAMA_VERSION;
    report(engine, progress, 0, "Chargement du modèle");
    engine
        .submit([&](EngineSession& s) {
            engine.set_status("Chargement du modèle");
            const auto& info = s.load(cfg.load);
            result.load_ms = info.load_ms;
            // Nom et quantification lus dans l'en-tête GGUF ; la description de llama.cpp sert de secours.
            result.model_name = info.description;
            result.model_file_type = info.description;
            try {
                const ModelInfo gguf = read_gguf(cfg.load.model_path, false);
                result.model_name = gguf.name;
                result.model_file_type = gguf.file_type;
            } catch (const Error&) {
            }
            result.gpu_name = telemetry ? telemetry->device_name() : s.device_memory().name;
        })
        .get();
    const int64_t total = int64_t(cfg.warmup) + cfg.repetitions;
    for (int64_t i = 0; i < total; ++i) {
        const bool warmup = i < cfg.warmup;
        const auto label = warmup ? "Préchauffage" : "Mesure " + std::to_string(i - cfg.warmup + 1);
        report(engine, progress, static_cast<float>(i) / static_cast<float>(total), label);
        auto sample = engine
                          .submit([&](EngineSession& s) {
                              engine.set_status(label);
                              std::unique_ptr<VramPeakWatcher> watcher;
                              if (telemetry) watcher = std::make_unique<VramPeakWatcher>(*telemetry);
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
    report(engine, progress, 1, "Mesures terminées");
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
            if (errors) errors->push_back(config_label(configs[i]) + " : " + e.what());
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
           " ub=" + std::to_string(p.n_ubatch);
}

std::string to_csv(const std::vector<BenchResult>& results) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12);
    // Schéma stable : identité, réglages, effectif, moyennes/écarts-types (tokens/s, ms), VRAM (octets), provenance.
    out << "model,model_file_type,n_gpu_layers,n_ctx,type_k,type_v,flash_attn,n_ubatch,n_batch,n_threads,n_prompt,n_"
           "gen,repetitions,warmup,samples,pp_mean,pp_stddev,tg_mean,tg_stddev,ttft_mean_ms,ttft_stddev_ms,load_ms,"
           "vram_peak_bytes,gpu,llama_version,timestamp\n";
    for (const auto& r : results) {
        const auto& c = r.config;
        const auto& p = c.load;
        out << csv_cell(r.model_name) << ',' << csv_cell(r.model_file_type) << ',' << p.n_gpu_layers << ',' << p.n_ctx
            << ',' << csv_cell(p.type_k) << ',' << csv_cell(p.type_v) << ',' << p.flash_attn << ',' << p.n_ubatch << ','
            << p.n_batch << ',' << p.n_threads << ',' << c.n_prompt << ',' << c.n_gen << ',' << c.repetitions << ','
            << c.warmup << ',' << r.samples.size() << ',' << r.pp.mean << ',' << r.pp.stddev << ',' << r.tg.mean << ','
            << r.tg.stddev << ',' << r.ttft.mean << ',' << r.ttft.stddev << ',' << r.load_ms << ',' << r.vram_peak
            << ',' << csv_cell(r.gpu_name) << ',' << csv_cell(r.llama_version) << ',' << csv_cell(r.timestamp) << '\n';
    }
    return out.str();
}

std::string to_markdown(const std::vector<BenchResult>& results) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(2);
    out << "| Modèle | Configuration | n | pp (tokens/s) | tg (tokens/s) | TTFT (ms) | Chargement (ms) | VRAM max (Gio) "
           "|\n"
           "|---|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : results)
        out << "| " << markdown_cell(r.model_name) << " | " << markdown_cell(config_label(r.config)) << " | "
            << r.samples.size() << " | " << r.pp.mean << " ± " << r.pp.stddev << " | " << r.tg.mean << " ± "
            << r.tg.stddev << " | " << r.ttft.mean << " ± " << r.ttft.stddev << " | " << r.load_ms << " | "
            << static_cast<double>(r.vram_peak) / GiB << " |\n";
    return out.str();
}
} // namespace banc
