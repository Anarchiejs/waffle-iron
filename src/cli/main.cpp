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
    const char* units[] = {"o", "Kio", "Mio", "Gio", "Tio"};
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
    if (parts.empty()) throw banc::Error("Liste vide pour " + flag);
    return parts;
}

long long parse_integer(const std::string& text, const std::string& flag) {
    if (text.empty()) throw banc::Error("Nombre entier attendu pour " + flag);
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        throw banc::Error("Nombre entier attendu pour " + flag + " : " + text);
    }
    if (used != text.size()) throw banc::Error("Nombre entier attendu pour " + flag + " : " + text);
    return value;
}

double parse_double(const std::string& text, const std::string& flag) {
    if (text.empty()) throw banc::Error("Nombre attendu pour " + flag);
    size_t used = 0;
    double value = 0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw banc::Error("Nombre attendu pour " + flag + " : " + text);
    }
    if (used != text.size()) throw banc::Error("Nombre attendu pour " + flag + " : " + text);
    return value;
}

uint32_t parse_u32_at_least(const std::string& text, const std::string& flag, uint32_t minimum) {
    const long long value = parse_integer(text, flag);
    if (value < static_cast<long long>(minimum) || value > static_cast<long long>(UINT32_MAX))
        throw banc::Error(flag + " hors plage : " + text);
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
    // Rempli par main sur les autres plateformes.
    return {};
#endif
}

const char* kGeneralHelp = R"HELP(banc — banc d'essai local de modèles GGUF

Usage : banc [--db CHEMIN] COMMANDE [options]

Commandes :
  scan [dossiers…]     Enregistre les GGUF trouvés (dossiers par défaut sinon)
  list                 Bibliothèque : nom, quantification, taille, paramètres, contexte, chemin
  info MODÈLE          Métadonnées principales et octets par type
  plan MODÈLE          Estimation de VRAM et couches GPU tenables
  bench MODÈLE         Mesure de vitesse, grille de réglages, sauvegarde
  history MODÈLE       Historique des mesures de ce modèle
  campaigns            Campagnes enregistrées
  export ID            Exporte une campagne (--md par défaut, ou --csv)
  gpu                  Un échantillon de télémétrie GPU

  --db CHEMIN          Base SQLite (défaut : %LOCALAPPDATA%\BancLocal\banc.db)
  --help               Cette aide

MODÈLE est un chemin de fichier ou une sous-chaîne unique du nom en base.
banc COMMANDE --help   Aide de la commande.
)HELP";

const char* kScanHelp = "banc scan [dossiers…]\n\nParcourt les dossiers (ou les dossiers de modèles par défaut) et "
                        "enregistre chaque GGUF.\n";
const char* kListHelp = "banc list\n\nAffiche la bibliothèque enregistrée dans la base.\n";
const char* kInfoHelp =
    "banc info MODÈLE\n\nLit le fichier et affiche les métadonnées principales, puis les octets par type.\n";
const char* kPlanHelp = R"HELP(banc plan MODÈLE [options]

  --ngl N            Couches sur le GPU (-1 = toutes)
  --ctx N            Taille de contexte
  --ctk TYPE         Type du cache K (f16, q8_0, …)
  --ctv TYPE         Type du cache V
  --fa 0|1           Flash attention
  --swa-full 0|1     Cache SWA de la taille du contexte (1, défaut de llama.cpp) ou de la fenêtre seule (0)
  --budget-gib X     Budget VRAM en Gio (défaut : VRAM libre selon la télémétrie)
  --verify           Charge réellement le modèle sur le GPU et mesure l'écart avec l'estimation
)HELP";
const char* kBenchHelp = R"HELP(banc bench MODÈLE [options]

Chaque réglage accepte une liste séparée par des virgules (produit cartésien).

  --ngl LISTE        Couches GPU, ex. 0,20,99
  --ctx LISTE        Contextes
  --ctk LISTE        Types de cache K et V
  --fa LISTE         Flash attention, 0 ou 1
  --pp LISTE         Tokens de consigne
  --tg LISTE         Tokens générés
  --reps LISTE       Répétitions mesurées
  --name NOM         Nom de la campagne enregistrée
)HELP";
const char* kHistoryHelp = "banc history MODÈLE\n\nUn point par résultat dont le chemin de chargement est ce modèle, "
                           "du plus ancien au plus récent.\n";
const char* kCampaignsHelp = "banc campaigns\n\nListe les campagnes enregistrées.\n";
const char* kExportHelp = R"HELP(banc export ID [options]

  --md               Markdown (défaut)
  --csv              CSV
  -o FICHIER         Écrit le fichier au lieu de la sortie standard
)HELP";
const char* kGpuHelp = "banc gpu\n\nAffiche un échantillon NVML, ou la raison de l'indisponibilité.\n";

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
            if (i + 1 >= args.size()) throw banc::Error("Valeur manquante pour --db");
            invocation.db = path_from_utf8(args[++i]);
        } else if (arg.rfind("--db=", 0) == 0) {
            if (arg.size() == 5) throw banc::Error("Valeur manquante pour --db");
            invocation.db = path_from_utf8(arg.substr(5));
        } else {
            tokens.push_back(arg);
        }
    }
    if (tokens.empty() || is_help(tokens[0])) {
        invocation.general_help = true;
        if (tokens.empty()) throw banc::Error("Commande manquante. Essayez : banc --help");
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
                if (i + 1 >= args_.size()) throw banc::Error("Valeur manquante pour " + name);
                value = args_[i + 1];
                erase_count = 2;
            } else if (arg.rfind(name + "=", 0) == 0) {
                value = arg.substr(name.size() + 1);
                if (value->empty()) throw banc::Error("Valeur manquante pour " + name);
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
            if (!arg.empty() && arg[0] == '-') throw banc::Error("Option inconnue : " + arg);
        }
        return args_;
    }

private:
    std::vector<std::string> args_;
};

struct ResolvedModel {
    std::filesystem::path path;
    std::string key;
    std::optional<banc::ModelRecord> record;
};

ResolvedModel resolve_model(banc::Store& store, const std::string& spec) {
    const auto as_path = path_from_utf8(spec);
    std::error_code ec;
    if (std::filesystem::is_regular_file(as_path, ec)) {
        const auto path = normalize_path(as_path);
        const std::string key = utf8_from_path(path);
        return ResolvedModel{path, key, store.find_model(key)};
    }

    std::vector<banc::ModelRecord> hits;
    for (const auto& model : store.list_models()) {
        if (model.name.find(spec) != std::string::npos || model.file_name.find(spec) != std::string::npos ||
            model.path.find(spec) != std::string::npos)
            hits.push_back(model);
    }
    if (hits.empty()) throw banc::Error("Modèle introuvable : " + spec);
    if (hits.size() > 1) {
        std::string message = "Plusieurs modèles correspondent à « " + spec + " » :";
        const size_t shown = std::min<size_t>(hits.size(), 8);
        for (size_t i = 0; i < shown; ++i) message += "\n  " + hits[i].name + "  " + hits[i].path;
        if (hits.size() > shown) message += "\n  …";
        throw banc::Error(message);
    }
    return ResolvedModel{path_from_utf8(hits[0].path), hits[0].path, hits[0]};
}

int cmd_scan(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    std::vector<std::filesystem::path> roots;
    if (positionals.empty()) {
        roots = banc::default_model_roots();
    } else {
        for (const auto& dir : positionals) roots.push_back(path_from_utf8(dir));
    }
    const auto files = banc::scan_model_dirs(roots);
    int recorded = 0;
    for (const auto& file : files) {
        try {
            const auto path = normalize_path(file);
            auto info = banc::read_gguf(path, false);
            if (!banc::is_language_model(info)) continue;
            info.path = path;
            store.upsert_model(info);
            ++recorded;
        } catch (const banc::Error& error) {
            std::cerr << error.what() << '\n';
        }
    }
    if (recorded <= 1)
        std::cout << recorded << " modèle enregistré\n";
    else
        std::cout << recorded << " modèles enregistrés\n";
    return 0;
}

int cmd_list(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    if (!parser.positionals().empty()) throw banc::Error("banc list n'accepte pas d'argument");
    std::vector<std::vector<std::string>> rows;
    for (const auto& model : store.list_models()) {
        // Fiche enregistrée avant qu'on reconnaisse les modèles draft d'Ollama : on la masque.
        if (const auto blob = banc::find_ollama_blob(std::filesystem::u8path(model.path)); blob && !blob->model)
            continue;
        rows.push_back({model.name, model.file_type, format_bytes(model.file_size), format_count(model.n_params),
                        std::to_string(model.context_length), model.path});
    }
    if (rows.empty()) {
        std::cout << "Aucun modèle enregistré\n";
        return 0;
    }
    print_table({"Nom", "Quantif", "Taille", "Paramètres", "Contexte", "Chemin"}, rows);
    return 0;
}

int cmd_info(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw banc::Error("Usage : banc info MODÈLE");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw banc::Error("Fichier de modèle introuvable : " + utf8_from_path(model.path));
    const banc::ModelInfo info = banc::read_gguf(model.path, true);
    std::cout << "Nom           " << info.name << '\n'
              << "Fichier       " << info.file_name << '\n'
              << "Chemin        " << utf8_from_path(info.path) << '\n'
              << "Architecture  " << info.architecture << '\n'
              << "Quantif       " << info.file_type << '\n'
              << "Taille        " << format_bytes(info.file_size) << '\n'
              << "Paramètres    " << format_count(info.n_params) << '\n'
              << "Contexte      " << info.context_length << '\n'
              << "Couches       " << info.n_layer << '\n'
              << "Embedding     " << info.n_embd << '\n'
              << "Têtes         " << info.n_head << " (KV " << info.n_head_kv << ")\n"
              << "Dim tête K/V  " << info.head_dim_k << " / " << info.head_dim_v << '\n';
    if (info.sliding_window) std::cout << "Fenêtre       " << info.sliding_window << '\n';
    if (info.n_expert) std::cout << "Experts       " << info.n_expert << " (actifs " << info.n_expert_used << ")\n";
    if (model.record && !model.record->notes.empty()) std::cout << "Notes         " << model.record->notes << '\n';
    if (model.record && !model.record->tags.empty()) {
        std::cout << "Étiquettes    ";
        for (size_t i = 0; i < model.record->tags.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << model.record->tags[i];
        }
        std::cout << '\n';
    }
    std::cout << "Octets par type\n";
    if (info.bytes_by_type.empty()) {
        std::cout << "  (aucun)\n";
    } else {
        for (const auto& [type, bytes] : info.bytes_by_type)
            std::cout << "  " << type << "  " << format_bytes(bytes) << '\n';
    }
    return 0;
}

void print_estimate(const banc::VramEstimate& estimate) {
    std::cout << "Poids GPU       " << format_bytes(estimate.weights_gpu) << '\n'
              << "Poids CPU       " << format_bytes(estimate.weights_cpu) << '\n'
              << "Cache KV GPU    " << format_bytes(estimate.kv_cache_gpu) << '\n'
              << "Cache KV CPU    " << format_bytes(estimate.kv_cache_cpu) << '\n'
              << "Calcul GPU      " << format_bytes(estimate.compute_gpu) << '\n'
              << "Calcul CPU      " << format_bytes(estimate.compute_cpu) << '\n'
              << "Réserve GPU     " << format_bytes(estimate.overhead_gpu) << '\n'
              << "Total GPU       " << format_bytes(estimate.total_gpu()) << '\n'
              << "Total CPU       " << format_bytes(estimate.total_cpu()) << '\n';
    for (const auto& note : estimate.notes) std::cout << "Note            " << note << '\n';
}

// Charge le modèle avec les réglages du plan et compare la VRAM réellement prise à l'estimation.
void verify_plan(banc::Store& store, const std::filesystem::path& path, const banc::VramPlanInput& input,
                 const banc::VramEstimate& estimate) {
    banc::Telemetry telemetry;
    const auto before = telemetry.sample();
    if (!before) throw banc::Error("Vérification impossible : " + telemetry.error());

    banc::LoadParams params;
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

    banc::Engine engine;
    // Garde les lignes où llama.cpp annonce ses allocations : utiles pour comprendre un écart.
    std::vector<std::string> allocations;
    std::mutex allocations_mutex;
    engine.set_log_sink([&](int, std::string_view text) {
        if (text.find("buffer size") == std::string_view::npos && text.find("cells") == std::string_view::npos)
            return;
        std::string line(text);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        std::lock_guard lock(allocations_mutex);
        allocations.push_back(std::move(line));
    });
    uint64_t measured = 0;
    engine
        .submit([&](banc::EngineSession& session) {
            session.load(params);
            // Un premier décodage alloue les tampons de calcul définitifs.
            const auto tokens = session.tokenize("Bonjour", true);
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
    std::cout << "Mesuré GPU      " << format_bytes(measured) << '\n'
              << "Écart           " << gap_text.str() << " (mesure - estimation)\n";
    if (!allocations.empty()) {
        std::cout << "Allocations annoncées par llama.cpp :\n";
        for (const auto& line : allocations) std::cout << "  " << line << '\n';
    }

    banc::VramCheck check;
    check.model_path = utf8_from_path(path);
    check.timestamp = now_iso();
    check.config = "ngl=" + std::to_string(input.n_gpu_layers) + " ctx=" + std::to_string(input.n_ctx) +
                   " kv=" + input.type_k + "/" + input.type_v + " fa=" + (input.flash_attn ? "on" : "off") +
                   (input.swa_full ? "" : " swa=fenêtre");
    check.estimated_gpu = estimated;
    check.measured_gpu = measured;
    store.save_vram_check(check);
}

int cmd_plan(banc::Store& store, const std::vector<std::string>& args) {
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
    if (positionals.size() != 1) throw banc::Error("Usage : banc plan MODÈLE [options]");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw banc::Error("Fichier de modèle introuvable : " + utf8_from_path(model.path));

    banc::VramPlanInput input;
    if (ngl) input.n_gpu_layers = static_cast<int>(parse_integer(*ngl, "--ngl"));
    if (ctx) input.n_ctx = parse_u32_at_least(*ctx, "--ctx", 1);
    if (ctk) input.type_k = *ctk;
    if (ctv) input.type_v = *ctv;
    if (fa) {
        const long long value = parse_integer(*fa, "--fa");
        if (value != 0 && value != 1) throw banc::Error("--fa attend 0 ou 1");
        input.flash_attn = value == 1;
    }
    if (swa_full) {
        const long long value = parse_integer(*swa_full, "--swa-full");
        if (value != 0 && value != 1) throw banc::Error("--swa-full attend 0 ou 1");
        input.swa_full = value == 1;
    }

    const banc::ModelInfo info = banc::read_gguf(model.path, true);
    const banc::VramEstimate estimate = banc::estimate_vram(info, input);
    print_estimate(estimate);

    std::optional<uint64_t> budget;
    if (budget_text) {
        const double gib = parse_double(*budget_text, "--budget-gib");
        if (!(gib > 0)) throw banc::Error("--budget-gib doit être positif");
        budget = static_cast<uint64_t>(gib * static_cast<double>(banc::GiB));
        std::cout << "Budget          " << format_bytes(*budget) << '\n';
    } else {
        banc::Telemetry telemetry;
        if (const auto sample = telemetry.sample()) {
            // Ce que les autres programmes occupent déjà n'est pas disponible pour le modèle.
            budget = sample->vram_total - std::min(sample->vram_used, sample->vram_total);
            std::cout << "Budget          " << format_bytes(*budget) << " (VRAM libre)\n";
        } else {
            std::cout << "Budget          indisponible : " << telemetry.error() << '\n';
        }
    }
    if (budget) {
        const int layers = banc::max_gpu_layers_for(info, input, *budget);
        std::cout << "Couches GPU max " << layers << '\n';
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

int cmd_bench(banc::Store& store, const std::vector<std::string>& args) {
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
    if (positionals.size() != 1) throw banc::Error("Usage : banc bench MODÈLE [options]");
    const auto model = resolve_model(store, positionals[0]);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model.path, ec))
        throw banc::Error("Fichier de modèle introuvable : " + utf8_from_path(model.path));

    banc::CampaignGrid grid;
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
            if (value != 0 && value != 1) throw banc::Error("--fa attend 0 ou 1");
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
            if (value < 1) throw banc::Error("--reps doit être au moins 1");
            repetitions.push_back(static_cast<int>(value));
        }
    }

    banc::Engine engine;
    banc::Telemetry telemetry;
    banc::Telemetry* telemetry_ptr = telemetry.available() ? &telemetry : nullptr;
    std::vector<banc::BenchResult> results;
    std::vector<std::string> errors;
    bool printed = false;
    for (uint32_t n_prompt : prompts) {
        for (uint32_t n_gen : generated) {
            for (int reps : repetitions) {
                banc::CampaignGrid one = grid;
                one.base.n_prompt = n_prompt;
                one.base.n_gen = n_gen;
                one.base.repetitions = reps;
                auto batch = banc::run_campaign(engine, telemetry_ptr, one, show_progress, &errors);
                printed = true;
                results.insert(results.end(), std::make_move_iterator(batch.begin()),
                               std::make_move_iterator(batch.end()));
            }
        }
    }
    if (printed) std::cerr << '\n';
    for (const auto& error : errors) std::cerr << error << '\n';
    if (results.empty()) throw banc::Error("Aucune mesure enregistrée");

    std::string markdown = banc::to_markdown(results);
    if (markdown.empty() || markdown.back() != '\n') markdown.push_back('\n');
    std::cout << markdown;

    const std::string name = name_text && !name_text->empty() ? *name_text : "banc " + now_iso();
    const int64_t id = store.save_campaign(name, results);
    std::cerr << "Campagne " << id << " enregistrée (" << name << ")\n";
    return errors.empty() ? 0 : 1;
}

int cmd_history(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw banc::Error("Usage : banc history MODÈLE");
    const auto model = resolve_model(store, positionals[0]);
    const auto points = store.history(model.key);
    if (points.empty()) {
        std::cout << "Aucune mesure pour ce modèle\n";
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

int cmd_campaigns(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    if (!parser.positionals().empty()) throw banc::Error("banc campaigns n'accepte pas d'argument");
    const auto campaigns = store.list_campaigns();
    if (campaigns.empty()) {
        std::cout << "Aucune campagne\n";
        return 0;
    }
    std::vector<std::vector<std::string>> rows;
    for (const auto& campaign : campaigns) {
        rows.push_back({std::to_string(campaign.id), campaign.created_at, std::to_string(campaign.n_results),
                        campaign.model_names, campaign.name});
    }
    print_table({"Id", "Date", "Résultats", "Modèles", "Nom"}, rows);
    return 0;
}

int cmd_export(banc::Store& store, const std::vector<std::string>& args) {
    FlagParser parser(args);
    const bool csv = parser.take_flag("--csv");
    const bool markdown = parser.take_flag("--md");
    if (csv && markdown) throw banc::Error("Choisir --csv ou --md");
    const auto out = parser.take_value("-o");
    const auto positionals = parser.positionals();
    if (positionals.size() != 1) throw banc::Error("Usage : banc export ID [--csv|--md] [-o FICHIER]");
    const long long id_value = parse_integer(positionals[0], "ID");
    if (id_value < 1) throw banc::Error("Identifiant de campagne invalide");
    const auto results = store.load_campaign(id_value);
    std::string text = csv ? banc::to_csv(results) : banc::to_markdown(results);
    if (!text.empty() && text.back() != '\n') text.push_back('\n');
    if (!out) {
        std::cout << text;
        return 0;
    }
    const auto path = path_from_utf8(*out);
    if (!path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) throw banc::Error("Impossible de créer le dossier : " + utf8_from_path(path.parent_path()));
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) throw banc::Error("Impossible d'écrire : " + utf8_from_path(path));
    file << text;
    if (!file) throw banc::Error("Impossible d'écrire : " + utf8_from_path(path));
    return 0;
}

int cmd_gpu() {
    banc::Telemetry telemetry;
    if (!telemetry.available()) {
        std::cout << "Télémétrie indisponible : " << telemetry.error() << '\n';
        return 0;
    }
    const auto sample = telemetry.sample();
    if (!sample) {
        std::cout << "Télémétrie indisponible : " << telemetry.error() << '\n';
        return 0;
    }
    std::cout << "Carte           " << telemetry.device_name() << '\n'
              << "Utilisation     " << sample->gpu_util_pct << " %\n"
              << "Mémoire         " << sample->mem_util_pct << " %\n"
              << "VRAM            " << format_bytes(sample->vram_used) << " / " << format_bytes(sample->vram_total)
              << '\n'
              << "Température     " << sample->temperature_c << " °C\n"
              << "Puissance       " << std::fixed << std::setprecision(1) << sample->power_w << " W\n"
              << "Horloge SM      " << sample->sm_clock_mhz << " MHz\n";
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
        throw banc::Error("Commande inconnue : " + command);
    }
    if (command == "gpu") {
        FlagParser parser(invocation.args);
        if (!parser.positionals().empty()) throw banc::Error("banc gpu n'accepte pas d'argument");
        return cmd_gpu();
    }
    if (!help_of()) throw banc::Error("Commande inconnue : " + command + ". Essayez : banc --help");

    banc::Store store(invocation.db ? *invocation.db : banc::Store::default_path());
    if (command == "scan") return cmd_scan(store, invocation.args);
    if (command == "list") return cmd_list(store, invocation.args);
    if (command == "info") return cmd_info(store, invocation.args);
    if (command == "plan") return cmd_plan(store, invocation.args);
    if (command == "bench") return cmd_bench(store, invocation.args);
    if (command == "history") return cmd_history(store, invocation.args);
    if (command == "campaigns") return cmd_campaigns(store, invocation.args);
    if (command == "export") return cmd_export(store, invocation.args);
    throw banc::Error("Commande inconnue : " + command);
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
    } catch (const banc::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Erreur : " << error.what() << '\n';
        return 1;
    }
}
