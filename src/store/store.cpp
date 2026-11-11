#include "store/store.hpp"

#include "core/bench.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string_view>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace waffle {

namespace {

// The error message shown to the user always includes sqlite3_errmsg.
[[noreturn]] void fail(sqlite3* db) {
    const char* msg = db ? sqlite3_errmsg(db) : nullptr;
    throw Error(std::string("Database: ") + (msg ? msg : "unknown error"));
}

void exec_sql(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : sqlite3_errmsg(db);
        sqlite3_free(err);
        throw Error("Database: " + msg);
    }
}

std::string utf8_from_path(const std::filesystem::path& path) {
    const auto u8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

std::filesystem::path path_from_utf8(std::string_view utf8) {
    const std::u8string u8(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size());
    return std::filesystem::path(u8);
}

std::string now_iso() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
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

std::string join_ints(const std::vector<int>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(values[i]);
    }
    return out;
}

std::vector<int> split_ints(const std::string& text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

// Prepared statement: finalized even if a step throws.
class Stmt {
public:
    Stmt(sqlite3* db, const char* sql) : db_(db) {
        if (sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) fail(db);
    }
    ~Stmt() { sqlite3_finalize(stmt_); }
    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;

    void bind_text(int index, const std::string& value) {
        if (sqlite3_bind_text(stmt_, index, value.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK) fail(db_);
    }
    void bind_int(int index, int value) {
        if (sqlite3_bind_int(stmt_, index, value) != SQLITE_OK) fail(db_);
    }
    void bind_int64(int index, int64_t value) {
        if (sqlite3_bind_int64(stmt_, index, value) != SQLITE_OK) fail(db_);
    }
    void bind_double(int index, double value) {
        if (sqlite3_bind_double(stmt_, index, value) != SQLITE_OK) fail(db_);
    }
    void reset() {
        sqlite3_reset(stmt_);
        sqlite3_clear_bindings(stmt_);
    }

    // SQLITE_ROW -> true, SQLITE_DONE -> false, otherwise Error.
    bool step() {
        const int rc = sqlite3_step(stmt_);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        fail(db_);
    }

    std::string text(int column) const {
        const auto* p = sqlite3_column_text(stmt_, column);
        return p ? reinterpret_cast<const char*>(p) : std::string();
    }
    int integer(int column) const { return sqlite3_column_int(stmt_, column); }
    int64_t integer64(int column) const { return sqlite3_column_int64(stmt_, column); }
    double real(int column) const { return sqlite3_column_double(stmt_, column); }

private:
    sqlite3* db_ = nullptr;
    sqlite3_stmt* stmt_ = nullptr;
};

struct Transaction {
    explicit Transaction(sqlite3* db) : db_(db) { exec_sql(db_, "BEGIN IMMEDIATE"); }
    ~Transaction() {
        if (open_) sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
    }
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    void commit() {
        exec_sql(db_, "COMMIT");
        open_ = false;
    }

private:
    sqlite3* db_ = nullptr;
    bool open_ = true;
};

int user_version(sqlite3* db) {
    Stmt query(db, "PRAGMA user_version");
    if (!query.step()) return 0;
    return query.integer(0);
}

// Schema v1. Versions already applied are not replayed.
void migrate(sqlite3* db) {
    if (user_version(db) >= 1) return;

    exec_sql(db, R"SQL(
CREATE TABLE IF NOT EXISTS models (
    path TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    n_params INTEGER NOT NULL,
    context_length INTEGER NOT NULL,
    n_layer INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_tags (
    path TEXT NOT NULL REFERENCES models(path) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (path, tag)
);
CREATE TABLE IF NOT EXISTS campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_file_type TEXT NOT NULL,
    load_ms REAL NOT NULL,
    vram_peak INTEGER NOT NULL,
    gpu_name TEXT NOT NULL,
    llama_version TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model_path TEXT NOT NULL,
    n_gpu_layers INTEGER NOT NULL,
    n_ctx INTEGER NOT NULL,
    n_batch INTEGER NOT NULL,
    n_ubatch INTEGER NOT NULL,
    type_k TEXT NOT NULL,
    type_v TEXT NOT NULL,
    flash_attn INTEGER NOT NULL,
    n_threads INTEGER NOT NULL,
    n_seq_max INTEGER NOT NULL,
    n_prompt INTEGER NOT NULL,
    n_gen INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    warmup INTEGER NOT NULL,
    pp_n INTEGER NOT NULL,
    pp_mean REAL NOT NULL,
    pp_stddev REAL NOT NULL,
    pp_median REAL NOT NULL,
    pp_min REAL NOT NULL,
    pp_max REAL NOT NULL,
    pp_outliers TEXT NOT NULL,
    tg_n INTEGER NOT NULL,
    tg_mean REAL NOT NULL,
    tg_stddev REAL NOT NULL,
    tg_median REAL NOT NULL,
    tg_min REAL NOT NULL,
    tg_max REAL NOT NULL,
    tg_outliers TEXT NOT NULL,
    ttft_n INTEGER NOT NULL,
    ttft_mean REAL NOT NULL,
    ttft_stddev REAL NOT NULL,
    ttft_median REAL NOT NULL,
    ttft_min REAL NOT NULL,
    ttft_max REAL NOT NULL,
    ttft_outliers TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    pp_tps REAL NOT NULL,
    tg_tps REAL NOT NULL,
    ttft_ms REAL NOT NULL,
    vram_peak INTEGER NOT NULL,
    PRIMARY KEY (result_id, idx)
);
CREATE TABLE IF NOT EXISTS vram_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config TEXT NOT NULL,
    estimated_gpu INTEGER NOT NULL,
    measured_gpu INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_campaign ON results(campaign_id);
CREATE INDEX IF NOT EXISTS idx_results_model_path ON results(model_path);
CREATE INDEX IF NOT EXISTS idx_vram_model ON vram_checks(model_path);
)SQL");
    exec_sql(db, "PRAGMA user_version = 1");
}

void bind_stats(Stmt& stmt, int& index, const Stats& stats) {
    stmt.bind_int(index++, stats.n);
    stmt.bind_double(index++, stats.mean);
    stmt.bind_double(index++, stats.stddev);
    stmt.bind_double(index++, stats.median);
    stmt.bind_double(index++, stats.min);
    stmt.bind_double(index++, stats.max);
    stmt.bind_text(index++, join_ints(stats.outliers));
}

Stats read_stats(const Stmt& stmt, int& column) {
    Stats stats;
    stats.n = stmt.integer(column++);
    stats.mean = stmt.real(column++);
    stats.stddev = stmt.real(column++);
    stats.median = stmt.real(column++);
    stats.min = stmt.real(column++);
    stats.max = stmt.real(column++);
    stats.outliers = split_ints(stmt.text(column++));
    return stats;
}

std::vector<std::string> tags_of(sqlite3* db, const std::string& path) {
    Stmt query(db, "SELECT tag FROM model_tags WHERE path = ?1 ORDER BY tag");
    query.bind_text(1, path);
    std::vector<std::string> tags;
    while (query.step()) tags.push_back(query.text(0));
    return tags;
}

ModelRecord read_model_row(sqlite3* db, Stmt& stmt) {
    ModelRecord record;
    record.path = stmt.text(0);
    record.file_name = stmt.text(1);
    record.name = stmt.text(2);
    record.architecture = stmt.text(3);
    record.file_type = stmt.text(4);
    record.file_size = static_cast<uint64_t>(stmt.integer64(5));
    record.n_params = static_cast<uint64_t>(stmt.integer64(6));
    record.context_length = static_cast<uint32_t>(stmt.integer(7));
    record.n_layer = static_cast<uint32_t>(stmt.integer(8));
    record.notes = stmt.text(9);
    record.last_seen = stmt.text(10);
    record.tags = tags_of(db, record.path);
    return record;
}

void require_model(sqlite3* db, const std::string& path) {
    Stmt query(db, "SELECT 1 FROM models WHERE path = ?1");
    query.bind_text(1, path);
    if (!query.step()) throw Error("Unknown model: " + path);
}

} // namespace

Store::Store(const std::filesystem::path& db_path) {
    const auto parent = db_path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) throw Error("Database: Cannot create directory " + parent.string());
    }

    const int rc =
        sqlite3_open_v2(utf8_from_path(db_path).c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (rc != SQLITE_OK || !db_) {
        std::string msg = db_ ? sqlite3_errmsg(db_) : "Cannot open";
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        throw Error("Database: " + msg);
    }

    try {
        exec_sql(db_, "PRAGMA journal_mode = WAL");
        exec_sql(db_, "PRAGMA foreign_keys = ON");
        exec_sql(db_, "PRAGMA busy_timeout = 5000");
        migrate(db_);
    } catch (...) {
        sqlite3_close(db_);
        db_ = nullptr;
        throw;
    }
}

Store::~Store() {
    std::lock_guard lock(mutex_);
    if (db_) sqlite3_close(db_);
}

std::filesystem::path Store::default_path() {
#ifdef _WIN32
    std::filesystem::path dir;
    const DWORD needed = GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0);
    if (needed > 0) {
        std::wstring value(needed, L'\0');
        const DWORD written = GetEnvironmentVariableW(L"LOCALAPPDATA", value.data(), needed);
        if (written > 0 && written < needed) {
            value.resize(written);
            dir = std::filesystem::path(value);
        }
    }
    if (dir.empty()) dir = std::filesystem::temp_directory_path();
    dir /= "WaffleIron";
#else
    const char* home = std::getenv("HOME");
    std::filesystem::path dir = home && *home ? std::filesystem::path(home) / ".local" / "share" / "WaffleIron"
                                              : std::filesystem::temp_directory_path() / "WaffleIron";
#endif
    std::filesystem::create_directories(dir);
    return dir / "waffle.db";
}

void Store::upsert_model(const ModelInfo& info) {
    std::lock_guard lock(mutex_);
    const std::string path = utf8_from_path(info.path);
    Stmt stmt(db_, R"SQL(
INSERT INTO models(path, file_name, name, architecture, file_type, file_size, n_params,
                   context_length, n_layer, notes, last_seen)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, '', ?10)
ON CONFLICT(path) DO UPDATE SET
    file_name = excluded.file_name,
    name = excluded.name,
    architecture = excluded.architecture,
    file_type = excluded.file_type,
    file_size = excluded.file_size,
    n_params = excluded.n_params,
    context_length = excluded.context_length,
    n_layer = excluded.n_layer,
    last_seen = excluded.last_seen
)SQL");
    stmt.bind_text(1, path);
    stmt.bind_text(2, info.file_name);
    stmt.bind_text(3, info.name);
    stmt.bind_text(4, info.architecture);
    stmt.bind_text(5, info.file_type);
    stmt.bind_int64(6, static_cast<int64_t>(info.file_size));
    stmt.bind_int64(7, static_cast<int64_t>(info.n_params));
    stmt.bind_int(8, static_cast<int>(info.context_length));
    stmt.bind_int(9, static_cast<int>(info.n_layer));
    stmt.bind_text(10, now_iso());
    stmt.step();
}

std::vector<ModelRecord> Store::list_models() const {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, R"SQL(
SELECT path, file_name, name, architecture, file_type, file_size, n_params,
       context_length, n_layer, notes, last_seen
FROM models
ORDER BY name, path
)SQL");
    std::vector<ModelRecord> records;
    while (stmt.step()) records.push_back(read_model_row(db_, stmt));
    return records;
}

std::optional<ModelRecord> Store::find_model(const std::string& path) const {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, R"SQL(
SELECT path, file_name, name, architecture, file_type, file_size, n_params,
       context_length, n_layer, notes, last_seen
FROM models WHERE path = ?1
)SQL");
    stmt.bind_text(1, path);
    if (!stmt.step()) return std::nullopt;
    return read_model_row(db_, stmt);
}

void Store::set_model_tags(const std::string& path, const std::vector<std::string>& tags) {
    std::lock_guard lock(mutex_);
    require_model(db_, path);
    Transaction tx(db_);
    Stmt clear(db_, "DELETE FROM model_tags WHERE path = ?1");
    clear.bind_text(1, path);
    clear.step();
    Stmt insert(db_, "INSERT INTO model_tags(path, tag) VALUES (?1, ?2)");
    std::vector<std::string> seen;
    for (const auto& tag : tags) {
        if (tag.empty()) continue;
        bool duplicate = false;
        for (const auto& previous : seen) {
            if (previous == tag) duplicate = true;
        }
        if (duplicate) continue;
        seen.push_back(tag);
        insert.bind_text(1, path);
        insert.bind_text(2, tag);
        insert.step();
        insert.reset();
    }
    tx.commit();
}

void Store::set_model_notes(const std::string& path, const std::string& notes) {
    std::lock_guard lock(mutex_);
    require_model(db_, path);
    Stmt stmt(db_, "UPDATE models SET notes = ?2 WHERE path = ?1");
    stmt.bind_text(1, path);
    stmt.bind_text(2, notes);
    stmt.step();
}

void Store::remove_missing_models() {
    std::lock_guard lock(mutex_);
    std::vector<std::string> stale;
    {
        Stmt stmt(db_, "SELECT path FROM models");
        while (stmt.step()) {
            const std::string path = stmt.text(0);
            std::error_code ec;
            if (!std::filesystem::is_regular_file(path_from_utf8(path), ec)) stale.push_back(path);
        }
    }
    if (stale.empty()) return;
    Transaction tx(db_);
    Stmt stmt(db_, "DELETE FROM models WHERE path = ?1");
    for (const auto& path : stale) {
        stmt.bind_text(1, path);
        stmt.step();
        stmt.reset();
    }
    tx.commit();
}

int64_t Store::save_campaign(const std::string& name, const std::vector<BenchResult>& results) {
    std::lock_guard lock(mutex_);
    Transaction tx(db_);
    Stmt campaign(db_, "INSERT INTO campaigns(name, created_at) VALUES (?1, ?2)");
    campaign.bind_text(1, name);
    campaign.bind_text(2, now_iso());
    campaign.step();
    const int64_t campaign_id = sqlite3_last_insert_rowid(db_);

    for (const auto& result : results) {
        Stmt insert(db_, R"SQL(
INSERT INTO results(
    campaign_id, model_name, model_file_type, load_ms, vram_peak, gpu_name, llama_version, timestamp,
    model_path, n_gpu_layers, n_ctx, n_batch, n_ubatch, type_k, type_v, flash_attn, n_threads, n_seq_max,
    n_prompt, n_gen, repetitions, warmup,
    pp_n, pp_mean, pp_stddev, pp_median, pp_min, pp_max, pp_outliers,
    tg_n, tg_mean, tg_stddev, tg_median, tg_min, tg_max, tg_outliers,
    ttft_n, ttft_mean, ttft_stddev, ttft_median, ttft_min, ttft_max, ttft_outliers)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
        ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
        ?19, ?20, ?21, ?22,
        ?23, ?24, ?25, ?26, ?27, ?28, ?29,
        ?30, ?31, ?32, ?33, ?34, ?35, ?36,
        ?37, ?38, ?39, ?40, ?41, ?42, ?43)
)SQL");
        const auto& load = result.config.load;
        int index = 1;
        insert.bind_int64(index++, campaign_id);
        insert.bind_text(index++, result.model_name);
        insert.bind_text(index++, result.model_file_type);
        insert.bind_double(index++, result.load_ms);
        insert.bind_int64(index++, static_cast<int64_t>(result.vram_peak));
        insert.bind_text(index++, result.gpu_name);
        insert.bind_text(index++, result.llama_version);
        insert.bind_text(index++, result.timestamp);
        insert.bind_text(index++, utf8_from_path(load.model_path));
        insert.bind_int(index++, load.n_gpu_layers);
        insert.bind_int(index++, static_cast<int>(load.n_ctx));
        insert.bind_int(index++, static_cast<int>(load.n_batch));
        insert.bind_int(index++, static_cast<int>(load.n_ubatch));
        insert.bind_text(index++, load.type_k);
        insert.bind_text(index++, load.type_v);
        insert.bind_int(index++, load.flash_attn);
        insert.bind_int(index++, load.n_threads);
        insert.bind_int(index++, static_cast<int>(load.n_seq_max));
        insert.bind_int(index++, static_cast<int>(result.config.n_prompt));
        insert.bind_int(index++, static_cast<int>(result.config.n_gen));
        insert.bind_int(index++, result.config.repetitions);
        insert.bind_int(index++, result.config.warmup);
        bind_stats(insert, index, result.pp);
        bind_stats(insert, index, result.tg);
        bind_stats(insert, index, result.ttft);
        insert.step();
        const int64_t result_id = sqlite3_last_insert_rowid(db_);

        Stmt sample(db_, R"SQL(
INSERT INTO samples(result_id, idx, pp_tps, tg_tps, ttft_ms, vram_peak)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
)SQL");
        for (size_t i = 0; i < result.samples.size(); ++i) {
            sample.bind_int64(1, result_id);
            sample.bind_int(2, static_cast<int>(i));
            sample.bind_double(3, result.samples[i].pp_tps);
            sample.bind_double(4, result.samples[i].tg_tps);
            sample.bind_double(5, result.samples[i].ttft_ms);
            sample.bind_int64(6, static_cast<int64_t>(result.samples[i].vram_peak));
            sample.step();
            sample.reset();
        }
    }
    tx.commit();
    return campaign_id;
}

std::vector<CampaignRecord> Store::list_campaigns() const {
    std::lock_guard lock(mutex_);
    Stmt campaigns(db_, "SELECT id, name, created_at FROM campaigns ORDER BY id");
    std::vector<CampaignRecord> records;
    while (campaigns.step()) {
        CampaignRecord record;
        record.id = campaigns.integer64(0);
        record.name = campaigns.text(1);
        record.created_at = campaigns.text(2);
        Stmt count(db_, "SELECT COUNT(*) FROM results WHERE campaign_id = ?1");
        count.bind_int64(1, record.id);
        if (count.step()) record.n_results = count.integer(0);
        Stmt names(db_, "SELECT DISTINCT model_name FROM results WHERE campaign_id = ?1 ORDER BY model_name");
        names.bind_int64(1, record.id);
        while (names.step()) {
            if (!record.model_names.empty()) record.model_names += ", ";
            record.model_names += names.text(0);
        }
        records.push_back(std::move(record));
    }
    return records;
}

std::vector<BenchResult> Store::load_campaign(int64_t id) const {
    std::lock_guard lock(mutex_);
    Stmt exists(db_, "SELECT 1 FROM campaigns WHERE id = ?1");
    exists.bind_int64(1, id);
    if (!exists.step()) throw Error("Campaign not found: " + std::to_string(id));

    Stmt results(db_, R"SQL(
SELECT id, model_name, model_file_type, load_ms, vram_peak, gpu_name, llama_version, timestamp,
       model_path, n_gpu_layers, n_ctx, n_batch, n_ubatch, type_k, type_v, flash_attn, n_threads, n_seq_max,
       n_prompt, n_gen, repetitions, warmup,
       pp_n, pp_mean, pp_stddev, pp_median, pp_min, pp_max, pp_outliers,
       tg_n, tg_mean, tg_stddev, tg_median, tg_min, tg_max, tg_outliers,
       ttft_n, ttft_mean, ttft_stddev, ttft_median, ttft_min, ttft_max, ttft_outliers
FROM results WHERE campaign_id = ?1 ORDER BY id
)SQL");
    results.bind_int64(1, id);
    std::vector<BenchResult> loaded;
    while (results.step()) {
        BenchResult result;
        const int64_t result_id = results.integer64(0);
        result.model_name = results.text(1);
        result.model_file_type = results.text(2);
        result.load_ms = results.real(3);
        result.vram_peak = static_cast<uint64_t>(results.integer64(4));
        result.gpu_name = results.text(5);
        result.llama_version = results.text(6);
        result.timestamp = results.text(7);
        auto& load = result.config.load;
        load.model_path = path_from_utf8(results.text(8));
        load.n_gpu_layers = results.integer(9);
        load.n_ctx = static_cast<uint32_t>(results.integer(10));
        load.n_batch = static_cast<uint32_t>(results.integer(11));
        load.n_ubatch = static_cast<uint32_t>(results.integer(12));
        load.type_k = results.text(13);
        load.type_v = results.text(14);
        load.flash_attn = results.integer(15);
        load.n_threads = results.integer(16);
        load.n_seq_max = static_cast<uint32_t>(results.integer(17));
        result.config.n_prompt = static_cast<uint32_t>(results.integer(18));
        result.config.n_gen = static_cast<uint32_t>(results.integer(19));
        result.config.repetitions = results.integer(20);
        result.config.warmup = results.integer(21);
        int column = 22;
        result.pp = read_stats(results, column);
        result.tg = read_stats(results, column);
        result.ttft = read_stats(results, column);

        Stmt samples(db_, R"SQL(
SELECT pp_tps, tg_tps, ttft_ms, vram_peak FROM samples WHERE result_id = ?1 ORDER BY idx
)SQL");
        samples.bind_int64(1, result_id);
        while (samples.step()) {
            BenchSample sample;
            sample.pp_tps = samples.real(0);
            sample.tg_tps = samples.real(1);
            sample.ttft_ms = samples.real(2);
            sample.vram_peak = static_cast<uint64_t>(samples.integer64(3));
            result.samples.push_back(sample);
        }
        loaded.push_back(std::move(result));
    }
    return loaded;
}

void Store::delete_campaign(int64_t id) {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, "DELETE FROM campaigns WHERE id = ?1");
    stmt.bind_int64(1, id);
    stmt.step();
    if (sqlite3_changes(db_) == 0) throw Error("Campaign not found: " + std::to_string(id));
}

std::vector<HistoryPoint> Store::history(const std::string& model_path) const {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, R"SQL(
SELECT timestamp, llama_version, pp_mean, tg_mean, ttft_mean, pp_stddev, tg_stddev,
       n_gpu_layers, n_ctx, type_k, type_v, flash_attn, n_ubatch
FROM results
WHERE model_path = ?1
ORDER BY timestamp, id
)SQL");
    stmt.bind_text(1, model_path);
    std::vector<HistoryPoint> points;
    while (stmt.step()) {
        HistoryPoint point;
        point.timestamp = stmt.text(0);
        point.llama_version = stmt.text(1);
        point.pp_mean = stmt.real(2);
        point.tg_mean = stmt.real(3);
        point.ttft_mean = stmt.real(4);
        point.pp_stddev = stmt.real(5);
        point.tg_stddev = stmt.real(6);
        BenchConfig cfg;
        cfg.load.n_gpu_layers = stmt.integer(7);
        cfg.load.n_ctx = static_cast<uint32_t>(stmt.integer(8));
        cfg.load.type_k = stmt.text(9);
        cfg.load.type_v = stmt.text(10);
        cfg.load.flash_attn = stmt.integer(11);
        cfg.load.n_ubatch = static_cast<uint32_t>(stmt.integer(12));
        point.config = config_label(cfg);
        points.push_back(std::move(point));
    }
    return points;
}

void Store::save_vram_check(const VramCheck& check) {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, R"SQL(
INSERT INTO vram_checks(model_path, timestamp, config, estimated_gpu, measured_gpu)
VALUES (?1, ?2, ?3, ?4, ?5)
)SQL");
    stmt.bind_text(1, check.model_path);
    stmt.bind_text(2, check.timestamp.empty() ? now_iso() : check.timestamp);
    stmt.bind_text(3, check.config);
    stmt.bind_int64(4, static_cast<int64_t>(check.estimated_gpu));
    stmt.bind_int64(5, static_cast<int64_t>(check.measured_gpu));
    stmt.step();
}

std::vector<VramCheck> Store::vram_checks(const std::string& model_path) const {
    std::lock_guard lock(mutex_);
    Stmt stmt(db_, R"SQL(
SELECT model_path, timestamp, config, estimated_gpu, measured_gpu
FROM vram_checks WHERE model_path = ?1 ORDER BY timestamp, id
)SQL");
    stmt.bind_text(1, model_path);
    std::vector<VramCheck> checks;
    while (stmt.step()) {
        VramCheck check;
        check.model_path = stmt.text(0);
        check.timestamp = stmt.text(1);
        check.config = stmt.text(2);
        check.estimated_gpu = static_cast<uint64_t>(stmt.integer64(3));
        check.measured_gpu = static_cast<uint64_t>(stmt.integer64(4));
        checks.push_back(std::move(check));
    }
    return checks;
}

} // namespace waffle
