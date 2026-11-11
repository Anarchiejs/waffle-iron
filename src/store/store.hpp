// SQLite storage: model library (tags, notes), benchmark campaigns, VRAM checks.
// Thread-safe (an internal mutex); schema versioned by PRAGMA user_version with migrations.
#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace waffle {

struct ModelRecord {
    std::string path; // key
    std::string file_name;
    std::string name;
    std::string architecture;
    std::string file_type;
    uint64_t file_size = 0;
    uint64_t n_params = 0;
    uint32_t context_length = 0;
    uint32_t n_layer = 0;
    std::vector<std::string> tags;
    std::string notes;
    std::string last_seen; // ISO 8601
};

struct CampaignRecord {
    int64_t id = 0;
    std::string name;
    std::string created_at; // ISO 8601
    int n_results = 0;
    std::string model_names; // distinct names separated by ", "
};

// One history point: a given configuration of a model measured on a date.
struct HistoryPoint {
    std::string timestamp;
    std::string llama_version;
    std::string config; // config_label()
    double pp_mean = 0, tg_mean = 0, ttft_mean = 0;
    double pp_stddev = 0, tg_stddev = 0;
};

struct VramCheck {
    std::string model_path;
    std::string timestamp;
    std::string config; // "ngl=… ctx=… kv=… fa=…"
    uint64_t estimated_gpu = 0;
    uint64_t measured_gpu = 0;
};

class Store {
public:
    // Opens or creates the database; throws Error on failure.
    explicit Store(const std::filesystem::path& db_path);
    ~Store();
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    // %LOCALAPPDATA%\WaffleIron\waffle.db (directory created if needed).
    static std::filesystem::path default_path();

    // --- Library
    void upsert_model(const ModelInfo& info); // keeps existing tags and notes
    std::vector<ModelRecord> list_models() const;
    std::optional<ModelRecord> find_model(const std::string& path) const;
    void set_model_tags(const std::string& path, const std::vector<std::string>& tags);
    void set_model_notes(const std::string& path, const std::string& notes);
    void remove_missing_models(); // deletes records whose file no longer exists

    // --- Benchmark
    int64_t save_campaign(const std::string& name, const std::vector<BenchResult>& results);
    std::vector<CampaignRecord> list_campaigns() const;
    std::vector<BenchResult> load_campaign(int64_t id) const; // including samples
    void delete_campaign(int64_t id);
    std::vector<HistoryPoint> history(const std::string& model_path) const; // sorted by ascending date

    // --- VRAM planner
    void save_vram_check(const VramCheck& check);
    std::vector<VramCheck> vram_checks(const std::string& model_path) const;

private:
    mutable std::mutex mutex_;
    sqlite3* db_ = nullptr;
};

} // namespace waffle
