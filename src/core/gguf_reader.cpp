#include "core/gguf_reader.hpp"

#include <gguf.h>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <type_traits>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

namespace waffle {
namespace {
template <class T> std::string number(const void* data, size_t i) {
    T value;
    std::memcpy(&value, static_cast<const char*>(data) + i * sizeof(T), sizeof(T));
    std::ostringstream out;
    out.precision(std::numeric_limits<T>::max_digits10);
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
        out << int(value);
    else
        out << value;
    return out.str();
}
std::string value_text(const gguf_context* ctx, int64_t id, bool array = false, size_t i = 0) {
    const auto type = array ? gguf_get_arr_type(ctx, id) : gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_ARRAY) return "?";
    if (type == GGUF_TYPE_STRING) return array ? gguf_get_arr_str(ctx, id, i) : gguf_get_val_str(ctx, id);
    const void* data = array ? gguf_get_arr_data(ctx, id) : gguf_get_val_data(ctx, id);
    switch (type) {
#define VALUE(kind, cpp)                                                                                               \
    case GGUF_TYPE_##kind:                                                                                             \
        return number<cpp>(data, i)
        VALUE(UINT8, uint8_t);
        VALUE(INT8, int8_t);
        VALUE(UINT16, uint16_t);
        VALUE(INT16, int16_t);
        VALUE(UINT32, uint32_t);
        VALUE(INT32, int32_t);
        VALUE(UINT64, uint64_t);
        VALUE(INT64, int64_t);
        VALUE(FLOAT32, float);
        VALUE(FLOAT64, double);
#undef VALUE
        case GGUF_TYPE_BOOL:
            return number<uint8_t>(data, i) == "0" ? "false" : "true";
        default:
            return "?";
    }
}
uint32_t integer(const gguf_context* ctx, const std::string& key, uint32_t fallback = 0) {
    const auto id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return fallback;
    const auto text = value_text(ctx, id);
    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() ? value : fallback;
}
std::string string_value(const gguf_context* ctx, const char* key, const std::string& fallback = {}) {
    const auto id = gguf_find_key(ctx, key);
    return id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING ? gguf_get_val_str(ctx, id) : fallback;
}
} // namespace

bool is_language_model(const ModelInfo& info) {
    return info.n_layer > 0 && info.architecture != "clip" && info.architecture != "mmproj";
}

bool is_gguf_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    char magic[4]{};
    return bool(file.read(magic, 4)) && std::memcmp(magic, "GGUF", 4) == 0;
}

std::string ftype_name(uint32_t type) {
    static const char* names[] = {"F32",    "F16",    "Q4_0",    "Q4_1",      "?",      "?",       "?",
                                  "Q8_0",   "Q5_0",   "Q5_1",    "Q2_K",      "Q3_K_S", "Q3_K_M",  "Q3_K_L",
                                  "Q4_K_S", "Q4_K_M", "Q5_K_S",  "Q5_K_M",    "Q6_K",   "IQ2_XXS", "IQ2_XS",
                                  "Q2_K_S", "IQ3_XS", "IQ3_XXS", "IQ1_S",     "IQ4_NL", "IQ3_S",   "IQ3_M",
                                  "IQ2_S",  "IQ2_M",  "IQ4_XS",  "IQ1_M",     "BF16",   "?",       "?",
                                  "?",      "TQ1_0",  "TQ2_0",   "MXFP4_MOE", "NVFP4",  "Q1_0",    "Q2_0"};
    return type < std::size(names) ? names[type] : "?";
}

ModelInfo read_gguf(const std::filesystem::path& path, bool with_tensors) {
    try {
        if (!is_gguf_file(path)) throw Error("Missing GGUF signature: " + path.string());
        ggml_context* raw = nullptr;
        const auto utf8 = path.u8string();
        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
            gguf_init_from_file(reinterpret_cast<const char*>(utf8.c_str()), {true, &raw}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw, ggml_free);
        if (!ctx || !tensors) throw Error("Unreadable GGUF header: " + path.string());
        ModelInfo model;
        model.path = path;
        model.file_name = path.filename().string();
        model.file_size = std::filesystem::file_size(path);
        model.architecture = string_value(ctx.get(), "general.architecture");
        model.name = string_value(ctx.get(), "general.name", path.stem().string());
        if (const auto blob = find_ollama_blob(path); blob && blob->model) model.name = blob->tags + " (Ollama)";
        model.file_type = ftype_name(integer(ctx.get(), "general.file_type", UINT32_MAX));
        const auto prefix = model.architecture + ".";
        model.context_length = integer(ctx.get(), prefix + "context_length");
        model.n_layer = integer(ctx.get(), prefix + "block_count");
        model.n_embd = integer(ctx.get(), prefix + "embedding_length");
        model.n_head = integer(ctx.get(), prefix + "attention.head_count");
        model.n_head_kv = integer(ctx.get(), prefix + "attention.head_count_kv", model.n_head);
        const auto dim = model.n_head ? model.n_embd / model.n_head : 0;
        model.head_dim_k = integer(ctx.get(), prefix + "attention.key_length", dim);
        model.head_dim_v = integer(ctx.get(), prefix + "attention.value_length", dim);
        model.sliding_window = integer(ctx.get(), prefix + "attention.sliding_window");
        model.head_dim_k_swa = integer(ctx.get(), prefix + "attention.key_length_swa", model.head_dim_k);
        model.head_dim_v_swa = integer(ctx.get(), prefix + "attention.value_length_swa", model.head_dim_v);
        model.shared_kv_layers = integer(ctx.get(), prefix + "attention.shared_kv_layers");
        if (model.sliding_window) {
            // As in llama.cpp: per-layer array, or period p (SWA layer if il % p < p - 1; 0 = all).
            const auto key = prefix + "attention.sliding_window_pattern";
            const auto id = gguf_find_key(ctx.get(), key.c_str());
            model.swa_layers.assign(model.n_layer, 1);
            if (id >= 0 && gguf_get_kv_type(ctx.get(), id) == GGUF_TYPE_ARRAY) {
                const size_t n = gguf_get_arr_n(ctx.get(), id);
                for (uint32_t il = 0; il < model.n_layer && il < n; ++il)
                    model.swa_layers[il] = value_text(ctx.get(), id, true, il) == "true" ||
                                           value_text(ctx.get(), id, true, il) == "1";
            } else if (id >= 0) {
                const uint32_t period = integer(ctx.get(), key);
                for (uint32_t il = 0; il < model.n_layer; ++il)
                    model.swa_layers[il] = period == 0 || il % period < period - 1;
            }
        }
        model.n_expert = integer(ctx.get(), prefix + "expert_count");
        model.n_expert_used = integer(ctx.get(), prefix + "expert_used_count");
        const auto chat = gguf_find_key(ctx.get(), "tokenizer.chat_template");
        if (chat >= 0 && gguf_get_kv_type(ctx.get(), chat) == GGUF_TYPE_STRING)
            model.chat_template = gguf_get_val_str(ctx.get(), chat);
        for (int64_t id = 0; id < gguf_get_n_kv(ctx.get()); ++id) {
            const std::string key = gguf_get_key(ctx.get(), id);
            std::string value;
            if (gguf_get_kv_type(ctx.get(), id) == GGUF_TYPE_ARRAY) {
                const auto n = gguf_get_arr_n(ctx.get(), id);
                value = "[" + std::to_string(n) + " × " + gguf_type_name(gguf_get_arr_type(ctx.get(), id)) + "]";
                if (key != "tokenizer.ggml.tokens" && key != "tokenizer.ggml.scores" &&
                    key != "tokenizer.ggml.token_type" && key != "tokenizer.ggml.merges") {
                    for (size_t i = 0; i < std::min<size_t>(n, 8); ++i)
                        value += (i ? ", " : " ") + value_text(ctx.get(), id, true, i);
                    if (n > 8) value += ", …";
                }
            } else
                value = value_text(ctx.get(), id);
            model.metadata.emplace_back(key, std::move(value));
        }
        for (int64_t id = 0; id < gguf_get_n_tensors(ctx.get()); ++id) {
            TensorInfo info;
            info.name = gguf_get_tensor_name(ctx.get(), id);
            const auto* tensor = ggml_get_tensor(tensors.get(), info.name.c_str());
            info.type = ggml_type_name(gguf_get_tensor_type(ctx.get(), id));
            info.n_bytes = gguf_get_tensor_size(ctx.get(), id);
            info.n_elements = static_cast<uint64_t>(ggml_nelements(tensor));
            info.shape.assign(tensor->ne, tensor->ne + ggml_n_dims(tensor));
            if (info.name.starts_with("blk.")) {
                const auto end = info.name.find('.', 4);
                if (end != std::string::npos) {
                    int layer = -1;
                    const auto parsed = std::from_chars(info.name.data() + 4, info.name.data() + end, layer);
                    if (parsed.ec == std::errc{} && parsed.ptr == info.name.data() + end && layer >= 0)
                        info.layer = layer;
                }
            }
            model.n_params += info.n_elements;
            model.bytes_by_type[info.type] += info.n_bytes;
            if (with_tensors) model.tensors.push_back(std::move(info));
        }
        return model;
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Error("Cannot read GGUF: " + path.string() + ": " + e.what());
    }
}

std::vector<std::filesystem::path> scan_model_dirs(const std::vector<std::filesystem::path>& roots) {
    std::set<std::filesystem::path> result;
    for (const auto& root : roots) {
        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(
            root, std::filesystem::directory_options::skip_permission_denied, ec),
            end;
        while (it != end) {
            const auto path = it->path();
            if (it->is_regular_file(ec)) {
                auto name = path.filename().string();
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return char(std::tolower(c)); });
                const auto split = name.find("-of-");
                const bool later_shard = split >= 6 && split != std::string::npos && name[split - 6] == '-' &&
                                         name.substr(split - 5, 5) != "00001";
                if (!name.starts_with("mmproj") && !later_shard && is_gguf_file(path)) {
                    const auto blob = find_ollama_blob(path);
                    if (!blob || blob->model) result.insert(path.lexically_normal());
                }
            }
            ec.clear();
            it.increment(ec);
        }
    }
    return {result.begin(), result.end()};
}
} // namespace waffle
