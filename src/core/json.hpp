// Small JSON value used for campaign export and import. No third-party parser.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace waffle {

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    static Json nul();
    static Json boolean_value(bool value);
    static Json number_value(double value);
    static Json string_value(std::string value);
    static Json array_value(std::vector<Json> value);
    static Json object_value(std::map<std::string, Json> value);

    const Json* find(std::string_view key) const;
    bool as_bool(bool fallback = false) const;
    double as_number(double fallback = 0) const;
    int64_t as_int(int64_t fallback = 0) const;
    std::string as_string(const std::string& fallback = {}) const;

    std::string dump(int indent = 2) const;
};

Json parse_json(std::string_view text);

} // namespace waffle
