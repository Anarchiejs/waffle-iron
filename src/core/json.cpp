#include "core/json.hpp"

#include "core/types.hpp"
#include "core/util.hpp"

#include <cctype>
#include <cmath>
#include <sstream>

namespace waffle {
namespace {

struct Parser {
    std::string_view in;
    size_t i = 0;

    [[noreturn]] void fail(const char* message) const { throw Error(std::string("JSON: ") + message); }

    void skip() {
        while (i < in.size() && std::isspace(static_cast<unsigned char>(in[i]))) ++i;
    }

    char peek() {
        skip();
        if (i >= in.size()) fail("unexpected end");
        return in[i];
    }

    char take() {
        const char c = peek();
        ++i;
        return c;
    }

    void expect(char c) {
        if (take() != c) fail("unexpected character");
    }

    Json parse_value() {
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Json::string_value(parse_string());
        if (c == 't' || c == 'f') return Json::boolean_value(parse_bool());
        if (c == 'n') {
            parse_null();
            return Json::nul();
        }
        return Json::number_value(parse_number());
    }

    Json parse_object() {
        expect('{');
        std::map<std::string, Json> fields;
        if (peek() == '}') {
            ++i;
            return Json::object_value(std::move(fields));
        }
        while (true) {
            if (peek() != '"') fail("object key must be a string");
            auto key = parse_string();
            expect(':');
            fields.emplace(std::move(key), parse_value());
            const char sep = take();
            if (sep == '}') break;
            if (sep != ',') fail("expected comma in object");
        }
        return Json::object_value(std::move(fields));
    }

    Json parse_array() {
        expect('[');
        std::vector<Json> items;
        if (peek() == ']') {
            ++i;
            return Json::array_value(std::move(items));
        }
        while (true) {
            items.push_back(parse_value());
            const char sep = take();
            if (sep == ']') break;
            if (sep != ',') fail("expected comma in array");
        }
        return Json::array_value(std::move(items));
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (i < in.size()) {
            const char c = in[i++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i >= in.size()) fail("bad escape");
                const char e = in[i++];
                switch (e) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(e);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        if (i + 4 > in.size()) fail("bad unicode escape");
                        int code = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = in[i++];
                            code <<= 4;
                            if (h >= '0' && h <= '9')
                                code += h - '0';
                            else if (h >= 'a' && h <= 'f')
                                code += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F')
                                code += h - 'A' + 10;
                            else
                                fail("bad unicode escape");
                        }
                        if (code < 0x80) {
                            out.push_back(static_cast<char>(code));
                        } else if (code < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        fail("bad escape");
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                fail("raw control character in string");
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    bool parse_bool() {
        if (in.substr(i, 4) == "true") {
            i += 4;
            return true;
        }
        if (in.substr(i, 5) == "false") {
            i += 5;
            return false;
        }
        fail("expected boolean");
    }

    void parse_null() {
        if (in.substr(i, 4) != "null") fail("expected null");
        i += 4;
    }

    double parse_number() {
        skip();
        const size_t start = i;
        if (i < in.size() && (in[i] == '-' || in[i] == '+')) ++i;
        if (i >= in.size() || !std::isdigit(static_cast<unsigned char>(in[i]))) fail("expected number");
        while (i < in.size() && std::isdigit(static_cast<unsigned char>(in[i]))) ++i;
        if (i < in.size() && in[i] == '.') {
            ++i;
            while (i < in.size() && std::isdigit(static_cast<unsigned char>(in[i]))) ++i;
        }
        if (i < in.size() && (in[i] == 'e' || in[i] == 'E')) {
            ++i;
            if (i < in.size() && (in[i] == '-' || in[i] == '+')) ++i;
            while (i < in.size() && std::isdigit(static_cast<unsigned char>(in[i]))) ++i;
        }
        try {
            return std::stod(std::string(in.substr(start, i - start)));
        } catch (const std::exception&) {
            fail("bad number");
        }
    }
};

void dump_into(const Json& value, std::string& out, int indent, int level) {
    switch (value.kind) {
        case Json::Kind::Null:
            out += "null";
            break;
        case Json::Kind::Bool:
            out += value.boolean ? "true" : "false";
            break;
        case Json::Kind::Number: {
            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            stream.precision(15);
            stream << value.number;
            out += stream.str();
            break;
        }
        case Json::Kind::String:
            out.push_back('"');
            out += json_escape(value.text);
            out.push_back('"');
            break;
        case Json::Kind::Array: {
            out.push_back('[');
            if (!value.items.empty() && indent > 0) out.push_back('\n');
            for (size_t n = 0; n < value.items.size(); ++n) {
                if (indent > 0) out.append(static_cast<size_t>(indent) * static_cast<size_t>(level + 1), ' ');
                dump_into(value.items[n], out, indent, level + 1);
                if (n + 1 < value.items.size()) out.push_back(',');
                if (indent > 0) out.push_back('\n');
            }
            if (indent > 0 && !value.items.empty())
                out.append(static_cast<size_t>(indent) * static_cast<size_t>(level), ' ');
            out.push_back(']');
            break;
        }
        case Json::Kind::Object: {
            out.push_back('{');
            if (!value.fields.empty() && indent > 0) out.push_back('\n');
            size_t n = 0;
            for (const auto& [key, child] : value.fields) {
                if (indent > 0) out.append(static_cast<size_t>(indent) * static_cast<size_t>(level + 1), ' ');
                out.push_back('"');
                out += json_escape(key);
                out += indent > 0 ? "\": " : "\":";
                dump_into(child, out, indent, level + 1);
                if (++n < value.fields.size()) out.push_back(',');
                if (indent > 0) out.push_back('\n');
            }
            if (indent > 0 && !value.fields.empty())
                out.append(static_cast<size_t>(indent) * static_cast<size_t>(level), ' ');
            out.push_back('}');
            break;
        }
    }
}

} // namespace

Json Json::nul() {
    return {};
}
Json Json::boolean_value(bool value) {
    Json j;
    j.kind = Kind::Bool;
    j.boolean = value;
    return j;
}
Json Json::number_value(double value) {
    Json j;
    j.kind = Kind::Number;
    j.number = value;
    return j;
}
Json Json::string_value(std::string value) {
    Json j;
    j.kind = Kind::String;
    j.text = std::move(value);
    return j;
}
Json Json::array_value(std::vector<Json> value) {
    Json j;
    j.kind = Kind::Array;
    j.items = std::move(value);
    return j;
}
Json Json::object_value(std::map<std::string, Json> value) {
    Json j;
    j.kind = Kind::Object;
    j.fields = std::move(value);
    return j;
}

const Json* Json::find(std::string_view key) const {
    if (kind != Kind::Object) return nullptr;
    auto it = fields.find(std::string(key));
    return it == fields.end() ? nullptr : &it->second;
}

bool Json::as_bool(bool fallback) const {
    return kind == Kind::Bool ? boolean : fallback;
}
double Json::as_number(double fallback) const {
    return kind == Kind::Number ? number : fallback;
}
int64_t Json::as_int(int64_t fallback) const {
    if (kind != Kind::Number || !std::isfinite(number)) return fallback;
    return static_cast<int64_t>(number);
}
std::string Json::as_string(const std::string& fallback) const {
    return kind == Kind::String ? text : fallback;
}

std::string Json::dump(int indent) const {
    std::string out;
    dump_into(*this, out, indent, 0);
    if (indent > 0) out.push_back('\n');
    return out;
}

Json parse_json(std::string_view text) {
    Parser parser{text, 0};
    auto value = parser.parse_value();
    parser.skip();
    if (parser.i != text.size()) throw Error("JSON: trailing data");
    return value;
}

} // namespace waffle
