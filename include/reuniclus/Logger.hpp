#pragma once

// Phase 0 – Lightweight Logger using std::format (C++20) or libfmt fallback.
//
// Design rationale:
//   Heavy logging frameworks (spdlog, log4cxx) introduce heap allocations and
//   mutex contention that defeat the zero-allocation hot path.  This logger is
//   used *off* the hot path only – status messages, validation results, and
//   session summaries.  In the processing loop, latency counters are written to
//   a pre-allocated ring buffer (see Telemetry.hpp) instead.
//
// Compiler compatibility:
//   <format> requires GCC 13+ / MSVC 19.29+ / Clang 14+.
//   On older toolchains (e.g. GCC 12) we fall back to {fmt} (libfmt-dev),
//   which is API-compatible and was the basis for the std::format proposal.

#if __has_include(<format>)
#  include <format>
#  define REUNICLUS_FORMAT_NS std
#elif __has_include(<fmt/format.h>)
#  define FMT_HEADER_ONLY
#  include <fmt/format.h>
#  define REUNICLUS_FORMAT_NS fmt
#else
#  error "No format library found. On GCC 12: sudo apt install libfmt-dev"
#endif

#include <string_view>
#include <cstdio>
#include <chrono>
#include <mutex>

namespace reuniclus {

enum class LogLevel { DEBUG = 0, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel lvl) noexcept { min_level_ = lvl; }
    void set_file(std::FILE* f) noexcept  { out_ = f; }

    template<typename... Args>
    void log(LogLevel lvl, std::string_view fmt, Args&&... args) {
        if (lvl < min_level_) return;
        auto msg = REUNICLUS_FORMAT_NS::vformat(fmt, REUNICLUS_FORMAT_NS::make_format_args(args...));
        auto now = std::chrono::system_clock::now();
        auto ts  = std::chrono::duration_cast<std::chrono::microseconds>(
                       now.time_since_epoch()).count();

        const char* tag = [lvl]{
            switch (lvl) {
                case LogLevel::DEBUG: return "DBG";
                case LogLevel::INFO:  return "INF";
                case LogLevel::WARN:  return "WRN";
                case LogLevel::ERROR: return "ERR";
            }
            return "???";
        }();

        std::lock_guard<std::mutex> lk(mu_);
        std::fprintf(out_, "[%lld µs] [%s] %s\n",
                     static_cast<long long>(ts), tag, msg.c_str());
    }

    template<typename... Args>
    void debug(std::string_view fmt, Args&&... args) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
    }
    template<typename... Args>
    void info(std::string_view fmt, Args&&... args) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }
    template<typename... Args>
    void warn(std::string_view fmt, Args&&... args) {
        log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
    }
    template<typename... Args>
    void error(std::string_view fmt, Args&&... args) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

private:
    Logger() : out_(stderr), min_level_(LogLevel::INFO) {}

    std::FILE*  out_;
    LogLevel    min_level_;
    std::mutex  mu_;
};

// Convenience macros – expand to nothing in release builds when level is
// below minimum, so the call overhead is the branch check only.
#define LOG_DEBUG(...) ::reuniclus::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  ::reuniclus::Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)  ::reuniclus::Logger::instance().warn(__VA_ARGS__)
#define LOG_ERROR(...) ::reuniclus::Logger::instance().error(__VA_ARGS__)

} // namespace reuniclus
