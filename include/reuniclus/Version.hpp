#pragma once

// Phase 0 – Version & Build Metadata
// constexpr strings allow compile-time version checks and are embedded
// directly in the binary so any result can be traced to an exact build.

namespace reuniclus {

inline constexpr const char* kVersion     = "0.1.0";
inline constexpr const char* kBuildDate   = __DATE__;
inline constexpr const char* kBuildTime   = __TIME__;
inline constexpr const char* kProjectName = "reuniclus";

/// Full version string: "reuniclus 0.1.0 (Mar 14 2026 12:00:00)"
inline constexpr auto version_string() noexcept {
    return kVersion;  // Use Logger::info(Version::full()) in main for full trace.
}

} // namespace reuniclus
