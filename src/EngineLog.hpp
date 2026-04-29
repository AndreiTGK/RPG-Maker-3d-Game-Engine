#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <stdexcept>

enum class LogLevel { Info, Warning, Error };

struct LogEntry {
    LogLevel    level;
    std::string timestamp;  // "hh:mm:ss.mmm"
    std::string file;       // __FILE__ (basename)
    int         line = 0;
    std::string message;
};

class EngineLog {
public:
    static EngineLog& get() {
        static EngineLog instance;
        return instance;
    }

    void logAt(LogLevel level, const char* file, int line, const char* fmt, ...) {
        char msgBuf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msgBuf, sizeof(msgBuf), fmt, args);
        va_end(args);

        // Timestamp
        struct timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        struct tm tm_info{};
        localtime_r(&ts.tv_sec, &tm_info);
        char tsBuf[16];
        snprintf(tsBuf, sizeof(tsBuf), "%02d:%02d:%02d.%03d",
                 tm_info.tm_hour, tm_info.tm_min, tm_info.tm_sec,
                 (int)(ts.tv_nsec / 1000000));

        // Basename only
        const char* base = file;
        for (const char* p = file; *p; ++p)
            if (*p == '/' || *p == '\\') base = p + 1;

        LogEntry e{level, tsBuf, base, line, msgBuf};
        entries.push_back(e);
        if ((int)entries.size() > maxEntries)
            entries.erase(entries.begin());
        scrollToBottom = true;

        // Write to file
        if (!logFile) logFile = fopen("engine.log", "w");
        if (logFile) {
            const char* lvlStr = level == LogLevel::Error   ? "ERROR"
                               : level == LogLevel::Warning ? "WARN "
                                                            : "INFO ";
            fprintf(logFile, "[%s][%s] %s:%d  %s\n", tsBuf, lvlStr, base, line, msgBuf);
            fflush(logFile);
        }

        // Mirror to stderr for errors
        if (level == LogLevel::Error)
            fprintf(stderr, "[ERROR] %s:%d  %s\n", base, line, msgBuf);
    }

    void clear() { entries.clear(); }

    std::vector<LogEntry> entries;
    bool scrollToBottom = false;

private:
    static constexpr int maxEntries = 500;
    FILE* logFile = nullptr;

    EngineLog() = default;
    ~EngineLog() { if (logFile) { fclose(logFile); logFile = nullptr; } }
};

// Convenience macros
#define LOG_INFO(...)    EngineLog::get().logAt(LogLevel::Info,    __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) EngineLog::get().logAt(LogLevel::Warning, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)   EngineLog::get().logAt(LogLevel::Error,   __FILE__, __LINE__, __VA_ARGS__)

// Logs the error then throws — use to replace throw std::runtime_error(msg)
[[noreturn]] inline void logAndThrow(const char* file, int line, const std::string& msg) {
    EngineLog::get().logAt(LogLevel::Error, file, line, "%s", msg.c_str());
    throw std::runtime_error(msg);
}
#define THROW_ENGINE_ERROR(msg) ::logAndThrow(__FILE__, __LINE__, (msg))
