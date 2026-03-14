#pragma once

// Phase 2 – Binary UDP Ingestor
//
// Listens on a UDP socket and parses raw NeuralFrame packets.
//
// Why std::memcpy instead of reinterpret_cast:
//   Strict aliasing rules make reinterpret_cast undefined behaviour when the
//   pointer types differ.  memcpy of a known-size POD struct is well-defined,
//   and the compiler generates a single register-width move for small structs.
//
// Protocol assumption: each UDP datagram contains exactly sizeof(NeuralFrame)
// bytes laid out in host byte order.  For production, add a magic header and
// checksum, but the hot-path structure is identical.

#include <reuniclus/Ingestor.hpp>
#include <reuniclus/NeuralFrame.hpp>
#include <reuniclus/SPSCRingBuffer.hpp>
#include <reuniclus/Logger.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <cstddef>
#include <cstring>

// POSIX socket headers – included only on non-Windows or via winsock2 on MSVC.
#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
inline constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
using socket_t = int;
inline constexpr socket_t kInvalidSocket = -1;
#endif

namespace reuniclus {

class UDPIngestor final : public Ingestor {
public:
    UDPIngestor(SPSCRingBuffer<NeuralFrame, 1024>& buffer,
                std::uint16_t port = 12345)
        : buffer_(buffer), port_(port) {}

    ~UDPIngestor() override { stop(); }

    void start() override {
        dropped_  = 0;
        running_  = true;
        worker_   = std::thread([this] { run(); });
        LOG_INFO("UDPIngestor started on port {}", port_);
    }

    void stop() override {
        running_ = false;
        if (sock_ != kInvalidSocket) {
#ifdef _WIN32
            closesocket(sock_);
#else
            close(sock_);
#endif
            sock_ = kInvalidSocket;
        }
        if (worker_.joinable()) worker_.join();
        LOG_INFO("UDPIngestor stopped, dropped_frames={}", dropped_.load());
    }

    [[nodiscard]] std::size_t dropped_frames() const noexcept override {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    void handle_packet(const std::uint8_t* data, std::size_t len,
                       std::uint32_t& seq) {
        if (len < sizeof(NeuralFrame)) return;  // Undersized packet – discard
        NeuralFrame frame;
        std::memcpy(&frame, data, sizeof(NeuralFrame));
        frame.timestamp = std::chrono::high_resolution_clock::now();
        frame.sequence  = seq++;
        if (!buffer_.try_push(frame)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void run() {
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ == kInvalidSocket) {
            LOG_ERROR("UDPIngestor: socket() failed");
            return;
        }

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port_);

        if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            LOG_ERROR("UDPIngestor: bind() failed on port {}", port_);
            return;
        }

        alignas(64) std::uint8_t recv_buf[sizeof(NeuralFrame) + 64];
        std::uint32_t seq = 0;

        while (running_) {
            auto n = ::recv(sock_, reinterpret_cast<char*>(recv_buf),
                            sizeof(recv_buf), 0);
            if (n > 0) {
                handle_packet(recv_buf, static_cast<std::size_t>(n), seq);
            }
        }
    }

    SPSCRingBuffer<NeuralFrame, 1024>& buffer_;
    std::uint16_t                       port_;
    socket_t                            sock_{kInvalidSocket};
    std::thread                         worker_;
    std::atomic<bool>                   running_{false};
    std::atomic<std::size_t>            dropped_{0};
};

} // namespace reuniclus
