// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jack Sidman Smith
//
// serve: the clock as a small local service for the website. It keeps the camera, the arm
// interface and the digit network loaded, and answers one JSON line per TCP connection on
// 127.0.0.1 (the website's PHP talks to it; nothing else can reach it):
//
//   {"cmd":"status"}  -> what the display shows, the spare tiles, whether an update runs and
//                        how far it is, and how the last update went
//   {"cmd":"update"}  -> start showing the current local time (24 h); refused while an update
//                        runs or when the display already shows this minute
//
// An update runs in a worker thread, so status stays answerable while the arm moves.
#include "Commands.h"

#include "DigitNet.h"
#include "Log.h"
#include "State.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace spc {
namespace {

std::atomic<bool> gStop{false};

void onSignal(int /*sig*/) { gStop = true; }

// Local wall-clock time as HHMM (the worker runs on Europe/Berlin).
std::string nowHHMM()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[8];
    std::snprintf(buf, sizeof buf, "%02d%02d", tm.tm_hour, tm.tm_min);
    return buf;
}

std::string isoNow()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S%z", &tm);
    return buf;
}

// Closes the descriptor when it goes out of scope.
class Fd {
public:
    explicit Fd(int fd) : fd_(fd) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&&) = delete;
    Fd& operator=(Fd&&) = delete;
    ~Fd()
    {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    [[nodiscard]] int get() const { return fd_; }

private:
    int fd_;
};

class Service {
public:
    Service(App& app, DigitNet& net) : app_(app), net_(net) {}
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;
    Service(Service&&) = delete;
    Service& operator=(Service&&) = delete;
    ~Service()
    {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    nlohmann::json handle(const nlohmann::json& req)
    {
        const std::string cmd = req.value("cmd", "");
        if (cmd == "status") {
            return status();
        }
        if (cmd == "update") {
            return update();
        }
        return {{"ok", false}, {"error", "unknown cmd"}};
    }

private:
    nlohmann::json status()
    {
        nlohmann::json out{{"ok", true}, {"now", nowHHMM()}};
        try {
            const State st = State::load(app_.stateFile);  // saved atomically by the worker
            nlohmann::json display = nlohmann::json::array();
            for (const auto& s : st.display) {
                display.push_back(s.digit);
            }
            if (display.empty()) {
                display = {-1, -1, -1, -1};
            }
            out["display"] = display;
            out["shown"] = st.shown;
            nlohmann::json spare = nlohmann::json::array();
            for (int d = 0; d < 10; ++d) {
                spare.push_back(st.count(d));
            }
            out["spare"] = spare;
        } catch (const std::exception& e) {
            out["ok"] = false;
            out["error"] = std::string("state: ") + e.what();
        }
        const std::lock_guard<std::mutex> lock(mu_);
        out["busy"] = busy_;
        out["target"] = target_;
        out["progress"] = progress_;
        out["last"] = last_;
        return out;
    }

    nlohmann::json update()
    {
        const std::string target = nowHHMM();
        {
            const std::lock_guard<std::mutex> lock(mu_);
            if (busy_) {
                return {{"ok", false}, {"error", "an update is already running"}, {"target", target_}};
            }
            try {
                if (State::load(app_.stateFile).shown == target) {
                    return {{"ok", false}, {"error", "the display already shows the time"}, {"target", target}};
                }
            } catch (const std::exception&) {
                // the update will report it
            }
            busy_ = true;
            target_ = target;
            progress_ = "starting";
        }
        if (worker_.joinable()) {
            worker_.join();  // the previous update, finished
        }
        worker_ = std::thread([this, target] { run(target); });
        return {{"ok", true}, {"target", target}};
    }

    void run(const std::string& target)
    {
        const auto t0 = std::chrono::steady_clock::now();
        nlohmann::json result{{"target", target}, {"started", isoNow()}};
        try {
            log("serve: update to ", target);
            show(app_, target, &net_, [this](const std::string& m) {
                const std::lock_guard<std::mutex> lock(mu_);
                progress_ = m;
            });
            result["ok"] = true;
        } catch (const std::exception& e) {
            log("serve: update failed: ", e.what());
            result["ok"] = false;
            result["error"] = e.what();
            try {
                const ArmLock lock(app_.lockFile, 5.0);  // show() has let go of it by now
                app_.mover.park();  // out of the way; the pump off
            } catch (const std::exception&) {
                // nothing more to do; the next update starts from wherever the arm is
            }
        }
        result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::lock_guard<std::mutex> lock(mu_);
        last_ = result;
        busy_ = false;
        progress_.clear();
    }

    App& app_;
    DigitNet& net_;
    std::mutex mu_;
    bool busy_ = false;
    std::string target_;
    std::string progress_;
    nlohmann::json last_ = nullptr;
    std::thread worker_;
};

std::string readLine(int fd)
{
    std::string line;
    char buf[512];
    while (line.size() < 4096) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) {
            break;
        }
        line.append(buf, static_cast<size_t>(n));
        if (const auto nl = line.find('\n'); nl != std::string::npos) {
            line.resize(nl);
            break;
        }
    }
    return line;
}

}  // namespace

void serve(App& app, int port)
{
    DigitNet net;
    net.load(app.modelFile);
    Service service(app, net);

    const Fd srv(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (srv.get() < 0) {
        throw std::runtime_error(std::string("serve: socket: ") + std::strerror(errno));
    }
    const int one = 1;
    ::setsockopt(srv.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // local only
    if (::bind(srv.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0 ||
        ::listen(srv.get(), 8) != 0) {
        throw std::runtime_error("serve: cannot listen on 127.0.0.1:" + std::to_string(port) + ": " +
                                 std::strerror(errno));
    }
    struct sigaction sa {};
    sa.sa_handler = onSignal;  // no SA_RESTART: accept() returns on SIGTERM
    ::sigaction(SIGTERM, &sa, nullptr);
    ::sigaction(SIGINT, &sa, nullptr);
    log("serve: listening on 127.0.0.1:", port);

    while (!gStop) {
        const Fd c(::accept4(srv.get(), nullptr, nullptr, SOCK_CLOEXEC));
        if (c.get() < 0) {
            if (errno == EINTR) {
                continue;
            }
            log("serve: accept: ", std::strerror(errno));
            continue;
        }
        const timeval tv{2, 0};
        ::setsockopt(c.get(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        nlohmann::json reply;
        const auto req = nlohmann::json::parse(readLine(c.get()), nullptr, false);
        if (req.is_discarded() || !req.is_object()) {
            reply = {{"ok", false}, {"error", "bad request"}};
        } else {
            reply = service.handle(req);
        }
        const std::string out = reply.dump() + "\n";
        if (::send(c.get(), out.data(), out.size(), MSG_NOSIGNAL) < 0) {
            log("serve: send: ", std::strerror(errno));
        }
    }
    log("serve: stopping (an update in progress finishes first)");
}

}  // namespace spc
