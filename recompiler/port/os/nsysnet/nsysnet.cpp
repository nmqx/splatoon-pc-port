#include "../../runtime/rbrew_diagnostics.h"
// nsysnet / nn_ac / nlibcurl
//
// Maps Wii U nsysnet functions to POSIX/Winsock sockets so game network code
// runs against real TCP/IP (LAN or port-forwarded self-hosted NEX servers).
//
// Key Wii U <-> Windows compatibility bridges:
//   - Guest FD table (0..31) <-> native SOCKET handles, so Wii U's 32-bit
//     fd_set bitmaps work even when Winsock SOCKET values exceed 31.
//   - Winsock / POSIX errno -> Wii U SO_* error code translation (e.g.
//     WSAEWOULDBLOCK -> SO_EWOULDBLOCK = 6).
//   - Wii U SOL_SOCKET (-1 / 0xFFFFFFFF), SO_NBIO (0x1016), SO_BIO (0x1014),
//     MSG_DONTWAIT (0x20) translation, and Windows SIO_UDP_CONNRESET disable.
//   - sendto_multi and recvfrom_ex support for PIA transport.
//   - Optional RBREW_NEX_HOST / RBREW_NEX_PORT / RBREW_LOCAL_IP /
//     RBREW_AC_CONNECT environment overrides and RBREW_NET_TRACE logging.
//
// Wii U sockaddr_in layout (BSD, big-endian in guest memory):
//   +0: uint8  sin_len    (= 0x10)
//   +1: uint8  sin_family (AF_INET = 2)
//   +2: uint16 sin_port   (network byte order)
//   +4: uint32 sin_addr   (network byte order)
//   +8: uint8  sin_zero[8]

#include "../os_common.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32")
typedef int socklen_t;
typedef unsigned short sa_family_t;
typedef int ssize_t;
typedef SOCKET NativeSocket;
#define INVALID_NATIVE_SOCKET INVALID_SOCKET
#define CLOSE_SOCKET(fd) closesocket(fd)
#define SOCK_ERRNO WSAGetLastError()
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
typedef int NativeSocket;
#define INVALID_NATIVE_SOCKET (-1)
#define CLOSE_SOCKET(fd) close(fd)
#define SOCK_ERRNO errno
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace {

// ---------------------------------------------------------------------------
// Wii U nsysnet SO_* error codes (matches WUT / Cemu nsysnet)
// ---------------------------------------------------------------------------
constexpr int kSoSuccess       = 0;
constexpr int kSoEnobufs       = 1;
constexpr int kSoEtimedout     = 2;
constexpr int kSoEisconn       = 3;
constexpr int kSoEopnotsupp    = 4;
constexpr int kSoEconnaborted  = 5;
constexpr int kSoEwouldblock   = 6; // == SO_EAGAIN
constexpr int kSoEconnrefused  = 7;
constexpr int kSoEconnreset    = 8;
constexpr int kSoEnotconn      = 9;
constexpr int kSoEalready      = 10;
constexpr int kSoEinval        = 11;
constexpr int kSoEmsgsize      = 12;
constexpr int kSoEpipe         = 13;
constexpr int kSoEdestaddrreq  = 14;
constexpr int kSoEshutdown     = 15;
constexpr int kSoEnoprotoopt   = 16;
constexpr int kSoEhaveoob      = 17;
constexpr int kSoEnomem        = 18;
constexpr int kSoEaddrnotavail = 19;
constexpr int kSoEaddrinuse    = 20;
constexpr int kSoEafnosupport  = 21;
constexpr int kSoEinprogress   = 22;
constexpr int kSoEnotsock      = 24;
constexpr int kSoEfault        = 29;
constexpr int kSoEnetreset     = 30;
constexpr int kSoEnetunreach   = 31;
constexpr int kSoEhostunreach  = 32;
constexpr int kSoEnetdown      = 38;
constexpr int kSoEhostdown     = 39;
constexpr int kSoEintr         = 46;
constexpr int kSoEacces        = 47;
constexpr int kSoEmfile        = 48;
constexpr int kSoEbadf         = 49;

// Wii U socket flags
constexpr int kWiiMsgOob       = 0x01;
constexpr int kWiiMsgPeek      = 0x02;
constexpr int kWiiMsgDontRoute = 0x04;
constexpr int kWiiMsgDontWait  = 0x20;

// Wii U SOL_SOCKET options
constexpr int kWiiSoDebug      = 0x0001;
constexpr int kWiiSoReuseAddr  = 0x0004;
constexpr int kWiiSoKeepAlive  = 0x0008;
constexpr int kWiiSoDontRoute  = 0x0010;
constexpr int kWiiSoBroadcast  = 0x0020;
constexpr int kWiiSoLinger     = 0x0080;
constexpr int kWiiSoOobInline  = 0x0100;
constexpr int kWiiSoTcpSack    = 0x0200;
constexpr int kWiiSoWinScale   = 0x0400;
constexpr int kWiiSoSndBuf     = 0x1001;
constexpr int kWiiSoRcvBuf     = 0x1002;
constexpr int kWiiSoSndTimeo   = 0x1005;
constexpr int kWiiSoRcvTimeo   = 0x1006;
constexpr int kWiiSoError      = 0x1007;
constexpr int kWiiSoType       = 0x1008;
constexpr int kWiiSoBio        = 0x1014;
constexpr int kWiiSoNbio       = 0x1016;
constexpr int kWiiSoRusrBuf    = 0x10000;

thread_local int s_last_so_err = kSoSuccess;
thread_local int s_last_host_err = 0;
thread_local char s_trace_detail[160] = {};

int map_host_errno(int err) {
    if (err == 0) return kSoSuccess;
#if defined(_WIN32)
    switch (err) {
    case WSAENOBUFS:        return kSoEnobufs;
    case WSAETIMEDOUT:      return kSoEtimedout;
    case WSAEISCONN:        return kSoEisconn;
    case WSAEOPNOTSUPP:     return kSoEopnotsupp;
    case WSAECONNABORTED:   return kSoEconnaborted;
    case WSAEWOULDBLOCK:    return kSoEwouldblock;
    case WSAECONNREFUSED:   return kSoEconnrefused;
    case WSAECONNRESET:     return kSoEconnreset;
    case WSAENOTCONN:       return kSoEnotconn;
    case WSAEALREADY:       return kSoEalready;
    case WSAEINVAL:         return kSoEinval;
    case WSAEMSGSIZE:       return kSoEmsgsize;
    case WSAEDESTADDRREQ:   return kSoEdestaddrreq;
    case WSAESHUTDOWN:      return kSoEshutdown;
    case WSAENOPROTOOPT:    return kSoEnoprotoopt;
    case WSA_NOT_ENOUGH_MEMORY: return kSoEnomem;
    case WSAEADDRNOTAVAIL:  return kSoEaddrnotavail;
    case WSAEADDRINUSE:     return kSoEaddrinuse;
    case WSAEAFNOSUPPORT:   return kSoEafnosupport;
    case WSAEINPROGRESS:    return kSoEinprogress;
    case WSAENOTSOCK:       return kSoEnotsock;
    case WSAEFAULT:         return kSoEfault;
    case WSAENETRESET:      return kSoEnetreset;
    case WSAENETUNREACH:    return kSoEnetunreach;
    case WSAEHOSTUNREACH:   return kSoEhostunreach;
    case WSAENETDOWN:       return kSoEnetdown;
    case WSAEHOSTDOWN:      return kSoEhostdown;
    case WSAEINTR:          return kSoEintr;
    case WSAEACCES:         return kSoEacces;
    case WSAEMFILE:         return kSoEmfile;
    case WSAEBADF:          return kSoEbadf;
    default:                return kSoEinval;
    }
#else
    switch (err) {
    case ENOBUFS:           return kSoEnobufs;
    case ETIMEDOUT:         return kSoEtimedout;
    case EISCONN:           return kSoEisconn;
    case EOPNOTSUPP:        return kSoEopnotsupp;
    case ECONNABORTED:      return kSoEconnaborted;
    case EWOULDBLOCK:       return kSoEwouldblock;
#if EAGAIN != EWOULDBLOCK
    case EAGAIN:            return kSoEwouldblock;
#endif
    case ECONNREFUSED:      return kSoEconnrefused;
    case ECONNRESET:        return kSoEconnreset;
    case ENOTCONN:          return kSoEnotconn;
    case EALREADY:          return kSoEalready;
    case EINVAL:            return kSoEinval;
    case EMSGSIZE:          return kSoEmsgsize;
    case EPIPE:             return kSoEpipe;
    case EDESTADDRREQ:      return kSoEdestaddrreq;
    case ESHUTDOWN:         return kSoEshutdown;
    case ENOPROTOOPT:       return kSoEnoprotoopt;
    case ENOMEM:            return kSoEnomem;
    case EADDRNOTAVAIL:     return kSoEaddrnotavail;
    case EADDRINUSE:        return kSoEaddrinuse;
    case EAFNOSUPPORT:      return kSoEafnosupport;
    case EINPROGRESS:       return kSoEinprogress;
    case ENOTSOCK:          return kSoEnotsock;
    case EFAULT:            return kSoEfault;
    case ENETRESET:         return kSoEnetreset;
    case ENETUNREACH:       return kSoEnetunreach;
    case EHOSTUNREACH:      return kSoEhostunreach;
    case ENETDOWN:          return kSoEnetdown;
    case EHOSTDOWN:         return kSoEhostdown;
    case EINTR:             return kSoEintr;
    case EACCES:            return kSoEacces;
    case EMFILE:            return kSoEmfile;
    case EBADF:             return kSoEbadf;
    default:                return kSoEinval;
    }
#endif
}

void set_so_err(int so_err, int host_err = 0) {
    s_last_so_err = so_err;
    s_last_host_err = host_err;
}

void record_host_error() {
    const int herr = SOCK_ERRNO;
    s_last_host_err = herr;
    s_last_so_err = map_host_errno(herr);
}

void clear_so_err() {
    s_last_so_err = kSoSuccess;
    s_last_host_err = 0;
}

void ensure_wsa_init() {
#if defined(_WIN32)
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA wd{};
        WSAStartup(MAKEWORD(2, 2), &wd);
    });
#endif
}

// ---------------------------------------------------------------------------
// Guest FD (0..31) <-> Native SOCKET table
// ---------------------------------------------------------------------------
constexpr int kMaxGuestFds = 32;

struct GuestFdEntry {
    bool in_use = false;
    NativeSocket sock = INVALID_NATIVE_SOCKET;
    int type = 0;
    bool nonblocking = false;
};

std::mutex s_fd_mutex;
GuestFdEntry s_fd_table[kMaxGuestFds];

int alloc_guest_fd(NativeSocket sock, int type) {
    std::lock_guard<std::mutex> lock(s_fd_mutex);
    for (int i = 0; i < kMaxGuestFds; ++i) {
        if (!s_fd_table[i].in_use) {
            s_fd_table[i].in_use = true;
            s_fd_table[i].sock = sock;
            s_fd_table[i].type = type;
            s_fd_table[i].nonblocking = false;
            return i;
        }
    }
    return -1;
}

bool lookup_guest_fd(int gfd, NativeSocket* out_sock, bool* out_nonblocking = nullptr, int* out_type = nullptr) {
    if (gfd < 0 || gfd >= kMaxGuestFds) {
        set_so_err(kSoEbadf);
        return false;
    }
    std::lock_guard<std::mutex> lock(s_fd_mutex);
    if (!s_fd_table[gfd].in_use) {
        set_so_err(kSoEbadf);
        return false;
    }
    if (out_sock) *out_sock = s_fd_table[gfd].sock;
    if (out_nonblocking) *out_nonblocking = s_fd_table[gfd].nonblocking;
    if (out_type) *out_type = s_fd_table[gfd].type;
    return true;
}

void set_guest_fd_nonblocking_flag(int gfd, bool nb) {
    if (gfd < 0 || gfd >= kMaxGuestFds) return;
    std::lock_guard<std::mutex> lock(s_fd_mutex);
    if (s_fd_table[gfd].in_use) s_fd_table[gfd].nonblocking = nb;
}

bool release_guest_fd(int gfd, NativeSocket* out_sock) {
    if (gfd < 0 || gfd >= kMaxGuestFds) {
        set_so_err(kSoEbadf);
        return false;
    }
    std::lock_guard<std::mutex> lock(s_fd_mutex);
    if (!s_fd_table[gfd].in_use) {
        set_so_err(kSoEbadf);
        return false;
    }
    if (out_sock) *out_sock = s_fd_table[gfd].sock;
    s_fd_table[gfd] = GuestFdEntry{};
    return true;
}

bool set_native_nonblocking(NativeSocket sock, bool enable) {
#if defined(_WIN32)
    u_long mode = enable ? 1u : 0u;
    return ioctlsocket(sock, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) return false;
    if (enable)
        flags |= O_NONBLOCK;
    else
        flags &= ~O_NONBLOCK;
    return fcntl(sock, F_SETFL, flags) == 0;
#endif
}

// ---------------------------------------------------------------------------
// Environment configuration for self-hosted multiplayer / redirects
// ---------------------------------------------------------------------------
struct NetConfig {
    bool online_enabled = false;
    bool has_nex_host = false;
    std::string nex_host;
    uint32_t nex_host_ipv4_nbo = 0; // network byte order
    uint16_t nex_port = 0;          // host byte order, 0 = not overridden
    bool force_redirect = false;
    uint32_t local_ipv4_nbo = 0;    // network byte order
};

bool resolve_ipv4_nbo(const char* host, uint32_t* out_nbo) {
    if (!host || !*host || !out_nbo) return false;
    struct in_addr ia{};
    if (inet_pton(AF_INET, host, &ia) == 1) {
        *out_nbo = ia.s_addr;
        return true;
    }
    ensure_wsa_init();
    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &res) == 0 && res) {
        auto* sin = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
        *out_nbo = sin->sin_addr.s_addr;
        freeaddrinfo(res);
        return true;
    }
    return false;
}

uint32_t detect_local_ipv4_nbo(uint32_t target_hint_nbo) {
    ensure_wsa_init();
    // If target is loopback (127.x.x.x), prefer 127.0.0.1 so local loopback peers work cleanly.
    const uint8_t first_octet = uint8_t(target_hint_nbo & 0xFFu);
    if (first_octet == 127u) {
        struct in_addr ia{};
        inet_pton(AF_INET, "127.0.0.1", &ia);
        return ia.s_addr;
    }
    NativeSocket s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s != INVALID_NATIVE_SOCKET) {
        struct sockaddr_in dst{};
        dst.sin_family = AF_INET;
        dst.sin_port = htons(53);
        if (target_hint_nbo != 0) {
            dst.sin_addr.s_addr = target_hint_nbo;
        } else {
            inet_pton(AF_INET, "8.8.8.8", &dst.sin_addr);
        }
        if (connect(s, reinterpret_cast<struct sockaddr*>(&dst), sizeof(dst)) == 0) {
            struct sockaddr_in local{};
            socklen_t llen = sizeof(local);
            if (getsockname(s, reinterpret_cast<struct sockaddr*>(&local), &llen) == 0 &&
                local.sin_addr.s_addr != 0) {
                CLOSE_SOCKET(s);
                return local.sin_addr.s_addr;
            }
        }
        CLOSE_SOCKET(s);
    }
    struct in_addr fallback{};
    inet_pton(AF_INET, "127.0.0.1", &fallback);
    return fallback.s_addr;
}

const NetConfig& get_net_config() {
    static const NetConfig cfg = [] {
        NetConfig c;
        const char* nex_host = std::getenv("RBREW_NEX_HOST");
        const char* nex_port = std::getenv("RBREW_NEX_PORT");
        const char* ac_connect = std::getenv("RBREW_AC_CONNECT");
        const char* online = std::getenv("RBREW_ONLINE");
        const char* force_redir = std::getenv("RBREW_NEX_FORCE_REDIRECT");
        const char* local_ip = std::getenv("RBREW_LOCAL_IP");

        if (nex_host && *nex_host) {
            c.has_nex_host = true;
            c.nex_host = nex_host;
            c.online_enabled = true;
            resolve_ipv4_nbo(nex_host, &c.nex_host_ipv4_nbo);
        }
        if (nex_port && *nex_port) {
            const long p = std::strtol(nex_port, nullptr, 10);
            if (p > 0 && p <= 65535) c.nex_port = static_cast<uint16_t>(p);
        }
        if (ac_connect && std::strcmp(ac_connect, "ok") == 0) {
            c.online_enabled = true;
        }
        if (online && (std::strcmp(online, "1") == 0 || std::strcmp(online, "true") == 0)) {
            c.online_enabled = true;
        }
        if (force_redir && std::strcmp(force_redir, "1") == 0) {
            c.force_redirect = true;
        }
        if (local_ip && *local_ip) {
            resolve_ipv4_nbo(local_ip, &c.local_ipv4_nbo);
        }
        if (c.local_ipv4_nbo == 0 && c.online_enabled) {
            c.local_ipv4_nbo = detect_local_ipv4_nbo(c.nex_host_ipv4_nbo);
        }
        return c;
    }();
    return cfg;
}

bool contains_icase(const char* haystack, const char* needle) {
    if (!haystack || !needle || !*needle) return false;
    const size_t nlen = std::strlen(needle);
    for (const char* p = haystack; *p; ++p) {
        size_t i = 0;
        while (i < nlen && p[i] &&
               std::tolower(static_cast<unsigned char>(p[i])) == std::tolower(static_cast<unsigned char>(needle[i]))) {
            ++i;
        }
        if (i == nlen) return true;
    }
    return false;
}

bool should_redirect_hostname(const char* hostname) {
    if (!hostname || !*hostname) return false;
    const auto& cfg = get_net_config();
    if (!cfg.has_nex_host) return false;
    return contains_icase(hostname, "nintendowifi.net") ||
           contains_icase(hostname, "nintendo.net") ||
           contains_icase(hostname, "pretendo.cc");
}

bool is_private_or_loopback_ipv4(uint32_t addr_nbo) {
    const uint32_t ip = ntohl(addr_nbo);
    const uint8_t a = uint8_t(ip >> 24);
    const uint8_t b = uint8_t(ip >> 16);
    if (a == 127) return true;                     // 127.0.0.0/8
    if (a == 10) return true;                      // 10.0.0.0/8
    if (a == 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12
    if (a == 192 && b == 168) return true;         // 192.168.0.0/16
    return false;
}

// Second NAT check server on a one-IP server. PIA 4.9 (v272) needs nncs1 and nncs2 at two different
// IPv4 addresses (its type-103 test, else "NatPropertyDetecter: Missing mandatory messages"), as
// Pretendo runs them. nncs2 resolves to a placeholder address (192.0.2.2, TEST-NET, never routed);
// packets to it go to the NEX host on port + 1000 (11025 / 11125, multiplayer/nncs_server.py) and
// the answers come back as if from the placeholder. RBREW_NNCS_SPLIT=0 resolves nncs2 to the host.
bool nncs_split_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("RBREW_NNCS_SPLIT");
        return !(v && v[0] == '0');
    }();
    return on && get_net_config().has_nex_host && get_net_config().nex_host_ipv4_nbo != 0;
}
constexpr uint32_t kNncs2PlaceholderHbo = 0xC0000202u; // 192.0.2.2
constexpr uint16_t kNncs2PortOffset = 1000;
bool is_nncs2_hostname(const char* hostname) {
    return nncs_split_enabled() && contains_icase(hostname, "nncs2.");
}
void nncs_split_outgoing(struct sockaddr_in* sa) {
    if (sa->sin_family != AF_INET || sa->sin_addr.s_addr != htonl(kNncs2PlaceholderHbo) || !nncs_split_enabled()) return;
    sa->sin_addr.s_addr = get_net_config().nex_host_ipv4_nbo;
    sa->sin_port = htons(uint16_t(ntohs(sa->sin_port) + kNncs2PortOffset));
}
void nncs_split_incoming(struct sockaddr_in* sa) {
    if (!nncs_split_enabled() || sa->sin_addr.s_addr != get_net_config().nex_host_ipv4_nbo) return;
    const uint16_t port = ntohs(sa->sin_port);
    if (port == 10025 + kNncs2PortOffset || port == 10125 + kNncs2PortOffset) {
        sa->sin_addr.s_addr = htonl(kNncs2PlaceholderHbo);
        sa->sin_port = htons(uint16_t(port - kNncs2PortOffset));
    }
}

void maybe_redirect_sockaddr(struct sockaddr_in* sa) {
    const auto& cfg = get_net_config();
    if (!cfg.has_nex_host || cfg.nex_host_ipv4_nbo == 0) return;
    nncs_split_outgoing(sa);
    if (!cfg.force_redirect) return;
    if (sa->sin_family != AF_INET) return;
    // Never rewrite packets already targeted at the configured NEX host (e.g.
    // Secure server port 60001 or NNCS 10025/10125 over WAN port forwarding)
    // or private/loopback addresses.
    if (sa->sin_addr.s_addr == 0 || sa->sin_addr.s_addr == cfg.nex_host_ipv4_nbo ||
        is_private_or_loopback_ipv4(sa->sin_addr.s_addr))
        return;
    sa->sin_addr.s_addr = cfg.nex_host_ipv4_nbo;
    if (cfg.nex_port != 0) sa->sin_port = htons(cfg.nex_port);
}

void format_sockaddr_str(const struct sockaddr_in& sa, char* out, size_t cap) {
    char ipbuf[32] = "?";
    inet_ntop(AF_INET, const_cast<struct in_addr*>(&sa.sin_addr), ipbuf, sizeof(ipbuf));
    std::snprintf(out, cap, "%s:%u", ipbuf, unsigned(ntohs(sa.sin_port)));
}

int translate_msg_flags(uint32_t wii_flags, bool* out_dontwait) {
    if (out_dontwait) *out_dontwait = (wii_flags & kWiiMsgDontWait) != 0;
    int host_flags = 0;
    if (wii_flags & kWiiMsgOob) host_flags |= MSG_OOB;
    if (wii_flags & kWiiMsgPeek) host_flags |= MSG_PEEK;
    if (wii_flags & kWiiMsgDontRoute) host_flags |= MSG_DONTROUTE;
    return host_flags;
}

bool socket_is_readable_now(NativeSocket sock) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(sock, &rfds);
    struct timeval tv{0, 0};
#if defined(_WIN32)
    int rc = select(0, &rfds, nullptr, nullptr, &tv);
#else
    int rc = select(sock + 1, &rfds, nullptr, nullptr, &tv);
#endif
    return rc > 0 && FD_ISSET(sock, &rfds);
}

bool socket_is_writable_now(NativeSocket sock) {
    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(sock, &wfds);
    struct timeval tv{0, 0};
#if defined(_WIN32)
    int rc = select(0, nullptr, &wfds, nullptr, &tv);
#else
    int rc = select(sock + 1, nullptr, &wfds, nullptr, &tv);
#endif
    return rc > 0 && FD_ISSET(sock, &wfds);
}

// ---------------------------------------------------------------------------
// Scratch area for gethostbyname / getaddrinfo / inet_ntoa results.
// All offsets are relative to NET_SCRATCH_BASE in the guest arena.
// ---------------------------------------------------------------------------
constexpr uint32_t NET_SCRATCH_BASE = 0x00100000u; // safely below game code

constexpr uint32_t HE_BASE         = NET_SCRATCH_BASE + 0x000u;
constexpr uint32_t HE_ALIASES_ARR  = NET_SCRATCH_BASE + 0x020u; // uint32[2] = {0,0}
constexpr uint32_t HE_ADDRLIST_ARR = NET_SCRATCH_BASE + 0x028u; // uint32[2] = {addr,0}
constexpr uint32_t HE_ADDR         = NET_SCRATCH_BASE + 0x030u; // 4-byte IP
constexpr uint32_t HE_NAME_STR     = NET_SCRATCH_BASE + 0x040u; // hostname (64 chars)

constexpr uint32_t AI_BASE         = NET_SCRATCH_BASE + 0x100u;
constexpr uint32_t AI_ENTRY_STRIDE = 48u; // 32 (addrinfo) + 16 (sockaddr_in)
constexpr uint32_t AI_MAX_ENTRIES  = 4u;
constexpr uint32_t AI_CANON_STR    = NET_SCRATCH_BASE + 0x200u; // 64 chars

constexpr uint32_t INET_NTOA_BASE  = NET_SCRATCH_BASE + 0x280u;
constexpr uint32_t INET_NTOA_SLOT  = 20u;
std::atomic<uint32_t> s_ntoa_slot{0};
std::mutex s_scratch_mutex;

// ---------------------------------------------------------------------------
// sockaddr_in guest <-> host conversion
// ---------------------------------------------------------------------------
void guest_to_host_sockaddr(const uint8_t* mem, uint32_t gptr, struct sockaddr_in* out) {
    std::memset(out, 0, sizeof(*out));
    if (!gptr) return;
    out->sin_family = static_cast<sa_family_t>(mem[gptr + 1]); // byte 1 = sin_family
    std::memcpy(&out->sin_port, mem + gptr + 2, 2);            // already network order
    std::memcpy(&out->sin_addr.s_addr, mem + gptr + 4, 4);     // already network order
}

void host_to_guest_sockaddr(uint8_t* mem, uint32_t gptr, const struct sockaddr_in* in) {
    if (!gptr) return;
    mem[gptr + 0] = 0x10;                                   // sin_len = 16
    mem[gptr + 1] = static_cast<uint8_t>(in->sin_family);   // sin_family
    std::memcpy(mem + gptr + 2, &in->sin_port, 2);          // network order
    std::memcpy(mem + gptr + 4, &in->sin_addr.s_addr, 4);   // network order
    std::memset(mem + gptr + 8, 0, 8);                      // sin_zero
}

// ---------------------------------------------------------------------------
// setsockopt level/option mapping: Wii U BSD -> host
// ---------------------------------------------------------------------------
int map_level(int wii_level) {
    if (wii_level == -1 || wii_level == 0xFFFF) return SOL_SOCKET;
    if (wii_level == 6) return IPPROTO_TCP;
    if (wii_level == 0) return IPPROTO_IP;
    return wii_level;
}

int map_so_option(int wii_opt) {
    switch (wii_opt) {
    case kWiiSoDebug:     return SO_DEBUG;
    case kWiiSoReuseAddr: return SO_REUSEADDR;
    case kWiiSoKeepAlive: return SO_KEEPALIVE;
    case kWiiSoDontRoute: return SO_DONTROUTE;
    case kWiiSoBroadcast: return SO_BROADCAST;
    case kWiiSoLinger:    return SO_LINGER;
    case kWiiSoOobInline: return SO_OOBINLINE;
    case kWiiSoSndBuf:    return SO_SNDBUF;
    case kWiiSoRcvBuf:    return SO_RCVBUF;
    case kWiiSoError:     return SO_ERROR;
    case kWiiSoType:      return SO_TYPE;
    default:              return wii_opt;
    }
}

int map_tcp_option(int wii_opt) {
    if (wii_opt == 0x2004 || wii_opt == 1) return TCP_NODELAY;
    return wii_opt;
}

// ---------------------------------------------------------------------------
// select() fd_set conversion through guest FD table
// ---------------------------------------------------------------------------
struct MappedFdSet {
    fd_set set;
    int count = 0;
    NativeSocket max_sock = 0;
    NativeSocket guest_to_native[kMaxGuestFds];
    bool guest_active[kMaxGuestFds];
};

void build_host_fdset(uint8_t* mem, uint32_t gptr, int nfds, MappedFdSet* out) {
    FD_ZERO(&out->set);
    out->count = 0;
    out->max_sock = 0;
    std::memset(out->guest_active, 0, sizeof(out->guest_active));
    for (int i = 0; i < kMaxGuestFds; ++i) out->guest_to_native[i] = INVALID_NATIVE_SOCKET;
    if (!gptr || !rbrew_memory_accessible(mem, gptr, 4u)) return;

    const uint32_t bits = rbrew_read32(mem, gptr);
    const int limit = std::min(kMaxGuestFds, std::max(0, nfds));
    std::lock_guard<std::mutex> lock(s_fd_mutex);
    for (int i = 0; i < limit; ++i) {
        if ((bits & (1u << i)) == 0) continue;
        if (!s_fd_table[i].in_use) continue;
        const NativeSocket s = s_fd_table[i].sock;
        FD_SET(s, &out->set);
        out->guest_to_native[i] = s;
        out->guest_active[i] = true;
        out->count++;
        if (s > out->max_sock) out->max_sock = s;
    }
}

void write_guest_fdset(uint8_t* mem, uint32_t gptr, const MappedFdSet& mapped) {
    if (!gptr || !rbrew_memory_accessible(mem, gptr, 4u)) return;
    uint32_t bits = 0;
    for (int i = 0; i < kMaxGuestFds; ++i) {
        if (mapped.guest_active[i] && FD_ISSET(mapped.guest_to_native[i], &mapped.set)) {
            bits |= (1u << i);
        }
    }
    rbrew_write32(mem, gptr, bits);
}

// ---------------------------------------------------------------------------
// socket_lib_init / NSSLInit — initialise the socket library
// ---------------------------------------------------------------------------
void socket_lib_init(CPUState* cpu) {
    ensure_wsa_init();
    clear_so_err();
    RBREW_LOG_INFO("nsysnet", "socket_lib_init\n");
    RET = 0;
}

void NSSLInit(CPUState* cpu) {
    (void)cpu;
    clear_so_err();
    RET = 0;
}

void set_resolver_allocator(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// mw_socket / socketclose / socketlasterr
// ---------------------------------------------------------------------------
void mw_socket(CPUState* cpu) {
    ensure_wsa_init();
    const int domain = static_cast<int>(ARG0);
    const int type = static_cast<int>(ARG1);
    const int proto = static_cast<int>(ARG2);

    NativeSocket s = socket(domain, type, proto);
    if (s == INVALID_NATIVE_SOCKET) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }

#if defined(_WIN32)
    // Prevent ICMP Port Unreachable on UDP from causing subsequent recvfrom()
    // calls to fail with WSAECONNRESET (classic Winsock UDP pitfall).
    if (type == SOCK_DGRAM) {
        BOOL new_behavior = FALSE;
        DWORD bytes_returned = 0;
        WSAIoctl(s, SIO_UDP_CONNRESET, &new_behavior, sizeof(new_behavior), nullptr, 0, &bytes_returned, nullptr,
                 nullptr);
    }
#endif

    const int gfd = alloc_guest_fd(s, type);
    if (gfd < 0) {
        CLOSE_SOCKET(s);
        set_so_err(kSoEmfile);
        RET = static_cast<uint32_t>(-1);
        return;
    }

    clear_so_err();
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "gfd=%d native=%llu type=%d", gfd,
                  static_cast<unsigned long long>(s), type);
    RET = static_cast<uint32_t>(gfd);
}

void socketclose(CPUState* cpu) {
    const int gfd = static_cast<int>(ARG0);
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!release_guest_fd(gfd, &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const int rc = CLOSE_SOCKET(s);
    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
    } else {
        clear_so_err();
        RET = 0;
    }
}

void socketlasterr(CPUState* cpu) {
    (void)cpu;
    RET = static_cast<uint32_t>(s_last_so_err);
}

// ---------------------------------------------------------------------------
// bind / connect / shutdown / getsockname
// ---------------------------------------------------------------------------
void bind_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    struct sockaddr_in sa{};
    guest_to_host_sockaddr(MEM, ARG1, &sa);
    // RBREW_BIND_IP=<ipv4>: bind "any address" sockets to this local address instead, so two
    // instances on one PC show different source IPs (PIA's host refuses a station whose IP matches
    // an already-known station with another connection id: "battle full" when both share one IP).
    static const uint32_t s_bind_ip_nbo = [] {
        const char* v = std::getenv("RBREW_BIND_IP");
        uint32_t nbo = 0;
        if (v && *v) resolve_ipv4_nbo(v, &nbo);
        return nbo;
    }();
    if (sa.sin_family == AF_INET && sa.sin_addr.s_addr == 0 && s_bind_ip_nbo != 0) sa.sin_addr.s_addr = s_bind_ip_nbo;
    const int rc = bind(s, reinterpret_cast<struct sockaddr*>(&sa), static_cast<socklen_t>(sizeof(sa)));
    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    struct sockaddr_in bound{};
    socklen_t blen = sizeof(bound);
    if (getsockname(s, reinterpret_cast<struct sockaddr*>(&bound), &blen) == 0) {
        char addr[48];
        format_sockaddr_str(bound, addr, sizeof(addr));
        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "bound=%s", addr);
    }
    RET = 0;
}

void connect_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    struct sockaddr_in sa{};
    guest_to_host_sockaddr(MEM, ARG1, &sa);
    maybe_redirect_sockaddr(&sa);
    const int rc = connect(s, reinterpret_cast<struct sockaddr*>(&sa), static_cast<socklen_t>(sizeof(sa)));
    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    RET = 0;
}

void shutdown_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const int rc = shutdown(s, static_cast<int>(ARG1));
    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    RET = 0;
}

void getsockname_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    struct sockaddr_in sa{};
    socklen_t slen = sizeof(sa);
    const int rc = getsockname(s, reinterpret_cast<struct sockaddr*>(&sa), &slen);
    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    // If socket is bound to INADDR_ANY (0.0.0.0) in online mode, report the
    // configured/detected local IP so StationURLs don't advertise 0.0.0.0.
    const auto& cfg = get_net_config();
    if (sa.sin_addr.s_addr == 0 && cfg.online_enabled && cfg.local_ipv4_nbo != 0) {
        sa.sin_addr.s_addr = cfg.local_ipv4_nbo;
    }
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 16u)) host_to_guest_sockaddr(MEM, ARG1, &sa);
    if (ARG2 && rbrew_memory_accessible(MEM, ARG2, 4u)) rbrew_write32(MEM, ARG2, 16u);
    clear_so_err();
    char addr[48];
    format_sockaddr_str(sa, addr, sizeof(addr));
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "local=%s", addr);
    RET = 0;
}

// ---------------------------------------------------------------------------
// send / recv / sendto / recvfrom / recvfrom_ex / sendto_multi
// ---------------------------------------------------------------------------
void send_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    bool nb = false;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s, &nb)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const uint32_t len = ARG2;
    if (!ARG1 || !rbrew_memory_accessible(MEM, ARG1, len)) {
        set_so_err(kSoEfault);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    bool dontwait = false;
    const int hflags = translate_msg_flags(ARG3, &dontwait);
    if (dontwait && !nb && !socket_is_writable_now(s)) {
        set_so_err(kSoEwouldblock);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const ssize_t n = send(s, reinterpret_cast<const char*>(MEM + ARG1), static_cast<int>(len), hflags);
    if (n < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    RET = static_cast<uint32_t>(n);
}

void recv_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    bool nb = false;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s, &nb)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const uint32_t len = ARG2;
    if (!ARG1 || !rbrew_memory_accessible(MEM, ARG1, len)) {
        set_so_err(kSoEfault);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    bool dontwait = false;
    const int hflags = translate_msg_flags(ARG3, &dontwait);
    if (dontwait && !nb && !socket_is_readable_now(s)) {
        set_so_err(kSoEwouldblock);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const ssize_t n = recv(s, reinterpret_cast<char*>(MEM + ARG1), static_cast<int>(len), hflags);
    if (n < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    RET = static_cast<uint32_t>(n);
}

void sendto_(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    bool nb = false;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s, &nb)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const uint32_t len = ARG2;
    if (!ARG1 || !rbrew_memory_accessible(MEM, ARG1, len)) {
        set_so_err(kSoEfault);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    if (!ARG4 || !rbrew_memory_accessible(MEM, ARG4, 16u)) {
        set_so_err(kSoEdestaddrreq);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    struct sockaddr_in sa{};
    guest_to_host_sockaddr(MEM, ARG4, &sa);
    maybe_redirect_sockaddr(&sa);
    bool dontwait = false;
    const int hflags = translate_msg_flags(ARG3, &dontwait);
    if (dontwait && !nb && !socket_is_writable_now(s)) {
        set_so_err(kSoEwouldblock);
        RET = static_cast<uint32_t>(-1);
        return;
    }

    const ssize_t n = sendto(s, reinterpret_cast<const char*>(MEM + ARG1), static_cast<int>(len), hflags,
                             reinterpret_cast<struct sockaddr*>(&sa), static_cast<socklen_t>(sizeof(sa)));
    if (n < 0) {
        record_host_error();
        char addr[48];
        format_sockaddr_str(sa, addr, sizeof(addr));
        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "dst=%s len=%u FAIL", addr, len);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    clear_so_err();
    char addr[48];
    format_sockaddr_str(sa, addr, sizeof(addr));
    const uint8_t b0 = len > 0 ? MEM[ARG1] : 0;
    const uint8_t b1 = len > 1 ? MEM[ARG1 + 1] : 0;
    const uint8_t b2 = len > 2 ? MEM[ARG1 + 2] : 0;
    const uint8_t b3 = len > 3 ? MEM[ARG1 + 3] : 0;
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "dst=%s len=%u head=%02x%02x%02x%02x", addr, len, b0, b1, b2,
                  b3);
    RET = static_cast<uint32_t>(n);
}

void recvfrom_impl(CPUState* cpu, bool is_ex) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    bool nb = false;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s, &nb)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const uint32_t len = ARG2;
    if (!ARG1 || !rbrew_memory_accessible(MEM, ARG1, len)) {
        set_so_err(kSoEfault);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    bool dontwait = false;
    const int hflags = translate_msg_flags(ARG3, &dontwait);
    if (dontwait && !nb && !socket_is_readable_now(s)) {
        set_so_err(kSoEwouldblock);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    struct sockaddr_in sa{};
    socklen_t slen = sizeof(sa);
    const ssize_t n = recvfrom(s, reinterpret_cast<char*>(MEM + ARG1), static_cast<int>(len), hflags,
                               reinterpret_cast<struct sockaddr*>(&sa), &slen);
    if (n < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }
    nncs_split_incoming(&sa);
    if (ARG4 && rbrew_memory_accessible(MEM, ARG4, 16u)) host_to_guest_sockaddr(MEM, ARG4, &sa);
    if (ARG5 && rbrew_memory_accessible(MEM, ARG5, 4u)) rbrew_write32(MEM, ARG5, 16u);
    if (is_ex && ARG6 && ARG7 > 0) {
        // Verified in Gambit_fn_02B7CD48 / Gambit_fn_02B7BEE8: ARG7 is the byte
        // length of the extended output buffer (e.g. 1 byte at r1+0x08 right
        // before the caller's sockaddr at r1+0x0C), NOT a count of 16-byte structs.
        const uint32_t bytes = std::min<uint32_t>(ARG7, 256u);
        if (rbrew_memory_accessible(MEM, ARG6, bytes)) std::memset(MEM + ARG6, 0, bytes);
    }
    clear_so_err();
    char addr[48];
    format_sockaddr_str(sa, addr, sizeof(addr));
    const uint8_t b0 = n > 0 ? MEM[ARG1] : 0;
    const uint8_t b1 = n > 1 ? MEM[ARG1 + 1] : 0;
    const uint8_t b2 = n > 2 ? MEM[ARG1 + 2] : 0;
    const uint8_t b3 = n > 3 ? MEM[ARG1 + 3] : 0;
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "src=%s n=%d head=%02x%02x%02x%02x", addr, int(n), b0, b1, b2,
                  b3);
    RET = static_cast<uint32_t>(n);
}

void recvfrom_(CPUState* cpu) {
    recvfrom_impl(cpu, false);
}

void recvfrom_ex(CPUState* cpu) {
    recvfrom_impl(cpu, true);
}

// sendto_multi(fd, buf, len, flags, dst_addrs_array, dst_count)
void sendto_multi(CPUState* cpu) {
    NativeSocket s = INVALID_NATIVE_SOCKET;
    bool nb = false;
    if (!lookup_guest_fd(static_cast<int>(ARG0), &s, &nb)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const uint32_t len = ARG2;
    const uint32_t dst_addrs = ARG4;
    const uint32_t dst_count = ARG5;
    if (!ARG1 || !rbrew_memory_accessible(MEM, ARG1, len)) {
        set_so_err(kSoEfault);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    if (dst_count == 0 || dst_count > 32u || !dst_addrs || !rbrew_memory_accessible(MEM, dst_addrs, dst_count * 16u)) {
        set_so_err(kSoEinval);
        RET = static_cast<uint32_t>(-1);
        return;
    }
    bool dontwait = false;
    const int hflags = translate_msg_flags(ARG3, &dontwait);
    if (dontwait && !nb && !socket_is_writable_now(s)) {
        set_so_err(kSoEwouldblock);
        RET = static_cast<uint32_t>(-1);
        return;
    }

    ssize_t last_ok = -1;
    int sent_count = 0;
    char first_addr[48] = "?";
    for (uint32_t i = 0; i < dst_count; ++i) {
        struct sockaddr_in sa{};
        guest_to_host_sockaddr(MEM, dst_addrs + i * 16u, &sa);
        maybe_redirect_sockaddr(&sa);
        if (i == 0) format_sockaddr_str(sa, first_addr, sizeof(first_addr));
        const ssize_t n = sendto(s, reinterpret_cast<const char*>(MEM + ARG1), static_cast<int>(len), hflags,
                                 reinterpret_cast<struct sockaddr*>(&sa), static_cast<socklen_t>(sizeof(sa)));
        if (n >= 0) {
            last_ok = n;
            ++sent_count;
        } else {
            record_host_error();
        }
    }
    if (sent_count > 0) {
        clear_so_err();
        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "multi(%u) dst0=%s len=%u", dst_count, first_addr, len);
        RET = static_cast<uint32_t>(last_ok);
    } else {
        RET = static_cast<uint32_t>(-1);
    }
}

// ---------------------------------------------------------------------------
// select
// ARG0=nfds, ARG1=readfds*, ARG2=writefds*, ARG3=exceptfds*, ARG4=timeval*
// Wii U timeval: +0 int32 sec (BE), +4 int32 usec (BE)
// ---------------------------------------------------------------------------
void select_(CPUState* cpu) {
    const int nfds = static_cast<int>(ARG0);

    MappedFdSet rfds{}, wfds{}, efds{};
    build_host_fdset(MEM, ARG1, nfds, &rfds);
    build_host_fdset(MEM, ARG2, nfds, &wfds);
    build_host_fdset(MEM, ARG3, nfds, &efds);

    struct timeval tv{};
    struct timeval* tvp = nullptr;
    if (ARG4 && rbrew_memory_accessible(MEM, ARG4, 8u)) {
        tv.tv_sec = static_cast<long>(static_cast<int32_t>(rbrew_read32(MEM, ARG4 + 0u)));
        tv.tv_usec = static_cast<long>(static_cast<int32_t>(rbrew_read32(MEM, ARG4 + 4u)));
        if (tv.tv_sec < 0) tv.tv_sec = 0;
        if (tv.tv_usec < 0) tv.tv_usec = 0;
        tvp = &tv;
    }

    const int total_fds = rfds.count + wfds.count + efds.count;
    if (total_fds == 0) {
        // Winsock select() returns WSAEINVAL if all three fd_sets are empty.
        if (tvp) {
            const int64_t total_us = int64_t(tv.tv_sec) * 1000000LL + int64_t(tv.tv_usec);
            if (total_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(std::min<int64_t>(total_us, 1000000LL)));
            }
        }
        if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) rbrew_write32(MEM, ARG1, 0u);
        if (ARG2 && rbrew_memory_accessible(MEM, ARG2, 4u)) rbrew_write32(MEM, ARG2, 0u);
        if (ARG3 && rbrew_memory_accessible(MEM, ARG3, 4u)) rbrew_write32(MEM, ARG3, 0u);
        clear_so_err();
        RET = 0;
        return;
    }

    NativeSocket max_s = rfds.max_sock;
    if (wfds.max_sock > max_s) max_s = wfds.max_sock;
    if (efds.max_sock > max_s) max_s = efds.max_sock;

#if defined(_WIN32)
    const int host_nfds = 0;
#else
    const int host_nfds = max_s + 1;
#endif

    const int rc = select(host_nfds, rfds.count ? &rfds.set : nullptr, wfds.count ? &wfds.set : nullptr,
                          efds.count ? &efds.set : nullptr, tvp);

    if (rc < 0) {
        record_host_error();
        RET = static_cast<uint32_t>(-1);
        return;
    }

    clear_so_err();
    write_guest_fdset(MEM, ARG1, rfds);
    write_guest_fdset(MEM, ARG2, wfds);
    write_guest_fdset(MEM, ARG3, efds);
    RET = static_cast<uint32_t>(rc);
}

// ---------------------------------------------------------------------------
// setsockopt
// ARG0=fd, ARG1=level, ARG2=optname, ARG3=optval_ptr, ARG4=optlen
// ---------------------------------------------------------------------------
void setsockopt_(CPUState* cpu) {
    const int gfd = static_cast<int>(ARG0);
    NativeSocket s = INVALID_NATIVE_SOCKET;
    if (!lookup_guest_fd(gfd, &s)) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    const int wlevel = static_cast<int>(ARG1);
    const int wopt = static_cast<int>(ARG2);
    const uint32_t vp = ARG3;
    const uint32_t vl = ARG4;

    const int hlevel = map_level(wlevel);
    const bool has_val = (vp != 0u && vl > 0u && rbrew_memory_accessible(MEM, vp, vl));

    if (hlevel == SOL_SOCKET) {
        if (wopt == kWiiSoNbio) {
            // On Wii U nsysnet, SO_NBIO may be called with NULL/0 (meaning enable) or int val.
            const bool enable = has_val ? ((vl >= 4u) ? (rbrew_read32(MEM, vp) != 0u) : (MEM[vp] != 0u)) : true;
            if (set_native_nonblocking(s, enable)) {
                set_guest_fd_nonblocking_flag(gfd, enable);
                clear_so_err();
                std::snprintf(s_trace_detail, sizeof(s_trace_detail), "SO_NBIO=%d", enable ? 1 : 0);
                RET = 0;
            } else {
                record_host_error();
                RET = static_cast<uint32_t>(-1);
            }
            return;
        }
        if (wopt == kWiiSoBio) {
            const bool blocking = has_val ? ((vl >= 4u) ? (rbrew_read32(MEM, vp) != 0u) : (MEM[vp] != 0u)) : true;
            if (set_native_nonblocking(s, !blocking)) {
                set_guest_fd_nonblocking_flag(gfd, !blocking);
                clear_so_err();
                std::snprintf(s_trace_detail, sizeof(s_trace_detail), "SO_BIO=%d", blocking ? 1 : 0);
                RET = 0;
            } else {
                record_host_error();
                RET = static_cast<uint32_t>(-1);
            }
            return;
        }
        // Wii U-specific tuning options with no host equivalent
        if (wopt == kWiiSoTcpSack || wopt == kWiiSoWinScale || wopt >= kWiiSoRusrBuf) {
            clear_so_err();
            RET = 0;
            return;
        }
    }

    if (!has_val) {
        clear_so_err();
        RET = 0;
        return;
    }

    if (hlevel == SOL_SOCKET) {
        if (wopt == kWiiSoLinger && vl >= 8u) {
            struct linger lg{};
            lg.l_onoff = static_cast<decltype(lg.l_onoff)>(rbrew_read32(MEM, vp));
            lg.l_linger = static_cast<decltype(lg.l_linger)>(rbrew_read32(MEM, vp + 4u));
            (void)setsockopt(s, SOL_SOCKET, SO_LINGER, reinterpret_cast<const char*>(&lg),
                             static_cast<socklen_t>(sizeof(lg)));
            clear_so_err();
            RET = 0;
            return;
        }
        if ((wopt == kWiiSoRcvTimeo || wopt == kWiiSoSndTimeo) && vl >= 8u) {
            const int32_t sec = static_cast<int32_t>(rbrew_read32(MEM, vp + 0u));
            const int32_t usec = static_cast<int32_t>(rbrew_read32(MEM, vp + 4u));
            const int hopt = (wopt == kWiiSoRcvTimeo) ? SO_RCVTIMEO : SO_SNDTIMEO;
#if defined(_WIN32)
            const int64_t ms64 = std::max<int64_t>(0, int64_t(sec) * 1000LL + int64_t(usec) / 1000LL);
            DWORD ms = static_cast<DWORD>(std::min<int64_t>(ms64, 0x7FFFFFFFLL));
            (void)setsockopt(s, SOL_SOCKET, hopt, reinterpret_cast<const char*>(&ms), static_cast<socklen_t>(sizeof(ms)));
#else
            struct timeval tv{};
            tv.tv_sec = std::max<int32_t>(0, sec);
            tv.tv_usec = std::max<int32_t>(0, usec);
            (void)setsockopt(s, SOL_SOCKET, hopt, reinterpret_cast<const char*>(&tv), static_cast<socklen_t>(sizeof(tv)));
#endif
            clear_so_err();
            RET = 0;
            return;
        }
    }

    int hopt = wopt;
    if (hlevel == SOL_SOCKET) {
        hopt = map_so_option(wopt);
    } else if (hlevel == IPPROTO_TCP) {
        hopt = map_tcp_option(wopt);
    }

    const int val_host = (vl >= 4u) ? static_cast<int>(rbrew_read32(MEM, vp)) : static_cast<int>(MEM[vp]);
    (void)setsockopt(s, hlevel, hopt, reinterpret_cast<const char*>(&val_host),
                     static_cast<socklen_t>(sizeof(val_host)));
    clear_so_err();
    RET = 0;
}

// ---------------------------------------------------------------------------
// Byte-order functions (identity on big-endian guest)
// ---------------------------------------------------------------------------
void htonl_(CPUState* cpu) {
    RET = ARG0;
}
void htons_(CPUState* cpu) {
    RET = ARG0 & 0xFFFFu;
}
void ntohl_(CPUState* cpu) {
    RET = ARG0;
}
void ntohs_(CPUState* cpu) {
    RET = ARG0 & 0xFFFFu;
}

// ---------------------------------------------------------------------------
// inet_ntoa / inet_pton
// ---------------------------------------------------------------------------
void inet_ntoa_(CPUState* cpu) {
    const uint32_t slot = s_ntoa_slot.fetch_add(1u, std::memory_order_relaxed) & 3u;
    const uint32_t gptr = INET_NTOA_BASE + slot * INET_NTOA_SLOT;

    // PowerPC EABI passes 4-byte `struct in_addr` by pointer in r3
    // (verified in Gambit_fn_02A94BDC: stores sin_addr at r1+8, passes r3=r1+8).
    uint32_t addr_be = ARG0;
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 4u)) {
        addr_be = rbrew_read32(MEM, ARG0);
    }
    const uint8_t b0 = static_cast<uint8_t>(addr_be >> 24);
    const uint8_t b1 = static_cast<uint8_t>(addr_be >> 16);
    const uint8_t b2 = static_cast<uint8_t>(addr_be >> 8);
    const uint8_t b3 = static_cast<uint8_t>(addr_be);

    char buf[20];
    const int len = std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", b0, b1, b2, b3);
    if (len > 0 && rbrew_memory_accessible(MEM, gptr, uint64_t(len) + 1u)) {
        std::memcpy(MEM + gptr, buf, static_cast<size_t>(len) + 1u);
        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "ip=%s", buf);
        RET = gptr;
    } else {
        RET = 0;
    }
}

void inet_pton_(CPUState* cpu) {
    if (ARG0 != 2u || !ARG1 || !ARG2) {
        RET = 0;
        return;
    }
    char buf[64] = {};
    for (int i = 0; i < 63; ++i) {
        if (!rbrew_memory_accessible(MEM, ARG1 + uint32_t(i), 1u)) break;
        buf[i] = static_cast<char>(MEM[ARG1 + uint32_t(i)]);
        if (!buf[i]) break;
    }
    struct in_addr addr{};
    const int rc = inet_pton(AF_INET, buf, &addr);
    if (rc == 1 && rbrew_memory_accessible(MEM, ARG2, 4u)) {
        std::memcpy(MEM + ARG2, &addr.s_addr, 4);
    }
    RET = static_cast<uint32_t>(rc);
}

// ---------------------------------------------------------------------------
// gethostbyname / getaddrinfo / freeaddrinfo
// ---------------------------------------------------------------------------
void gethostbyname_(CPUState* cpu) {
    ensure_wsa_init();
    if (!ARG0) {
        RET = 0;
        return;
    }
    char hostname[256] = {};
    for (int i = 0; i < 255; ++i) {
        if (!rbrew_memory_accessible(MEM, ARG0 + uint32_t(i), 1u)) break;
        hostname[i] = static_cast<char>(MEM[ARG0 + uint32_t(i)]);
        if (!hostname[i]) break;
    }

    uint32_t resolved_nbo = 0;
    std::string canonical = hostname;

    if (should_redirect_hostname(hostname)) {
        const auto& cfg = get_net_config();
        resolved_nbo = cfg.nex_host_ipv4_nbo;
        if (resolved_nbo == 0) resolve_ipv4_nbo(cfg.nex_host.c_str(), &resolved_nbo);
        if (is_nncs2_hostname(hostname)) resolved_nbo = htonl(kNncs2PlaceholderHbo);
    }
    if (resolved_nbo == 0) {
        if (!resolve_ipv4_nbo(hostname, &resolved_nbo)) {
            RBREW_LOG_ERROR("nsysnet", "gethostbyname(\"%s\"): failed\n", hostname);
            std::snprintf(s_trace_detail, sizeof(s_trace_detail), "host=\"%s\" FAIL", hostname);
            RET = 0;
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(s_scratch_mutex);
        size_t nlen = canonical.size();
        if (nlen > 63) nlen = 63;
        std::memcpy(MEM + HE_NAME_STR, canonical.c_str(), nlen);
        MEM[HE_NAME_STR + nlen] = 0;
        std::memcpy(MEM + HE_ADDR, &resolved_nbo, 4);

        rbrew_write32(MEM, HE_ALIASES_ARR, 0u);
        rbrew_write32(MEM, HE_ADDRLIST_ARR + 0u, HE_ADDR);
        rbrew_write32(MEM, HE_ADDRLIST_ARR + 4u, 0u);

        rbrew_write32(MEM, HE_BASE + 0x00u, HE_NAME_STR);
        rbrew_write32(MEM, HE_BASE + 0x04u, HE_ALIASES_ARR);
        rbrew_write32(MEM, HE_BASE + 0x08u, 2u); // AF_INET
        rbrew_write32(MEM, HE_BASE + 0x0Cu, 4u);
        rbrew_write32(MEM, HE_BASE + 0x10u, HE_ADDRLIST_ARR);

        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "host=\"%s\" -> %u.%u.%u.%u", hostname, MEM[HE_ADDR],
                      MEM[HE_ADDR + 1], MEM[HE_ADDR + 2], MEM[HE_ADDR + 3]);
        RBREW_LOG_INFO("nsysnet", "gethostbyname(\"%s\") -> %u.%u.%u.%u\n", hostname, MEM[HE_ADDR], MEM[HE_ADDR + 1],
                       MEM[HE_ADDR + 2], MEM[HE_ADDR + 3]);
    }
    RET = HE_BASE;
}

void getaddrinfo_(CPUState* cpu) {
    ensure_wsa_init();
    if (!ARG0 && !ARG1) {
        RET = static_cast<uint32_t>(-1);
        return;
    }

    char hostname[256] = {};
    if (ARG0) {
        for (int i = 0; i < 255; ++i) {
            if (!rbrew_memory_accessible(MEM, ARG0 + uint32_t(i), 1u)) break;
            hostname[i] = static_cast<char>(MEM[ARG0 + uint32_t(i)]);
            if (!hostname[i]) break;
        }
    }
    char servname[64] = {};
    if (ARG1) {
        for (int i = 0; i < 63; ++i) {
            if (!rbrew_memory_accessible(MEM, ARG1 + uint32_t(i), 1u)) break;
            servname[i] = static_cast<char>(MEM[ARG1 + uint32_t(i)]);
            if (!servname[i]) break;
        }
    }

    int hint_socktype = 0;
    if (ARG2 && rbrew_memory_accessible(MEM, ARG2, 16u)) {
        hint_socktype = static_cast<int>(rbrew_read32(MEM, ARG2 + 0x08u));
    }

    const char* lookup_host = hostname;
    if (should_redirect_hostname(hostname)) {
        lookup_host = get_net_config().nex_host.c_str();
    }

    struct addrinfo hints_h{};
    hints_h.ai_family = AF_INET;
    if (hint_socktype != 0) hints_h.ai_socktype = hint_socktype;

    struct addrinfo* res = nullptr;
    const int rc = getaddrinfo(*lookup_host ? lookup_host : nullptr, *servname ? servname : nullptr, &hints_h, &res);
    if (rc != 0 || !res) {
        std::snprintf(s_trace_detail, sizeof(s_trace_detail), "gai(\"%s\",\"%s\") FAIL rc=%d", hostname, servname, rc);
        RET = static_cast<uint32_t>(rc ? rc : -1);
        return;
    }

    uint32_t count = 0;
    uint32_t prev_guest = 0;
    char first_addr[48] = "?";
    {
        std::lock_guard<std::mutex> lock(s_scratch_mutex);
        for (struct addrinfo* a = res; a && count < AI_MAX_ENTRIES; a = a->ai_next) {
            if (a->ai_family != AF_INET) continue;

            const uint32_t entry_base = AI_BASE + count * AI_ENTRY_STRIDE;
            const uint32_t sa_base = entry_base + 32u;

            auto* sin = reinterpret_cast<struct sockaddr_in*>(a->ai_addr);
            if (is_nncs2_hostname(hostname)) sin->sin_addr.s_addr = htonl(kNncs2PlaceholderHbo);
            if (count == 0) format_sockaddr_str(*sin, first_addr, sizeof(first_addr));
            host_to_guest_sockaddr(MEM, sa_base, sin);

            if (a->ai_canonname) {
                size_t cnlen = std::strlen(a->ai_canonname);
                if (cnlen > 63) cnlen = 63;
                std::memcpy(MEM + AI_CANON_STR, a->ai_canonname, cnlen);
                MEM[AI_CANON_STR + cnlen] = 0;
            } else {
                MEM[AI_CANON_STR] = 0;
            }

            rbrew_write32(MEM, entry_base + 0x00u, 0u);
            rbrew_write32(MEM, entry_base + 0x04u, 2u); // AF_INET
            rbrew_write32(MEM, entry_base + 0x08u, static_cast<uint32_t>(a->ai_socktype));
            rbrew_write32(MEM, entry_base + 0x0Cu, static_cast<uint32_t>(a->ai_protocol));
            rbrew_write32(MEM, entry_base + 0x10u, 16u);
            rbrew_write32(MEM, entry_base + 0x14u, AI_CANON_STR);
            rbrew_write32(MEM, entry_base + 0x18u, sa_base);
            rbrew_write32(MEM, entry_base + 0x1Cu, 0u);

            if (prev_guest) rbrew_write32(MEM, prev_guest + 0x1Cu, entry_base);
            prev_guest = entry_base;
            ++count;
        }
    }
    freeaddrinfo(res);

    if (count == 0) {
        RET = static_cast<uint32_t>(-1);
        return;
    }
    if (ARG3 && rbrew_memory_accessible(MEM, ARG3, 4u)) rbrew_write32(MEM, ARG3, AI_BASE);

    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "gai(\"%s\",\"%s\") -> %s (%u)", hostname, servname,
                  first_addr, count);
    RET = 0;
}

void freeaddrinfo_(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// nn_ac (network availability)
// Offline by default (fast boot without waiting for Nintendo servers).
// Online when RBREW_NEX_HOST, RBREW_ONLINE=1, or RBREW_AC_CONNECT=ok is set.
// ---------------------------------------------------------------------------
constexpr uint32_t kAcResultNotConnected = 0xA0D1F400u;

void ac_Initialize(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}

void ac_Connect(CPUState* cpu) {
    const bool ok = get_net_config().online_enabled;
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "online=%d", ok ? 1 : 0);
    RET = ok ? 0u : kAcResultNotConnected;
}

void ac_Close(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}

void ac_GetLastErrorCode(CPUState* cpu) {
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 4u)) rbrew_write32(MEM, ARG0, 0u);
    RET = 0;
}

void ac_GetAssignedAddress(CPUState* cpu) {
    const auto& cfg = get_net_config();
    uint32_t addr_host_be = 0u;
    if (cfg.online_enabled && cfg.local_ipv4_nbo != 0) {
        // rbrew_write32 writes big-endian, so pass host-order IPv4 uint32 so bytes in guest RAM are network order.
        addr_host_be = ntohl(cfg.local_ipv4_nbo);
    }
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 4u)) rbrew_write32(MEM, ARG0, addr_host_be);
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "ip=%u.%u.%u.%u", (addr_host_be >> 24) & 0xFFu,
                  (addr_host_be >> 16) & 0xFFu, (addr_host_be >> 8) & 0xFFu, addr_host_be & 0xFFu);
    RET = 0;
}

void ac_GetAssignedSubnet(CPUState* cpu) {
    const uint32_t mask = get_net_config().online_enabled ? 0xFFFFFF00u : 0u;
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 4u)) rbrew_write32(MEM, ARG0, mask);
    RET = 0;
}

void ac_IsApplicationConnected(CPUState* cpu) {
    const bool ok = get_net_config().online_enabled;
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 1u)) MEM[ARG0] = ok ? 1u : 0u;
    std::snprintf(s_trace_detail, sizeof(s_trace_detail), "connected=%d", ok ? 1 : 0);
    RET = 0;
}

// ---------------------------------------------------------------------------
// nlibcurl lifecycle
// ---------------------------------------------------------------------------
void curl_global_init_mem_(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}
void curl_global_cleanup_(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// RBREW_NET_TRACE=<file>: trace every socket/nn_ac/nn_act/nn_fp call with args,
// result, mapped Wii U SO_* error, raw host error, and packet/address details.
// ---------------------------------------------------------------------------
FILE* get_net_trace_file() {
    static FILE* const file = [] {
        const char* p = std::getenv("RBREW_NET_TRACE");
        return p && *p ? std::fopen(p, "w") : nullptr;
    }();
    return file;
}

std::mutex& get_net_trace_mutex() {
    static std::mutex lock;
    return lock;
}

double net_trace_elapsed_sec() {
    static const auto start = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void net_trace(const char* name, CPUState* cpu, uint32_t a0, uint32_t a1, uint32_t a2) {
    FILE* const file = get_net_trace_file();
    if (!file) {
        s_trace_detail[0] = '\0';
        return;
    }
    std::lock_guard<std::mutex> guard(get_net_trace_mutex());
    if (s_trace_detail[0]) {
        std::fprintf(file, "%9.3f %-24s %08X %08X %08X -> %08X so_err=%d host_err=%d [%s]\n", net_trace_elapsed_sec(),
                     name, a0, a1, a2, RET, s_last_so_err, s_last_host_err, s_trace_detail);
        s_trace_detail[0] = '\0';
    } else {
        std::fprintf(file, "%9.3f %-24s %08X %08X %08X -> %08X so_err=%d host_err=%d\n", net_trace_elapsed_sec(), name,
                     a0, a1, a2, RET, s_last_so_err, s_last_host_err);
    }
    std::fflush(file);
}

} // namespace

void rbrew_net_trace_log(const char* tag, const char* fmt, ...) {
    FILE* const file = get_net_trace_file();
    if (!file) return;
    std::lock_guard<std::mutex> guard(get_net_trace_mutex());
    std::fprintf(file, "%9.3f [%s] ", net_trace_elapsed_sec(), tag ? tag : "net");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(file, fmt, ap);
    va_end(ap);
    std::fputc('\n', file);
    std::fflush(file);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
#include "../stubs/stubs_addrs.h"

#define NET_REG(addr, fn)                                                                                              \
    rbrew_register_func(cpu, addr, [](CPUState* cpu) {                                                                 \
        const uint32_t a0 = ARG0, a1 = ARG1, a2 = ARG2;                                                                \
        s_trace_detail[0] = '\0';                                                                                      \
        fn(cpu);                                                                                                       \
        net_trace(#fn, cpu, a0, a1, a2);                                                                               \
    })

void nsysnet_register(CPUState* cpu) {
    // nsysnet core
    NET_REG(ADDR_socket_lib_init, socket_lib_init);
    NET_REG(ADDR_NSSLInit, NSSLInit);
    NET_REG(ADDR_set_resolver_allocator, set_resolver_allocator);
    NET_REG(ADDR_mw_socket, mw_socket);
    NET_REG(ADDR_socketclose, socketclose);
    NET_REG(ADDR_socketlasterr, socketlasterr);
    NET_REG(ADDR_bind, bind_);
    NET_REG(ADDR_shutdown, shutdown_);
    NET_REG(ADDR_getsockname, getsockname_);
    NET_REG(ADDR_send, send_);
    NET_REG(ADDR_recv, recv_);
    NET_REG(ADDR_sendto, sendto_);
    NET_REG(ADDR_recvfrom, recvfrom_);
    NET_REG(ADDR_recvfrom_ex, recvfrom_ex);
    NET_REG(ADDR_sendto_multi, sendto_multi);
    NET_REG(ADDR_select, select_);
    NET_REG(ADDR_setsockopt, setsockopt_);
    NET_REG(ADDR_htonl, htonl_);
    NET_REG(ADDR_htons, htons_);
    NET_REG(ADDR_ntohl, ntohl_);
    NET_REG(ADDR_ntohs, ntohs_);
    NET_REG(ADDR_inet_ntoa, inet_ntoa_);
    NET_REG(ADDR_inet_pton, inet_pton_);
    NET_REG(ADDR_gethostbyname, gethostbyname_);
    NET_REG(ADDR_getaddrinfo, getaddrinfo_);
    NET_REG(ADDR_freeaddrinfo, freeaddrinfo_);

    // nn_ac
    NET_REG(ADDR_Initialize__Q2_2nn2acFv, ac_Initialize);
    NET_REG(ADDR_Connect__Q2_2nn2acFv, ac_Connect);
    NET_REG(ADDR_Close__Q2_2nn2acFv, ac_Close);
    NET_REG(ADDR_GetLastErrorCode__Q2_2nn2acFPUi, ac_GetLastErrorCode);
    NET_REG(ADDR_GetAssignedAddress__Q2_2nn2acFPUl, ac_GetAssignedAddress);
    NET_REG(ADDR_GetAssignedSubnet__Q2_2nn2acFPUl, ac_GetAssignedSubnet);
    NET_REG(ADDR_IsApplicationConnected__Q2_2nn2acFPb, ac_IsApplicationConnected);

    // nlibcurl lifecycle
    NET_REG(ADDR_curl_global_init_mem, curl_global_init_mem_);
    NET_REG(ADDR_curl_global_cleanup, curl_global_cleanup_);
}
