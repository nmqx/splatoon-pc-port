#include "nn_act.h"
#include "../stubs/stubs_addrs.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

extern void rbrew_net_trace_log(const char* tag, const char* fmt, ...);

// Evidence: Gambit FUN_027D626C (account/Mii owner, 0x9A4 bytes) calls
// Initialize (sign bit -> assert line 0x3D), GetSlotNo, IsNetworkAccount,
// GetMiiName, GetMii, then FUN_027F147C -> FFL FUN_02C6DB64(out, 4, store, 0, 0)
// whose nonzero result asserts at line 0x59. It then reads GetPrincipalId,
// GetCountry, GetSimpleAddressId and scans slots 1..12 with IsSlotOccupied and
// GetMiiEx.
//
// In offline mode (default when no RBREW_NEX_HOST / RBREW_ONLINE /
// RBREW_PRINCIPAL_ID is set), semantics mirror Cemu's default offline account.
// In online mode (RBREW_NEX_HOST set), provides a network account with a
// configurable per-instance Principal ID, NNID, and local NEX service token
// pointing at RBREW_NEX_HOST:RBREW_NEX_PORT (verified ACTNexAuthenticationResult
// layout from Gambit_fn_02B30748: 0x25C bytes).

namespace {
std::atomic<uint32_t> g_initialize_count{0};
constexpr std::u16string_view kMiiName = u"default";
constexpr char kOfflineCountry[] = "NN";
constexpr char kOnlineCountry[] = "US";
constexpr uint32_t kOnlineSimpleAddressId = 0x00310000u; // US
constexpr uint16_t kDefaultNexPort = 60000u;

std::mutex g_settings_mutex;
bool g_settings_loaded = false;
std::unordered_map<std::string, std::string> g_settings_cache;

std::string read_settings_value(const char* key) {
    std::lock_guard<std::mutex> lock(g_settings_mutex);
    if (!g_settings_loaded) {
        g_settings_loaded = true;
        std::ifstream in("settings.ini");
        if (in) {
            for (std::string line; std::getline(in, line);) {
                const size_t eq = line.find('=');
                if (eq == std::string::npos || eq == 0) continue;
                std::string k = line.substr(0, eq);
                std::string val = line.substr(eq + 1);
                while (!val.empty() && (val.back() == '\r' || val.back() == ' ' || val.back() == '\t')) val.pop_back();
                g_settings_cache[k] = std::move(val);
            }
        }
    }
    auto it = g_settings_cache.find(key);
    return (it != g_settings_cache.end()) ? it->second : std::string{};
}

bool is_online_account_enabled() {
    const char* nex_host = std::getenv("RBREW_NEX_HOST");
    if (nex_host && *nex_host) return true;
    const char* online = std::getenv("RBREW_ONLINE");
    if (online && (std::strcmp(online, "1") == 0 || std::strcmp(online, "true") == 0)) return true;
    const char* pid = std::getenv("RBREW_PRINCIPAL_ID");
    if (pid && *pid) return true;
    return false;
}

std::string get_active_player_name_ascii() {
    const char* env_player = std::getenv("RBREW_PLAYER_NAME");
    if (!env_player || !*env_player) env_player = std::getenv("RBREW_MII_NAME");
    if (env_player && *env_player) return std::string(env_player);
    std::string ini_name = read_settings_value("player_name");
    if (!ini_name.empty()) return ini_name;
    return "default";
}

std::u16string get_active_mii_name() {
    const std::string candidate = get_active_player_name_ascii();
    if (!candidate.empty()) {
        std::u16string name;
        for (char c : candidate) {
            if (name.size() >= RBREW_ACT_MII_NAME_MAX) break;
            name.push_back(static_cast<char16_t>(static_cast<unsigned char>(c)));
        }
        if (!name.empty()) return name;
    }
    return std::u16string(kMiiName);
}

uint32_t parse_u32_auto(const std::string& s) {
    if (s.empty()) return 0u;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 0);
    if (end == s.c_str()) return 0u;
    return static_cast<uint32_t>(v);
}

uint32_t fnv1a_32(std::string_view s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

uint32_t get_active_principal_id() {
    const char* env_pid = std::getenv("RBREW_PRINCIPAL_ID");
    if (env_pid && *env_pid) {
        const uint32_t parsed = parse_u32_auto(env_pid);
        if (parsed != 0) return parsed;
    }
    const std::string ini_pid = read_settings_value("principal_id");
    if (!ini_pid.empty()) {
        const uint32_t parsed = parse_u32_auto(ini_pid);
        if (parsed != 0) return parsed;
    }
    if (!is_online_account_enabled()) return 0u;
    // Derive a deterministic, non-zero Principal ID from player name so two
    // instances with distinct RBREW_PLAYER_NAME values never collide on PID.
    const std::string name = get_active_player_name_ascii();
    return 10000001u + (fnv1a_32(name) % 9000000u);
}

std::string get_active_account_id() {
    if (!is_online_account_enabled()) return {};
    const char* env_id = std::getenv("RBREW_ACCOUNT_ID");
    std::string raw = (env_id && *env_id) ? std::string(env_id) : get_active_player_name_ascii();
    std::string out;
    for (char c : raw) {
        if (out.size() >= 16) break;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
            c == '.') {
            out.push_back(c);
        }
    }
    if (out.empty()) out = "Player" + std::to_string(get_active_principal_id());
    if (out.size() > 16) out.resize(16);
    return out;
}

std::string resolve_nex_host_ipv4(const char* host) {
    if (!host || !*host) return "127.0.0.1";
    struct in_addr ia{};
    if (inet_pton(AF_INET, host, &ia) == 1) {
        char buf[16] = {};
        inet_ntop(AF_INET, &ia, buf, sizeof(buf));
        return std::string(buf);
    }
#if defined(_WIN32)
    static std::once_flag wsa_once;
    std::call_once(wsa_once, [] {
        WSADATA wd{};
        WSAStartup(MAKEWORD(2, 2), &wd);
    });
#endif
    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &res) == 0 && res) {
        auto* sin = reinterpret_cast<struct sockaddr_in*>(res->ai_addr);
        char buf[16] = {};
        inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
        freeaddrinfo(res);
        if (buf[0]) return std::string(buf);
    }
    return "127.0.0.1";
}

void write_cstr_padded(uint8_t* mem, uint32_t addr, size_t max_bytes, std::string_view text) {
    const size_t n = (text.size() < max_bytes - 1) ? text.size() : (max_bytes - 1);
    for (size_t i = 0; i < n; ++i) rbrew_write8(mem, addr + uint32_t(i), static_cast<uint8_t>(text[i]));
    rbrew_write8(mem, addr + uint32_t(n), 0u);
}

void require_guest(const CPUState* cpu, uint32_t address, uint64_t length, const char* what) {
    if (!address || uint64_t(address) + length > cpu->guest_memory_size ||
        !rbrew_memory_accessible(cpu->mem, address, length))
        throw std::runtime_error(std::string("nn::act ") + what + " is outside accessible guest memory");
}

// Account slots are 1..12; 0xFE names the current account, 0xFF common.
bool slot_occupied(uint32_t raw) {
    const uint8_t slot = uint8_t(raw);
    if (slot == RBREW_ACT_SLOT_CURRENT || slot == 0xFFu || slot == 1) return true;
    if (slot <= RBREW_ACT_SLOT_COUNT) return false;
    throw std::runtime_error("nn::act slot number is outside the proven range");
}

void write_mii(CPUState* cpu, uint32_t address) {
    require_guest(cpu, address, 0x60, "FFLStoreData output");
    const auto name = get_active_mii_name();
    const auto mii = rbrew_act_default_mii(name);
    for (uint32_t i = 0; i < mii.size(); ++i) rbrew_write8(cpu->mem, address + i, mii[i]);
}

void Initialize(CPUState* cpu) {
    ++g_initialize_count;
    cpu->r[3] = 0;
}
void Finalize(CPUState* cpu) {
    uint32_t count = g_initialize_count.load();
    while (count && !g_initialize_count.compare_exchange_weak(count, count - 1)) {}
    cpu->r[3] = 0;
}
void GetSlotNo(CPUState* cpu) {
    cpu->r[3] = 1;
}
void IsSlotOccupied(CPUState* cpu) {
    cpu->r[3] = slot_occupied(cpu->r[3]) ? 1 : 0;
}
void IsNetworkAccount(CPUState* cpu) {
    cpu->r[3] = is_online_account_enabled() ? 1u : 0u;
}
// Cemu returns 1; with 0 Splatoon attempts nn_fp.RegisterAccount.
void HasNfsAccount(CPUState* cpu) {
    cpu->r[3] = 1;
}
void GetPrincipalId(CPUState* cpu) {
    cpu->r[3] = get_active_principal_id();
}
void GetSimpleAddressId(CPUState* cpu) {
    cpu->r[3] = is_online_account_enabled() ? kOnlineSimpleAddressId : 0u;
}
void GetMii(CPUState* cpu) {
    write_mii(cpu, cpu->r[3]);
    cpu->r[3] = 0;
}
void GetMiiEx(CPUState* cpu) {
    if (!slot_occupied(cpu->r[4])) {
        cpu->r[3] = RBREW_ACT_RESULT_ACCOUNT_DOES_NOT_EXIST;
        return;
    }
    write_mii(cpu, cpu->r[3]);
    cpu->r[3] = 0;
}
void GetParentalControlSlotNoEx(CPUState* cpu) {
    if (!slot_occupied(cpu->r[4])) {
        cpu->r[3] = RBREW_ACT_RESULT_ACCOUNT_DOES_NOT_EXIST;
        return;
    }
    require_guest(cpu, cpu->r[3], 1, "parental slot output");
    rbrew_write8(cpu->mem, cpu->r[3], 1);
    cpu->r[3] = 0;
}
void GetMiiName(CPUState* cpu) {
    const uint32_t out = cpu->r[3];
    const auto name = get_active_mii_name();
    require_guest(cpu, out, (name.size() + 1) * 2, "Mii name output");
    for (size_t i = 0; i < name.size(); ++i) rbrew_write16(cpu->mem, out + uint32_t(i * 2), name[i]);
    rbrew_write16(cpu->mem, out + uint32_t(name.size() * 2), 0);
    cpu->r[3] = 0;
}
void GetCountry(CPUState* cpu) {
    const uint32_t out = cpu->r[3];
    char country[3] = {};
    if (is_online_account_enabled()) {
        const char* env_c = std::getenv("RBREW_COUNTRY");
        if (env_c && std::strlen(env_c) >= 2) {
            country[0] = env_c[0];
            country[1] = env_c[1];
        } else {
            country[0] = kOnlineCountry[0];
            country[1] = kOnlineCountry[1];
        }
    } else {
        country[0] = kOfflineCountry[0];
        country[1] = kOfflineCountry[1];
    }
    country[2] = '\0';
    require_guest(cpu, out, 3, "country output");
    for (uint32_t i = 0; i < 3; ++i) rbrew_write8(cpu->mem, out + i, uint8_t(country[i]));
    cpu->r[3] = 0;
}
void GetAccountId(CPUState* cpu) {
    const std::string acc = get_active_account_id();
    require_guest(cpu, cpu->r[3], acc.size() + 1, "account id output");
    for (size_t i = 0; i < acc.size(); ++i) rbrew_write8(cpu->mem, cpu->r[3] + uint32_t(i), uint8_t(acc[i]));
    rbrew_write8(cpu->mem, cpu->r[3] + uint32_t(acc.size()), 0u);
    cpu->r[3] = 0;
}

// GetErrorCode(const nn::Result& res) -> uint32_t errorCode
void GetErrorCode(CPUState* cpu) {
    uint32_t res_val = 0;
    if (cpu->r[3] && rbrew_memory_accessible(cpu->mem, cpu->r[3], 4u)) {
        res_val = rbrew_read32(cpu->mem, cpu->r[3]);
    }
    cpu->r[3] = (res_val & 0x80000000u) ? 1022000u : 0u;
}

// GetHostServerSettings(char* outHost, char* outEnv, uint8_t slot) -> nn::Result
void GetHostServerSettings(CPUState* cpu) {
    if (cpu->r[3] && rbrew_memory_accessible(cpu->mem, cpu->r[3], 4u)) write_cstr_padded(cpu->mem, cpu->r[3], 4, "L1");
    if (cpu->r[4] && rbrew_memory_accessible(cpu->mem, cpu->r[4], 8u)) write_cstr_padded(cpu->mem, cpu->r[4], 8, "prod");
    cpu->r[3] = 0;
}

// GetTransferableIdEx(uint64_t* outId, uint32_t uniqueId, uint8_t slot) -> nn::Result
void GetTransferableIdEx(CPUState* cpu) {
    if (!slot_occupied(cpu->r[5])) {
        cpu->r[3] = RBREW_ACT_RESULT_ACCOUNT_DOES_NOT_EXIST;
        return;
    }
    const uint32_t out = cpu->r[3];
    require_guest(cpu, out, 8u, "transferable id output");
    const uint32_t pid = get_active_principal_id();
    const uint32_t unique_id = cpu->r[4];
    const uint64_t tid = (uint64_t(pid ? pid : 1u) << 32) | uint64_t(unique_id ^ 0x53504C31u);
    rbrew_write32(cpu->mem, out + 0u, uint32_t(tid >> 32));
    rbrew_write32(cpu->mem, out + 4u, uint32_t(tid & 0xFFFFFFFFu));
    cpu->r[3] = 0;
}

// GetUtcOffset() -> int64_t (in r3:r4)
void GetUtcOffset(CPUState* cpu) {
    cpu->r[3] = 0;
    cpu->r[4] = 0;
}

// GetUuid(ACTUuid* out) -> nn::Result (16 bytes)
void GetUuid(CPUState* cpu) {
    const uint32_t out = cpu->r[3];
    require_guest(cpu, out, 16u, "UUID output");
    const uint32_t pid = get_active_principal_id();
    const uint32_t h1 = fnv1a_32(get_active_player_name_ascii());
    const uint32_t h2 = pid ^ 0xA55A5AA5u;
    rbrew_write32(cpu->mem, out + 0u, h1);
    rbrew_write32(cpu->mem, out + 4u, (h2 & 0xFFFF0FFFu) | 0x00004000u); // UUID v4
    rbrew_write32(cpu->mem, out + 8u, ((pid + 0x10176900u) & 0x3FFFFFFFu) | 0x80000000u);
    rbrew_write32(cpu->mem, out + 12u, h1 ^ h2 ^ 0x52425257u);
    cpu->r[3] = 0;
}

// AcquireNexServiceToken(ACTNexAuthenticationResult* out, uint32_t gameServerId)
// Verified layout (Gambit_fn_02B30748, 0x25C bytes):
//   +0x000: char nexToken[513]    (padded to 0x204)
//   +0x204: char nexPassword[65]  (padded to 0x248)
//   +0x248: char host[16]         (IPv4 string, e.g. "127.0.0.1")
//   +0x258: uint16_t port         (big-endian)
void AcquireNexServiceToken(CPUState* cpu) {
    const char* env_token = std::getenv("RBREW_NEX_TOKEN");
    if (env_token && std::string_view(env_token) == "stub") {
        // Legacy test hook: leave output untouched and return caller's r3.
        return;
    }

    const char* nex_host = std::getenv("RBREW_NEX_HOST");
    if (!nex_host || !*nex_host) {
        cpu->r[3] = RBREW_ACT_RESULT_NEX_UNAVAILABLE;
        return;
    }

    const uint32_t out = cpu->r[3];
    const uint32_t game_server_id = cpu->r[4];
    require_guest(cpu, out, 0x25Cu, "ACTNexAuthenticationResult output");
    std::memset(cpu->mem + out, 0, 0x25Cu);

    const uint32_t pid = get_active_principal_id();
    const std::string host_ip = resolve_nex_host_ipv4(nex_host);

    uint16_t port = kDefaultNexPort;
    if (const char* env_port = std::getenv("RBREW_NEX_PORT"); env_port && *env_port) {
        const long p = std::strtol(env_port, nullptr, 10);
        if (p > 0 && p <= 65535) port = static_cast<uint16_t>(p);
    }

    std::string token;
    if (env_token && *env_token) {
        token = env_token;
    } else {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "RBREW_TOKEN_%u_%08X", pid, game_server_id);
        token = buf;
    }

    std::string password = "rbrew_nex_pass";
    if (const char* env_pass = std::getenv("RBREW_NEX_PASSWORD"); env_pass && *env_pass) {
        password = env_pass;
    }

    write_cstr_padded(cpu->mem, out + 0x000u, 513u, token);
    write_cstr_padded(cpu->mem, out + 0x204u, 65u, password);
    write_cstr_padded(cpu->mem, out + 0x248u, 16u, host_ip);
    rbrew_write16(cpu->mem, out + 0x258u, port);
    rbrew_write16(cpu->mem, out + 0x25Au, 0u);

    rbrew_net_trace_log("nn_act", "AcquireNexServiceToken(gsid=%08X) pid=%u -> %s:%u", game_server_id, pid,
                        host_ip.c_str(), unsigned(port));

    cpu->r[3] = 0; // nn::Result success
}
} // namespace

uint16_t rbrew_ffl_crc16(const uint8_t* data, size_t length) {
    uint16_t crc = 0;
    for (size_t c = 0; c < length; ++c) {
        for (int bit = 0; bit < 8; ++bit) crc = (crc & 0x8000) ? uint16_t((crc << 1) ^ 0x1021) : uint16_t(crc << 1);
        crc ^= data[c];
    }
    return crc;
}

RbrewFflStoreData rbrew_act_default_mii(std::u16string_view name) {
    if (name.empty() || name.size() > RBREW_ACT_MII_NAME_MAX)
        throw std::invalid_argument("Mii name must be 1..10 UTF-16 units");
    RbrewFflStoreData mii{};
    auto put_le16 = [&](size_t at, uint16_t value) {
        mii[at] = uint8_t(value);
        mii[at + 1] = uint8_t(value >> 8);
    };
    for (size_t at = 0; at < mii.size(); at += 2) put_le16(at, 1);
    put_le16(0x02, 1 | (1 << 12));
    put_le16(0x3A, 1 | (3 << 9));
    mii[0x08] = 0x03;
    mii[0x09] = 0x00;
    mii[0x0A] = 0x34;
    mii[0x0B] = 0x33;
    for (size_t i = 0; i < RBREW_ACT_MII_NAME_MAX; ++i) put_le16(0x1A + i * 2, i < name.size() ? uint16_t(name[i]) : 0);
    for (uint32_t counter = 0;; ++counter) {
        mii[4] = uint8_t(counter >> 24);
        mii[5] = uint8_t(counter >> 16);
        mii[6] = uint8_t(counter >> 8);
        mii[7] = uint8_t(counter);
        if (rbrew_ffl_crc16(mii.data(), mii.size()) == 0) return mii;
        if (counter == UINT32_MAX) throw std::runtime_error("No CRC-valid default Mii exists");
    }
}

uint32_t rbrew_act_get_active_principal_id() {
    return get_active_principal_id();
}

void rbrew_act_reset_for_tests() {
    g_initialize_count = 0;
    std::lock_guard<std::mutex> lock(g_settings_mutex);
    g_settings_loaded = false;
    g_settings_cache.clear();
}

#define ACT_REG(fn, mangled) rbrew_register_func(cpu, ADDR_##mangled, fn)
void nn_act_register(CPUState* cpu) {
    ACT_REG(Initialize, Initialize__Q2_2nn3actFv);
    ACT_REG(Finalize, Finalize__Q2_2nn3actFv);
    ACT_REG(GetSlotNo, GetSlotNo__Q2_2nn3actFv);
    ACT_REG(IsSlotOccupied, IsSlotOccupied__Q2_2nn3actFUc);
    ACT_REG(IsNetworkAccount, IsNetworkAccount__Q2_2nn3actFv);
    ACT_REG(HasNfsAccount, HasNfsAccount__Q2_2nn3actFv);
    ACT_REG(GetPrincipalId, GetPrincipalId__Q2_2nn3actFv);
    ACT_REG(GetSimpleAddressId, GetSimpleAddressId__Q2_2nn3actFv);
    ACT_REG(GetMii, GetMii__Q2_2nn3actFP12FFLStoreData);
    ACT_REG(GetMiiEx, GetMiiEx__Q2_2nn3actFP12FFLStoreDataUc);
    ACT_REG(GetMiiName, GetMiiName__Q2_2nn3actFPw);
    ACT_REG(GetCountry, GetCountry__Q2_2nn3actFPc);
    ACT_REG(GetAccountId, GetAccountId__Q2_2nn3actFPc);
    ACT_REG(GetParentalControlSlotNoEx, GetParentalControlSlotNoEx__Q2_2nn3actFPUcUc);
    ACT_REG(AcquireNexServiceToken, AcquireNexServiceToken__Q2_2nn3actFP26ACTNexAuthenticationResultUi);
    ACT_REG(GetErrorCode, GetErrorCode__Q2_2nn3actFRCQ2_2nn6Result);
    ACT_REG(GetHostServerSettings, GetHostServerSettings__Q2_2nn3actFPcT1Uc);
    ACT_REG(GetTransferableIdEx, GetTransferableIdEx__Q2_2nn3actFPULUiUc);
    ACT_REG(GetUtcOffset, GetUtcOffset__Q2_2nn3actFv);
    ACT_REG(GetUuid, GetUuid__Q2_2nn3actFP7ACTUuid);
}
#undef ACT_REG
