// Catch-all stubs for remaining Wii U OS modules:
// nn_save, nn_olv, nn_boss, nn_fp, proc_ui, sysapp,
// zlib125, mic, nn_acp, nn_ndm
//
// nn_save file operations use the shared coreinit FS provider under ./save/.

#include "../os_common.h"
#include "../coreinit/coreinit_fs.h"
#include "../nn_act/nn_act.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

extern uint32_t rbrew_act_get_active_principal_id();
extern void rbrew_net_trace_log(const char* tag, const char* fmt, ...);

constexpr uint32_t SAVE_OK = 0;

static void SAVEInit(CPUState* cpu) {
    coreinit_save_init(cpu);
}
static void SAVEInitSaveDir(CPUState* cpu) {
    coreinit_save_init_dir(cpu);
}
static void SAVEOpenFile(CPUState* cpu) {
    coreinit_save_open_file(cpu);
}
static void SAVEFlushQuota(CPUState* cpu) {
    coreinit_save_flush_quota(cpu);
}
static void SAVEMakeDir(CPUState* cpu) {
    coreinit_save_make_dir(cpu);
}
static void SAVERemove(CPUState* cpu) {
    coreinit_save_remove(cpu);
}
static void SAVEGetStat(CPUState* cpu) {
    coreinit_save_get_stat(cpu);
}
static void SAVEOpenDir(CPUState* cpu) {
    RET = uint32_t(-6);
}

// SAVEGetShared*Path(u64 titleId in r3:r4, const char* name r5, char* out r6,
// u32 outLength r7). Formats match Cemu nn_save; the shared volume is served
// by FS from RBREW_MLC_DIR.
static void save_shared_path(CPUState* cpu, const char* format) {
    const uint32_t name = ARG2, out = ARG3, capacity = ARG4;
    std::string file;
    for (uint32_t at = name;; ++at) {
        if (!name || !rbrew_memory_accessible(MEM, at, 1))
            throw std::runtime_error("SAVE shared data name is outside accessible guest memory");
        const uint8_t c = rbrew_read8(MEM, at);
        if (!c) break;
        if (file.size() >= 0x200) throw std::runtime_error("SAVE shared data name is unterminated");
        file.push_back(char(c));
    }
    char path[0x300];
    const int written = snprintf(path, sizeof(path), format, ARG0, ARG1, file.c_str());
    if (written < 0 || uint32_t(written) >= capacity || size_t(written) >= sizeof(path))
        throw std::runtime_error("SAVE shared path does not fit the guest output buffer");
    if (!out || !rbrew_memory_accessible(MEM, out, uint64_t(written) + 1))
        throw std::runtime_error("SAVE shared path output is outside accessible guest memory");
    for (int i = 0; i <= written; ++i) rbrew_write8(MEM, out + uint32_t(i), uint8_t(path[i]));
    RET = SAVE_OK;
}
static void SAVEGetSharedDataTitlePath(CPUState* cpu) {
    save_shared_path(cpu, "/vol/storage_mlc01/sys/title/%08x/%08x/content/%s");
}
static void SAVEGetSharedSaveDataPath(CPUState* cpu) {
    save_shared_path(cpu, "/vol/storage_mlc01/usr/save/%08x/%08x/user/common/%s");
}

// ---------------------------------------------------------------------------
// nn_boss — SpotPass (safe local stubs; never return raw `this` as bool/size)
// ---------------------------------------------------------------------------
static void BOSSReturnZero(CPUState* cpu) {
    RET = 0;
}
static void BOSSReturnThis(CPUState* cpu) {
    RET = ARG0;
}
static void BOSSExistFalse(CPUState* cpu) {
    RET = 0;
}
static void BOSSWriteOutZeroU32(CPUState* cpu) {
    // Methods like Task::GetTurnState(u32* out), GetResult(u32* out) -> r3=this, r4=out*
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) rbrew_write32(MEM, ARG1, 0u);
    RET = 0;
}
static void BOSSWriteOutZeroU64(CPUState* cpu) {
    // Task::GetContentLength(u32* out), GetProcessedLength(u32* out) return int64_t in r3:r4.
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) rbrew_write32(MEM, ARG1, 0u);
    cpu->r[3] = 0;
    cpu->r[4] = 0;
}
static void BOSSGetHttpStatusCode(CPUState* cpu) {
    // Task::GetHttpStatusCode(u32* out) returns HTTP status code in r3 (out may be nullptr).
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) rbrew_write32(MEM, ARG1, 0u);
    RET = 200u;
}
// ---------------------------------------------------------------------------
// Offline BOSS content. Splatoon refuses online play unless the "News" download
// (task schdat2 -> VSSetting.byaml, the stage schedule) succeeds. We serve an
// embedded schedule (multiplayer/make_vssetting.py) and report every task as
// finished OK. Other storages (optdata = Splatfest) report "no storage", which
// the game handles as "nothing to read" (Gambit_fn_02793EE8 -> L_027942A8).
//
// Guest object layouts follow nn_boss (same as Cemu):
//   Storage (0x28): +0x04 kind, +0x0B name[8]
//   NsData  (0x58): +0x00 name[32], +0x20 Storage copy, +0x48 u64 seek
// ---------------------------------------------------------------------------
#if RBREW_GAME_REVISION >= 272
#include "boss_vssetting_v3.h" // v272 parser: Version <= 3, AfterFesBonusStart required
#else
#include "boss_vssetting.h" // v16 parser (02184868): Version == 0
#endif

// VSSetting with its DateTime set to the start of the current 4 hour UTC slot,
// so the schedule is always "the latest" (the game only plays the Inkopolis
// News, and only lets you into the lobby, for a fresh schedule).
static uint8_t g_boss_vssetting[sizeof(kBossVSSetting)];
static const uint8_t* boss_vssetting_now() {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    memcpy(g_boss_vssetting, kBossVSSetting, sizeof(kBossVSSetting));
    const time_t now = time(nullptr);
    // Start 24 h back (6 phases) so the current phase exists even if the game reads the time in another zone.
    const time_t slot = now - (now % (4 * 3600)) - 24 * 3600;
    struct tm t {};
#if defined(_WIN32)
    gmtime_s(&t, &slot);
#else
    gmtime_r(&slot, &t);
#endif
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d+00:00", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
             t.tm_hour, t.tm_min, t.tm_sec);
    if (strlen(buf) == kBossVSSettingDateLength)
        memcpy(g_boss_vssetting + kBossVSSettingDateOffset, buf, kBossVSSettingDateLength);
    return g_boss_vssetting;
}

struct BossFile { const char* storage_prefix; const char* name; const uint8_t* data; uint32_t size; };
static const BossFile kBossFiles[] = {
    { "schdat", "VSSetting.byaml", g_boss_vssetting, (uint32_t)sizeof(kBossVSSetting) },
};

static std::string boss_guest_str_mem(uint8_t* mem, uint32_t addr, uint32_t max) {
    if (!addr || !rbrew_memory_accessible(mem, addr, 1u)) return {};
    const char* p = (const char*)(mem + addr);
    size_t n = 0;
    while (n < max && rbrew_memory_accessible(mem, addr + (uint32_t)n, 1u) && p[n]) n++;
    return std::string(p, n);
}
#define boss_guest_str(addr, max) boss_guest_str_mem(MEM, (addr), (max))

static bool boss_storage_has_files(const std::string& storage) {
    for (const BossFile& f : kBossFiles)
        if (storage.rfind(f.storage_prefix, 0) == 0) return true;
    return false;
}

static const BossFile* boss_find(const std::string& storage, const std::string& name) {
    for (const BossFile& f : kBossFiles)
        if (storage.rfind(f.storage_prefix, 0) == 0 && name == f.name) return &f;
    return nullptr;
}

static void boss_log(const char* fmt, const std::string& a, const std::string& b) {
    static std::atomic<int> budget{400};
    if (budget.fetch_sub(1) <= 0) return;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (FILE* f = fopen("boss_log.txt", "a")) {
        fprintf(f, fmt, a.c_str(), b.c_str());
        fprintf(f, "\n");
        fclose(f);
    }
}

static void BOSSStorageInitialize(CPUState* cpu) {
    // Storage::Initialize(char const* dirName [r4], StorageKind kind [r5])
    const uint32_t self = ARG0;
    if (!self || !ARG1 || !rbrew_memory_accessible(MEM, self, 0x28u)) { RET = 0xA0223780u; return; }
    const std::string name = boss_guest_str(ARG1, 8);
    rbrew_write32(MEM, self + 0x04u, ARG2);
    memset(MEM + self + 0x0Bu, 0, 8);
    memcpy(MEM + self + 0x0Bu, name.data(), std::min<size_t>(name.size(), 7));
    boss_log("Storage::Initialize(%s)%s", name, "");
    RET = 0;
}

static void BOSSStorageExist(CPUState* cpu) {
    const uint32_t self = ARG0;
    if (!self || !rbrew_memory_accessible(MEM, self, 0x28u)) { RET = 0; return; }
    RET = boss_storage_has_files(boss_guest_str(self + 0x0Bu, 8)) ? 1u : 0u;
}

static void BOSSStorageGetDataList(CPUState* cpu) {
    // Storage::GetDataList(DataName* outNames [r4], uint32_t maxNames [r5], uint32_t* outCount [r6], uint32_t startIdx [r7])
    const uint32_t self = ARG0, out = ARG1, max = ARG2, out_count = ARG3, start = ARG4;
    uint32_t count = 0;
    if (self && rbrew_memory_accessible(MEM, self, 0x28u)) {
        const std::string storage = boss_guest_str(self + 0x0Bu, 8);
        uint32_t idx = 0;
        for (const BossFile& f : kBossFiles) {
            if (storage.rfind(f.storage_prefix, 0) != 0) continue;
            if (idx++ < start) continue;
            if (count >= max || !out || !rbrew_memory_accessible(MEM, out + count * 0x20u, 0x20u)) break;
            memset(MEM + out + count * 0x20u, 0, 0x20);
            memcpy(MEM + out + count * 0x20u, f.name, std::min<size_t>(strlen(f.name), 31));
            count++;
        }
        boss_log("Storage::GetDataList(%s) -> %s", storage, std::to_string(count));
    }
    if (out_count && rbrew_memory_accessible(MEM, out_count, 4u)) rbrew_write32(MEM, out_count, count);
    RET = 0;
}

static void BOSSNsDataInitialize(CPUState* cpu) {
    // NsData::Initialize(Storage const& storage [r4], char const* dataName [r5])
    const uint32_t self = ARG0, storage = ARG1;
    if (!self || !storage || !rbrew_memory_accessible(MEM, self, 0x50u) ||
        !rbrew_memory_accessible(MEM, storage, 0x28u)) { RET = 0xA0223780u; return; }
    const std::string name = boss_guest_str(ARG2, 32);
    memset(MEM + self, 0, 0x20);
    memcpy(MEM + self, name.data(), std::min<size_t>(name.size(), 31));
    memmove(MEM + self + 0x20u, MEM + storage, 0x28);
    rbrew_write32(MEM, self + 0x48u, 0u);
    rbrew_write32(MEM, self + 0x4Cu, 0u);
    RET = 0;
}

static const BossFile* boss_nsdata_file(CPUState* cpu, uint32_t self) {
    if (!self || !rbrew_memory_accessible(MEM, self, 0x50u)) return nullptr;
    return boss_find(boss_guest_str(self + 0x20u + 0x0Bu, 8), boss_guest_str(self, 32));
}

static void BOSSNsDataExist(CPUState* cpu) {
    RET = boss_nsdata_file(cpu, ARG0) ? 1u : 0u;
}

static void BOSSNsDataGetSize(CPUState* cpu) {
    // int64_t in r3:r4
    const BossFile* f = boss_nsdata_file(cpu, ARG0);
    if (f && f->data == g_boss_vssetting) boss_vssetting_now();
    cpu->r[3] = 0;
    cpu->r[4] = f ? f->size : 0u;
}

static void BOSSNsDataRead(CPUState* cpu) {
    // NsData::Read(void* buffer [r4], uint32_t size [r5]): copies from the seek position.
    const uint32_t self = ARG0, buf = ARG1, size = ARG2;
    const BossFile* f = boss_nsdata_file(cpu, self);
    uint32_t done = 0;
    if (f && buf) {
        const uint32_t seek = rbrew_read32(MEM, self + 0x4Cu);
        if (seek == 0 && f->data == g_boss_vssetting) boss_vssetting_now();
        if (seek < f->size) {
            done = std::min(size, f->size - seek);
            if (rbrew_memory_accessible(MEM, buf, done)) {
                memcpy(MEM + buf, f->data + seek, done);
                rbrew_write32(MEM, self + 0x4Cu, seek + done);
                boss_log("NsData::Read(%s) %s bytes", std::string(f->name), std::to_string(done));
            } else {
                done = 0;
            }
        }
    }
    cpu->r[3] = done;
    cpu->r[4] = done;
}

// Task registration. Splatoon asks "receive the latest News data?" until
// Task::IsRegistered() is true, so registered task ids are remembered (and kept
// in boss_tasks.txt next to the game, like the console keeps them across boots).
// Task (0x20): +0x08 taskId[8].
static std::mutex g_boss_task_mutex;
static std::vector<std::string> g_boss_tasks;
static bool g_boss_tasks_loaded = false;
static const char* kBossTaskFile = "boss_tasks.txt";

static void boss_tasks_load_locked() {
    if (g_boss_tasks_loaded) return;
    g_boss_tasks_loaded = true;
    if (FILE* f = fopen(kBossTaskFile, "r")) {
        char line[64];
        while (fgets(line, sizeof(line), f)) {
            std::string id(line);
            while (!id.empty() && (id.back() == '\n' || id.back() == '\r' || id.back() == ' ')) id.pop_back();
            if (!id.empty()) g_boss_tasks.push_back(id);
        }
        fclose(f);
    }
}

static void boss_tasks_save_locked() {
    if (FILE* f = fopen(kBossTaskFile, "w")) {
        for (const std::string& id : g_boss_tasks) fprintf(f, "%s\n", id.c_str());
        fclose(f);
    }
}

static std::string boss_task_id(CPUState* cpu, uint32_t self) {
    if (!self || !rbrew_memory_accessible(MEM, self, 0x20u)) return {};
    return boss_guest_str(self + 0x08u, 8);
}

static void BOSSTaskInitialize(CPUState* cpu) {
    // Task::Initialize(char const* taskId [r4])
    const uint32_t self = ARG0;
    if (!self || !ARG1 || !rbrew_memory_accessible(MEM, self, 0x20u)) { RET = 0xA0223780u; return; }
    const std::string id = boss_guest_str(ARG1, 8);
    memset(MEM + self + 0x08u, 0, 8);
    memcpy(MEM + self + 0x08u, id.data(), std::min<size_t>(id.size(), 7));
    boss_log("Task::Initialize(%s)%s", id, "");
    RET = 0;
}

static void BOSSTaskRegister(CPUState* cpu) {
    // Task::Register / RegisterForImmediateRun(TaskSetting const&)
    const std::string id = boss_task_id(cpu, ARG0);
    boss_log("Task::Register* this=%s id=%s", std::to_string(ARG0), id);
    if (!id.empty()) {
        std::lock_guard<std::mutex> lock(g_boss_task_mutex);
        boss_tasks_load_locked();
        if (std::find(g_boss_tasks.begin(), g_boss_tasks.end(), id) == g_boss_tasks.end()) {
            g_boss_tasks.push_back(id);
            boss_tasks_save_locked();
        }
        boss_log("Task::Register(%s)%s", id, "");
    }
    RET = 0;
}

static void BOSSTaskUnregister(CPUState* cpu) {
    const std::string id = boss_task_id(cpu, ARG0);
    if (!id.empty()) {
        std::lock_guard<std::mutex> lock(g_boss_task_mutex);
        boss_tasks_load_locked();
        auto it = std::find(g_boss_tasks.begin(), g_boss_tasks.end(), id);
        if (it != g_boss_tasks.end()) {
            g_boss_tasks.erase(it);
            boss_tasks_save_locked();
        }
    }
    RET = 0;
}

static void BOSSTaskIsRegistered(CPUState* cpu) {
    const std::string id = boss_task_id(cpu, ARG0);
    std::lock_guard<std::mutex> lock(g_boss_task_mutex);
    boss_tasks_load_locked();
    RET = (!id.empty() && std::find(g_boss_tasks.begin(), g_boss_tasks.end(), id) != g_boss_tasks.end()) ? 1u : 0u;
    boss_log("Task::IsRegistered(%s) -> %s", id, RET ? "1" : "0");
}

static void BOSSTaskWait(CPUState* cpu) {
    // Task::Wait(timeout, TaskWaitState): true = task finished (not a timeout).
    RET = 1u;
}

static void BOSSTaskGetTurnState(CPUState* cpu) {
    // Task::GetTurnState(u32* execCount): TaskTurnState, 16 = DoneSuccess (17 = DoneError).
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) rbrew_write32(MEM, ARG1, 1u);
    RET = 16u;
}

// nn::boss::DataName::DataName() - nn_boss.rpl 02027700: a null this is
// allocated from the nn_boss allocator (0x20 bytes); the 32-byte name is
// zeroed and this returned. Gambit constructs arrays of them (0279404C).
static void BOSSDataNameCtor(CPUState* cpu) {
    const uint32_t self = ARG0;
    if (!self) throw std::runtime_error("nn::boss::DataName::DataName(nullptr): nn_boss allocator not implemented");
    if (!rbrew_memory_accessible(MEM, self, 0x20))
        throw std::runtime_error("nn::boss::DataName::DataName: object outside guest memory");
    memset(MEM + self, 0, 0x20);
    RET = self;
}

// ---------------------------------------------------------------------------
// nn_fp — Friends Service (local LAN/self-hosted friend list & login)
//
// Configurable via RBREW_FRIENDS="pid:name,pid2:name2" (e.g. "10000002:Player2")
// so Private Battle / Join Friends can list LAN peers directly in-game.
// ---------------------------------------------------------------------------
struct LocalFriendEntry {
    uint32_t pid = 0;
    std::string name;
};

static std::atomic<bool> s_fp_initialized{false};
static std::atomic<bool> s_fp_logged_in{false};

static bool fp_online_mode() {
    const char* nex_host = std::getenv("RBREW_NEX_HOST");
    if (nex_host && *nex_host) return true;
    const char* online = std::getenv("RBREW_ONLINE");
    if (online && (strcmp(online, "1") == 0 || strcmp(online, "true") == 0)) return true;
    const char* ac = std::getenv("RBREW_AC_CONNECT");
    if (ac && strcmp(ac, "ok") == 0) return true;
    return false;
}

static const std::vector<LocalFriendEntry>& fp_get_configured_friends() {
    static const std::vector<LocalFriendEntry> friends = [] {
        std::vector<LocalFriendEntry> list;
        const char* env = std::getenv("RBREW_FRIENDS");
        if (!env || !*env) return list;
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            const size_t colon = item.find(':');
            std::string pid_str = (colon == std::string::npos) ? item : item.substr(0, colon);
            std::string name_str = (colon == std::string::npos) ? ("Friend_" + pid_str) : item.substr(colon + 1);
            char* end = nullptr;
            const unsigned long pid = strtoul(pid_str.c_str(), &end, 0);
            if (end != pid_str.c_str() && pid != 0) {
                if (name_str.empty()) name_str = "Friend";
                list.push_back(LocalFriendEntry{static_cast<uint32_t>(pid), name_str});
            }
        }
        return list;
    }();
    return friends;
}

static const LocalFriendEntry* fp_find_friend(uint32_t pid) {
    for (const auto& f : fp_get_configured_friends()) {
        if (f.pid == pid) return &f;
    }
    return nullptr;
}

static void FPInitialize(CPUState* cpu) {
    s_fp_initialized.store(true);
    RET = 0;
}

static void FPIsInitialized(CPUState* cpu) {
    RET = s_fp_initialized.load() ? 1u : 0u;
}

static void FPIsOnline(CPUState* cpu) {
    RET = (fp_online_mode() || s_fp_logged_in.load()) ? 1u : 0u;
}

static void FPInvokeAsyncCallback(CPUState* cpu, uint32_t callback, uint32_t user, uint32_t result_val) {
    if (!callback) return;
    CPUState call = *cpu;
    call.r[1] = (cpu->r[1] - 0x40) & ~0xFu; // frame below the caller's
    const uint32_t result = call.r[1] + 0x10;
    rbrew_write32(MEM, call.r[1], cpu->r[1]); // back chain
    rbrew_write32(MEM, result, result_val);   // nn::Result
    call.r[3] = result;
    call.r[4] = user;
    rbrew_dispatch(&call, callback);
}

// nn::fp::LoginAsync(callback, userParam)
static void FPLoginAsync(CPUState* cpu) {
    if (fp_online_mode()) s_fp_logged_in.store(true);
    FPInvokeAsyncCallback(cpu, ARG0, ARG1, 0u);
    RET = 0;
}

// nn::fp::RegisterAccountAsync(callback, userParam)
static void FPRegisterAccountAsync(CPUState* cpu) {
    if (fp_online_mode()) s_fp_logged_in.store(true);
    FPInvokeAsyncCallback(cpu, ARG0, ARG1, 0u);
    RET = 0;
}

static void FPLogout(CPUState* cpu) {
    s_fp_logged_in.store(false);
    RET = 0;
}

static void FPResultToErrorCode(CPUState* cpu) {
    uint32_t res_val = 0;
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 4u)) res_val = rbrew_read32(MEM, ARG0);
    RET = (res_val & 0x80000000u) ? 1010000u : 0u;
}

static void FPSetNotificationHandler(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}

// ---------------------------------------------------------------------------
// Live Friend Presence / GameMode sync with self-hosted NEX server (UDP RBFP)
// Shares the already-forwarded NEX Auth UDP port (default 60000) using non-PRUDP
// magic "RBFP", so zero extra ports need forwarding and Private Battle rooms
// (GameMode.joinGameId) propagate live between LAN/WAN peers.
// ---------------------------------------------------------------------------
constexpr uint32_t kGameModeSize = 0x2Cu;

struct RemotePresenceEntry {
    bool is_online = false;
    bool is_same_game = false;
    std::array<uint8_t, kGameModeSize> game_mode{};
};

struct FpPresenceClient {
    std::mutex mutex;
    bool initialized = false;
    bool enabled = false;
#if defined(_WIN32)
    SOCKET sock = INVALID_SOCKET;
#else
    int sock = -1;
#endif
    struct sockaddr_in server_addr{};
    std::array<uint8_t, kGameModeSize> local_game_mode{};
    bool has_local_game_mode = false;
    std::unordered_map<uint32_t, RemotePresenceEntry> cache;
    std::chrono::steady_clock::time_point last_query_time{};
    bool has_server_reply = false;

    bool ensure_ready() {
        if (initialized) return enabled;
        initialized = true;
        const char* nex_host = std::getenv("RBREW_NEX_HOST");
        if (!nex_host || !*nex_host) return false;
#if defined(_WIN32)
        WSADATA wd{};
        WSAStartup(MAKEWORD(2, 2), &wd);
#endif
        struct addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        struct addrinfo* res = nullptr;
        uint16_t port = 60000u;
        if (const char* env_port = std::getenv("RBREW_NEX_PORT"); env_port && *env_port) {
            const long p = std::strtol(env_port, nullptr, 10);
            if (p > 0 && p <= 65535) port = static_cast<uint16_t>(p);
        }
        if (getaddrinfo(nex_host, nullptr, &hints, &res) != 0 || !res) return false;
        std::memcpy(&server_addr, res->ai_addr, sizeof(server_addr));
        server_addr.sin_port = htons(port);
        freeaddrinfo(res);

        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#if defined(_WIN32)
        if (sock == INVALID_SOCKET) return false;
        u_long nb = 1u;
        ioctlsocket(sock, FIONBIO, &nb);
#else
        if (sock < 0) return false;
        const int fl = fcntl(sock, F_GETFL, 0);
        if (fl >= 0) fcntl(sock, F_SETFL, fl | O_NONBLOCK);
#endif
        enabled = true;
        return true;
    }

    static void put_be32(uint8_t* dst, uint32_t v) {
        dst[0] = static_cast<uint8_t>(v >> 24);
        dst[1] = static_cast<uint8_t>(v >> 16);
        dst[2] = static_cast<uint8_t>(v >> 8);
        dst[3] = static_cast<uint8_t>(v);
    }

    static uint32_t get_be32(const uint8_t* src) {
        return (uint32_t(src[0]) << 24) | (uint32_t(src[1]) << 16) | (uint32_t(src[2]) << 8) | uint32_t(src[3]);
    }

    void publish_game_mode(uint32_t my_pid, const uint8_t* gm_bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        std::memcpy(local_game_mode.data(), gm_bytes, kGameModeSize);
        has_local_game_mode = true;
        if (!ensure_ready() || my_pid == 0) return;

        uint8_t pkt[4 + 1 + 4 + kGameModeSize];
        pkt[0] = 'R';
        pkt[1] = 'B';
        pkt[2] = 'F';
        pkt[3] = 'P';
        pkt[4] = 1u; // UPDATE
        put_be32(pkt + 5, my_pid);
        std::memcpy(pkt + 9, gm_bytes, kGameModeSize);
        sendto(sock, reinterpret_cast<const char*>(pkt), static_cast<int>(sizeof(pkt)), 0,
               reinterpret_cast<const struct sockaddr*>(&server_addr), sizeof(server_addr));
    }

    bool query_friends(uint32_t my_pid, const std::vector<uint32_t>& pids) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ensure_ready() || pids.empty()) return false;

        const auto now = std::chrono::steady_clock::now();
        if (has_server_reply &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_query_time).count() < 250) {
            return true;
        }
        last_query_time = now;

        const size_t n = std::min<size_t>(pids.size(), 64u);
        std::vector<uint8_t> req(4 + 1 + 4 + 1 + n * 4);
        req[0] = 'R';
        req[1] = 'B';
        req[2] = 'F';
        req[3] = 'P';
        req[4] = 2u; // QUERY
        put_be32(req.data() + 5, my_pid);
        req[9] = static_cast<uint8_t>(n);
        for (size_t i = 0; i < n; ++i) put_be32(req.data() + 10 + i * 4, pids[i]);

        // Drain any stale replies before sending
        uint8_t drain[1024];
        while (recvfrom(sock, reinterpret_cast<char*>(drain), sizeof(drain), 0, nullptr, nullptr) > 0) {}

        sendto(sock, reinterpret_cast<const char*>(req.data()), static_cast<int>(req.size()), 0,
               reinterpret_cast<const struct sockaddr*>(&server_addr), sizeof(server_addr));

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sock, &rfds);
        struct timeval tv{0, 10000}; // 10ms max wait
#if defined(_WIN32)
        const int rc = select(0, &rfds, nullptr, nullptr, &tv);
#else
        const int rc = select(sock + 1, &rfds, nullptr, nullptr, &tv);
#endif
        if (rc <= 0) return has_server_reply;

        uint8_t resp[4096];
        const int rlen = recvfrom(sock, reinterpret_cast<char*>(resp), sizeof(resp), 0, nullptr, nullptr);
        if (rlen < 6 || resp[0] != 'R' || resp[1] != 'B' || resp[2] != 'F' || resp[3] != 'P' || resp[4] != 3u) {
            return has_server_reply;
        }
        const uint8_t count = resp[5];
        constexpr size_t kRecSize = 4 + 1 + 1 + kGameModeSize;
        size_t pos = 6;
        for (uint8_t i = 0; i < count && pos + kRecSize <= static_cast<size_t>(rlen); ++i) {
            const uint32_t pid = get_be32(resp + pos);
            RemotePresenceEntry entry;
            entry.is_online = resp[pos + 4] != 0;
            entry.is_same_game = resp[pos + 5] != 0;
            std::memcpy(entry.game_mode.data(), resp + pos + 6, kGameModeSize);
            cache[pid] = entry;
            pos += kRecSize;
        }
        has_server_reply = true;
        return true;
    }

    bool get_cached(uint32_t pid, RemotePresenceEntry* out) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(pid);
        if (it == cache.end()) return false;
        *out = it->second;
        return true;
    }
};

static FpPresenceClient g_fp_presence;

static void FPUpdateGameMode(CPUState* cpu) {
    // UpdateGameMode(const GameMode* gameMode [r3], const wchar_t* msg [r4], uint32_t msgLen [r5])
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, kGameModeSize)) {
        const uint32_t my_pid = rbrew_act_get_active_principal_id();
        const uint32_t join_flags = rbrew_read32(MEM, ARG0 + 0x00u);
        const uint32_t join_game_id = rbrew_read32(MEM, ARG0 + 0x08u);
        const uint32_t join_mode = rbrew_read32(MEM, ARG0 + 0x0Cu);
        rbrew_net_trace_log("nn_fp", "UpdateGameMode pid=%u joinFlags=%08X gid=%u mode=%u", my_pid, join_flags,
                            join_game_id, join_mode);
        g_fp_presence.publish_game_mode(my_pid, MEM + ARG0);
    }
    RET = 0;
}

static void FPAddRecentPlayRecord(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}

// GetFriendList(uint32_t* pidList [r3], uint32_t* outCount [r4], uint32_t offset [r5], uint32_t maxCount [r6])
static void FPGetFriendList(CPUState* cpu) {
    const uint32_t pid_list = ARG0;
    const uint32_t out_count_ptr = ARG1;
    const uint32_t offset = ARG2;
    const uint32_t max_count = ARG3;

    const auto& friends = fp_get_configured_friends();
    uint32_t written = 0;
    if (offset < friends.size() && pid_list && max_count > 0) {
        const uint32_t avail = static_cast<uint32_t>(friends.size() - offset);
        const uint32_t n = std::min(avail, max_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!rbrew_memory_accessible(MEM, pid_list + i * 4u, 4u)) break;
            rbrew_write32(MEM, pid_list + i * 4u, friends[offset + i].pid);
            ++written;
        }
    }
    if (out_count_ptr && rbrew_memory_accessible(MEM, out_count_ptr, 4u)) {
        rbrew_write32(MEM, out_count_ptr, written);
    }
    RET = 0;
}

// GetFriendAccountId(char (*out)[17] [r3], const uint32_t* pids [r4], uint32_t count [r5])
static void FPGetFriendAccountId(CPUState* cpu) {
    const uint32_t out = ARG0, pids = ARG1, count = ARG2;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rbrew_memory_accessible(MEM, pids + i * 4u, 4u) || !rbrew_memory_accessible(MEM, out + i * 17u, 17u))
            break;
        const uint32_t pid = rbrew_read32(MEM, pids + i * 4u);
        const auto* f = fp_find_friend(pid);
        std::string acc = f ? f->name : ("Player" + std::to_string(pid));
        memset(MEM + out + i * 17u, 0, 17u);
        const size_t n = std::min<size_t>(acc.size(), 16u);
        memcpy(MEM + out + i * 17u, acc.data(), n);
    }
    RET = 0;
}

// GetFriendScreenName(char16_t (*out)[11] [r3], const uint32_t* pids [r4], uint32_t count [r5], bool replace [r6], uint8_t* outLang [r7])
static void FPGetFriendScreenName(CPUState* cpu) {
    const uint32_t out = ARG0, pids = ARG1, count = ARG2, out_lang = ARG4;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rbrew_memory_accessible(MEM, pids + i * 4u, 4u) || !rbrew_memory_accessible(MEM, out + i * 22u, 22u))
            break;
        const uint32_t pid = rbrew_read32(MEM, pids + i * 4u);
        const auto* f = fp_find_friend(pid);
        std::string name = f ? f->name : ("Player" + std::to_string(pid));
        memset(MEM + out + i * 22u, 0, 22u);
        const size_t n = std::min<size_t>(name.size(), 10u);
        for (size_t j = 0; j < n; ++j) {
            rbrew_write16(MEM, out + i * 22u + uint32_t(j * 2), static_cast<uint16_t>(static_cast<unsigned char>(name[j])));
        }
        if (out_lang && rbrew_memory_accessible(MEM, out_lang + i, 1u)) rbrew_write8(MEM, out_lang + i, 1u); // EN
    }
    RET = 0;
}

// GetFriendMii(FFLStoreData* out [r3], const uint32_t* pids [r4], uint32_t count [r5])
static void FPGetFriendMii(CPUState* cpu) {
    const uint32_t out = ARG0, pids = ARG1, count = ARG2;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rbrew_memory_accessible(MEM, pids + i * 4u, 4u) || !rbrew_memory_accessible(MEM, out + i * 0x60u, 0x60u))
            break;
        const uint32_t pid = rbrew_read32(MEM, pids + i * 4u);
        const auto* f = fp_find_friend(pid);
        std::u16string u16;
        const std::string src = f ? f->name : "Friend";
        for (char c : src) {
            if (u16.size() >= RBREW_ACT_MII_NAME_MAX) break;
            u16.push_back(static_cast<char16_t>(static_cast<unsigned char>(c)));
        }
        if (u16.empty()) u16 = u"Friend";
        const auto mii = rbrew_act_default_mii(u16);
        memcpy(MEM + out + i * 0x60u, mii.data(), mii.size());
    }
    RET = 0;
}

// GetFriendProfile(Profile* out [r3], const uint32_t* pids [r4], uint32_t count [r5]) (4 bytes each)
static void FPGetFriendProfile(CPUState* cpu) {
    const uint32_t out = ARG0, count = ARG2;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rbrew_memory_accessible(MEM, out + i * 4u, 4u)) break;
        rbrew_write32(MEM, out + i * 4u, 0x00310000u); // region/country US
    }
    RET = 0;
}

// GetFriendApprovalTime(DateTime* out [r3], const uint32_t* pids [r4], uint32_t count [r5]) (8 bytes each)
static void FPGetFriendApprovalTime(CPUState* cpu) {
    const uint32_t out = ARG0, count = ARG2;
    for (uint32_t i = 0; i < count; ++i) {
        if (!rbrew_memory_accessible(MEM, out + i * 8u, 8u)) break;
        rbrew_write16(MEM, out + i * 8u + 0u, 2026u);
        rbrew_write8(MEM, out + i * 8u + 2u, 1u);
        rbrew_write8(MEM, out + i * 8u + 3u, 1u);
        rbrew_write8(MEM, out + i * 8u + 4u, 0u);
        rbrew_write8(MEM, out + i * 8u + 5u, 0u);
        rbrew_write8(MEM, out + i * 8u + 6u, 0u);
        rbrew_write8(MEM, out + i * 8u + 7u, 0u);
    }
    RET = 0;
}

// GetFriendPresence(FriendPresence* out [r3], const uint32_t* pids [r4], uint32_t count [r5])
// Verified 0x34-byte layout from Gambit_fn_026F30F0 (Gambit_part0090.cpp:3298-3401):
//   +0x00..+0x2B (44B): nn::fp::GameMode (joinFlagMask, matchmakeSystemType,
//                        joinGameId, joinGameMode, ownerPid, joinGroupId, appArg[20])
//   +0x2C (1B):         modeDescription / joinAvailability (1..7 mapped by fn_026F2F24)
//   +0x30 (1B):         isPlayingSameGame (bool)
//   +0x31 (1B):         isOnline (bool; checked FIRST -- if 0, Gambit treats friend as offline)
static void FPGetFriendPresence(CPUState* cpu) {
    const uint32_t out = ARG0, pids_ptr = ARG1, count = ARG2;
    constexpr uint32_t kPresenceSize = 0x34u;
    const bool online_mode = fp_online_mode();

    std::vector<uint32_t> pids;
    pids.reserve(std::min<uint32_t>(count, 100u));
    for (uint32_t i = 0; i < count && i < 100u; ++i) {
        if (!pids_ptr || !rbrew_memory_accessible(MEM, pids_ptr + i * 4u, 4u)) break;
        pids.push_back(rbrew_read32(MEM, pids_ptr + i * 4u));
    }

    bool synced = false;
    if (online_mode && !pids.empty()) {
        synced = g_fp_presence.query_friends(rbrew_act_get_active_principal_id(), pids);
    }

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t base = out + i * kPresenceSize;
        if (!rbrew_memory_accessible(MEM, base, kPresenceSize)) break;
        memset(MEM + base, 0, kPresenceSize);
        if (!online_mode) continue;

        const uint32_t pid = (i < pids.size()) ? pids[i] : 0u;
        RemotePresenceEntry remote{};
        if (synced && pid != 0 && g_fp_presence.get_cached(pid, &remote)) {
            if (remote.is_online) {
                memcpy(MEM + base + 0x00u, remote.game_mode.data(), kGameModeSize);
                const uint32_t join_flags = rbrew_read32(MEM, base + 0x00u);
                rbrew_write8(MEM, base + 0x2Cu, join_flags ? 2u : 1u);
                rbrew_write8(MEM, base + 0x30u, remote.is_same_game ? 1u : 0u);
                rbrew_write8(MEM, base + 0x31u, 1u); // isOnline = true
            }
        } else {
            // Fallback when server hasn't answered RBFP yet: report configured friend online in Splatoon.
            rbrew_write32(MEM, base + 0x04u, 1u); // matchmakeSystemType = 1
            rbrew_write8(MEM, base + 0x2Cu, 1u);  // valid modeDescription
            rbrew_write8(MEM, base + 0x30u, 1u);  // isPlayingSameGame = true
            rbrew_write8(MEM, base + 0x31u, 1u);  // isOnline = true
        }
    }
    RET = 0;
}

// ---------------------------------------------------------------------------
// nn_acp — Network Time
// ---------------------------------------------------------------------------
static constexpr int64_t kWiiUEpochUnixSec = 946684800LL; // 2000-01-01 00:00:00 UTC

// ACPGetNetworkTime(int64_t* outTime [r3], uint32_t* outValid [r4]) -> int32_t (0=ok)
static void ACPGetNetworkTime(CPUState* cpu) {
    const auto now = std::chrono::system_clock::now();
    const auto us_since_unix =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    const int64_t wiiu_us = us_since_unix - (kWiiUEpochUnixSec * 1000000LL);
    if (ARG0 && rbrew_memory_accessible(MEM, ARG0, 8u)) {
        rbrew_write32(MEM, ARG0 + 0u, static_cast<uint32_t>(static_cast<uint64_t>(wiiu_us) >> 32));
        rbrew_write32(MEM, ARG0 + 4u, static_cast<uint32_t>(static_cast<uint64_t>(wiiu_us) & 0xFFFFFFFFu));
    }
    if (ARG1 && rbrew_memory_accessible(MEM, ARG1, 4u)) {
        rbrew_write32(MEM, ARG1, 1u); // valid = 1
    }
    RET = 0;
}

// ACPConvertNetworkTimeToOSCalendarTime(int64_t netTime [r3:r4], OSCalendarTime* out [r5]) -> int32_t (0=ok)
static void ACPConvertNetworkTimeToOSCalendarTime(CPUState* cpu) {
    const int64_t wiiu_us = static_cast<int64_t>((uint64_t(ARG0) << 32) | uint64_t(ARG1));
    const int64_t total_sec = (wiiu_us / 1000000LL) + kWiiUEpochUnixSec;
    const int64_t rem_us = (wiiu_us >= 0) ? (wiiu_us % 1000000LL) : 0;
    const time_t t = static_cast<time_t>(total_sec);
    struct tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    const uint32_t out = ARG2;
    if (out && rbrew_memory_accessible(MEM, out, 40u)) {
        rbrew_write32(MEM, out + 0x00u, static_cast<uint32_t>(tm_utc.tm_sec));
        rbrew_write32(MEM, out + 0x04u, static_cast<uint32_t>(tm_utc.tm_min));
        rbrew_write32(MEM, out + 0x08u, static_cast<uint32_t>(tm_utc.tm_hour));
        rbrew_write32(MEM, out + 0x0Cu, static_cast<uint32_t>(tm_utc.tm_mday));
        rbrew_write32(MEM, out + 0x10u, static_cast<uint32_t>(tm_utc.tm_mon));
        rbrew_write32(MEM, out + 0x14u, static_cast<uint32_t>(tm_utc.tm_year + 1900));
        rbrew_write32(MEM, out + 0x18u, static_cast<uint32_t>(tm_utc.tm_wday));
        rbrew_write32(MEM, out + 0x1Cu, static_cast<uint32_t>(tm_utc.tm_yday));
        rbrew_write32(MEM, out + 0x20u, static_cast<uint32_t>(rem_us / 1000));
        rbrew_write32(MEM, out + 0x24u, static_cast<uint32_t>(rem_us % 1000));
    }
    RET = 0;
}

// ---------------------------------------------------------------------------
// nn_ndm — Network Daemon Manager
// ---------------------------------------------------------------------------
static void NDMReturnZero(CPUState* cpu) {
    (void)cpu;
    RET = 0;
}

// ---------------------------------------------------------------------------
// proc_ui — Process UI (foreground/background transitions)
// ---------------------------------------------------------------------------
static void ProcUIInit(CPUState* cpu) {
    (void)cpu;
}
static void ProcUIShutdown(CPUState* cpu) {
    (void)cpu;
}
static void ProcUIProcessMessages(CPUState* cpu) {
    RET = 0;
}
static void ProcUIRegisterCallback(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// sysapp — system application helpers
// ---------------------------------------------------------------------------
static void SYSRelaunchTitle(CPUState* cpu) {
    (void)cpu;
}
static void SYSLaunchMenu(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// mic — microphone stubs
// ---------------------------------------------------------------------------
static void MICInit(CPUState* cpu) {
    RET = 0;
}
static void MICOpen(CPUState* cpu) {
    RET = static_cast<uint32_t>(-1);
}
static void MICClose(CPUState* cpu) {
    (void)cpu;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
#include "stubs_addrs.h"

void zlib_guest_register(CPUState* cpu); // zlib_guest.cpp
void stubs_register(CPUState* cpu) {
    zlib_guest_register(cpu);
#define REG(name) rbrew_register_func(cpu, ADDR_##name, name)
#define REG_AT(addr_sym, fn) rbrew_register_func(cpu, ADDR_##addr_sym, fn)

    // nn_save
    REG(SAVEInit);
    REG(SAVEInitSaveDir);
    REG(SAVEGetSharedDataTitlePath);
    REG(SAVEGetSharedSaveDataPath);
    REG(SAVEOpenFile);
    REG(SAVEFlushQuota);
    REG(SAVEMakeDir);
    REG(SAVEOpenDir);
    REG(SAVEGetStat);
    REG(SAVERemove);

    // nn_acp
    REG(ACPGetNetworkTime);
    REG(ACPConvertNetworkTimeToOSCalendarTime);

    // nn_ndm
    REG_AT(Initialize__Q2_2nn3ndmFv, NDMReturnZero);
    REG_AT(Finalize__Q2_2nn3ndmFv, NDMReturnZero);
    REG_AT(SuspendDaemons__Q2_2nn3ndmFUi, NDMReturnZero);
    REG_AT(ResumeDaemons__Q2_2nn3ndmFUi, NDMReturnZero);

    // nn_fp (all 17 PLT imports)
    REG_AT(Initialize__Q2_2nn2fpFv, FPInitialize);
    REG_AT(IsInitialized__Q2_2nn2fpFv, FPIsInitialized);
    REG_AT(IsOnline__Q2_2nn2fpFv, FPIsOnline);
    REG_AT(LoginAsync__Q2_2nn2fpFPFQ2_2nn6ResultPv_vPv, FPLoginAsync);
    REG_AT(RegisterAccountAsync__Q2_2nn2fpFPFQ2_2nn6ResultPv_vPv, FPRegisterAccountAsync);
    REG_AT(Logout__Q2_2nn2fpFv, FPLogout);
    REG_AT(ResultToErrorCode__Q2_2nn2fpFQ2_2nn6Result, FPResultToErrorCode);
    REG_AT(SetNotificationHandler__Q2_2nn2fpFUiPFQ3_2nn2fp16NotificationTypeUiPv_vPv, FPSetNotificationHandler);
    REG_AT(UpdateGameMode__Q2_2nn2fpFPCQ3_2nn2fp8GameModePCwUi, FPUpdateGameMode);
    REG_AT(AddRecentPlayRecord__Q2_2nn2fpFPCQ3_2nn2fp16RecentPlayRecordUi, FPAddRecentPlayRecord);
    REG_AT(GetFriendList__Q2_2nn2fpFPUiT1UiT3, FPGetFriendList);
    REG_AT(GetFriendAccountId__Q2_2nn2fpFPA17_cPCUiUi, FPGetFriendAccountId);
    REG_AT(GetFriendApprovalTime__Q2_2nn2fpFPQ3_2nn2fp8DateTimePCUiUi, FPGetFriendApprovalTime);
    REG_AT(GetFriendMii__Q2_2nn2fpFP12FFLStoreDataPCUiUi, FPGetFriendMii);
    REG_AT(GetFriendPresence__Q2_2nn2fpFPQ3_2nn2fp14FriendPresencePCUiUi, FPGetFriendPresence);
    REG_AT(GetFriendProfile__Q2_2nn2fpFPQ3_2nn2fp7ProfilePCUiUi, FPGetFriendProfile);
    REG_AT(GetFriendScreenName__Q2_2nn2fpFPA11_wPCUiUibPUc, FPGetFriendScreenName);

    // nn_boss (all 46 PLT imports)
    REG_AT(__ct__Q3_2nn4boss8DataNameFv, BOSSDataNameCtor);
    REG_AT(Initialize__Q2_2nn4bossFv, BOSSReturnZero);
    REG_AT(Exist__Q3_2nn4boss6NsDataCFv, BOSSNsDataExist);
    REG_AT(Exist__Q3_2nn4boss7StorageCFv, BOSSStorageExist);
    REG_AT(GetSize__Q3_2nn4boss6NsDataCFv, BOSSNsDataGetSize);
    REG_AT(Read__Q3_2nn4boss6NsDataFPvUi, BOSSNsDataRead);
    REG_AT(GetDataList__Q3_2nn4boss7StorageCFPQ3_2nn4boss8DataNameUiPUiT2, BOSSStorageGetDataList);
    REG_AT(IsRegistered__Q3_2nn4boss4TaskCFv, BOSSTaskIsRegistered);
    REG_AT(GetContentLength__Q3_2nn4boss4TaskCFPUi, BOSSWriteOutZeroU64);
    REG_AT(GetProcessedLength__Q3_2nn4boss4TaskCFPUi, BOSSWriteOutZeroU64);
    REG_AT(GetResult__Q3_2nn4boss4TaskCFPUi, BOSSWriteOutZeroU32);
    REG_AT(GetTurnState__Q3_2nn4boss4TaskCFPUi, BOSSTaskGetTurnState);
    REG_AT(GetHttpStatusCode__Q3_2nn4boss4TaskCFPUi, BOSSGetHttpStatusCode);
    REG_AT(ErrorCode__Q3_2nn4boss10TaskResultCFv, BOSSReturnZero);
    REG_AT(AddInternalCaCert__Q3_2nn4boss14NetTaskSettingFSc, BOSSReturnZero);
    REG_AT(AddLargeHttpHeader__Q3_2nn4boss16RawUlTaskSettingFPCcT1, BOSSReturnZero);
    REG_AT(SetInternalClientCert__Q3_2nn4boss14NetTaskSettingFSc, BOSSReturnZero);
    REG_AT(Initialize__Q3_2nn4boss15NbdlTaskSettingFPCcLT1, BOSSReturnZero);
    REG_AT(Initialize__Q3_2nn4boss16RawUlTaskSettingFPCcPCUcL, BOSSReturnZero);
    REG_AT(Initialize__Q3_2nn4boss17PlayReportSettingFPvUi, BOSSReturnZero);
    REG_AT(Initialize__Q3_2nn4boss4TaskFPCc, BOSSTaskInitialize);
    REG_AT(Initialize__Q3_2nn4boss6NsDataFRCQ3_2nn4boss7StoragePCc, BOSSNsDataInitialize);
    REG_AT(Initialize__Q3_2nn4boss7StorageFPCcQ3_2nn4boss11StorageKind, BOSSStorageInitialize);
    REG_AT(RegisterForImmediateRun__Q3_2nn4boss4TaskFRCQ3_2nn4boss11TaskSetting, BOSSTaskRegister);
    REG_AT(Register__Q3_2nn4boss4TaskFRQ3_2nn4boss11TaskSetting, BOSSTaskRegister);
    REG_AT(Run__Q3_2nn4boss4TaskFb, BOSSReturnZero);
    REG_AT(SetNewArrivalFlagOff__Q3_2nn4boss5TitleFv, BOSSReturnZero);
    REG_AT(Set__Q3_2nn4boss17PlayReportSettingFPCcUi, BOSSReturnZero);
    REG_AT(StopScheduling__Q3_2nn4boss4TaskFv, BOSSReturnZero);
    REG_AT(Unregister__Q3_2nn4boss4TaskFv, BOSSTaskUnregister);
    REG_AT(Wait__Q3_2nn4boss4TaskFUiQ3_2nn4boss13TaskWaitState, BOSSTaskWait);
    REG_AT(__ct__Q3_2nn4boss10TaskResultFQ2_2nn6Result, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss15NbdlTaskSettingFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss16RawUlTaskSettingFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss17PlayReportSettingFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss4TaskFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss5TitleFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss6NsDataFv, BOSSReturnThis);
    REG_AT(__ct__Q3_2nn4boss7StorageFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss15NbdlTaskSettingFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss16RawUlTaskSettingFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss17PlayReportSettingFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss4TaskFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss5TitleFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss6NsDataFv, BOSSReturnThis);
    REG_AT(__dt__Q3_2nn4boss7StorageFv, BOSSReturnThis);

    // proc_ui
    REG(ProcUIInit);
    REG(ProcUIShutdown);
    REG(ProcUIProcessMessages);
    REG(ProcUIRegisterCallback);

    // sysapp
    REG(SYSRelaunchTitle);
    REG(SYSLaunchMenu);

    // mic
    REG(MICInit);
    REG(MICOpen);
    REG(MICClose);

#undef REG_AT
#undef REG
}
