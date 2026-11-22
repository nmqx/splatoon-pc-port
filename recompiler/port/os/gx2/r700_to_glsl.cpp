#include <cstring>
#include "../../runtime/rbrew_diagnostics.h"
#include "r700_to_glsl.h"
#include "r700_translation_reads.h"
#include "Standalone.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"
#include "util/helpers/StringBuf.h"
#include <string_view>

namespace {
struct OutputDeclaration {
    std::string_view location, qualifiers, name;
};
// Scan the emitter's output declaration grammar once. Keep captured spelling
// (including interpolation qualifiers and whitespace) identical to the old
// regex; other declarations are skipped without allocating a regex machine.
std::vector<OutputDeclaration> vertex_outputs(std::string_view source) {
    std::vector<OutputDeclaration> outputs;
    auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; };
    auto word = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    };
    size_t search = 0;
    while ((search = source.find("layout", search)) != std::string_view::npos) {
        size_t at = search + 6;
        ++search;
        auto ws = [&] {
            const auto begin = at;
            while (at < source.size() && space(source[at])) ++at;
            return at != begin;
        };
        auto take = [&](std::string_view token) {
            if (source.substr(at, token.size()) != token) return false;
            at += token.size();
            return true;
        };
        ws();
        if (!take("(location")) continue;
        ws();
        if (!take("=")) continue;
        ws();
        const size_t location = at;
        while (at < source.size() && source[at] >= '0' && source[at] <= '9') ++at;
        const auto number = source.substr(location, at - location);
        if (number.empty() || !take(")")) continue;
        ws();
        const size_t qualifiers = at;
        while (take("flat") || take("noperspective")) ws();
        const auto interpolation = source.substr(qualifiers, at - qualifiers);
        if (!take("out") || !ws() || !take("vec4") || !ws()) continue;
        const size_t name = at;
        while (at < source.size() && word(source[at])) ++at;
        const auto identifier = source.substr(name, at - name);
        ws();
        if (identifier.empty() || !take(";")) continue;
        outputs.push_back({number, interpolation, identifier});
        search = at;
    }
    return outputs;
}
} // namespace

std::string r700_rect_geometry(const char* vertex_source, bool quad) {
    const std::string_view vs = vertex_source ? vertex_source : "";
    const bool point = vs.find("gl_PointSize;") != std::string::npos;
    std::string gs = std::string("#version 430\nlayout(") + (quad ? "lines_adjacency" : "triangles") +
                     ") in;layout(triangle_strip,max_vertices=4) out;\nin gl_PerVertex{vec4 gl_Position;";
    if (point) gs += "float gl_PointSize;";
    gs += "}gl_in[];out gl_PerVertex{vec4 gl_Position;};\n";
    struct Varying {
        std::string name, input;
        bool flat;
    };
    std::vector<Varying> varyings;
    for (const auto& output : vertex_outputs(vs)) {
        const std::string name(output.name), location(output.location), qualifiers(output.qualifiers);
        const std::string input = "rbrew_rect_" + name;
        gs += "layout(location=" + location + ") " + qualifiers + "in vec4 " + input + "[];\n";
        gs += "layout(location=" + location + ") " + qualifiers + "out vec4 " + name + ";\n";
        varyings.push_back({name, input, qualifiers.find("flat") != std::string::npos});
    }
    gs += "void emitCorner(int a,int b,int c,bool generated){\n";
    for (const auto& v : varyings)
        gs += v.name + "=" +
              (v.flat ? v.input + (quad ? "[3]" : "[2]")
                      : ("generated?" + v.input + "[b]+" + v.input + "[c]-" + v.input + "[a]:" + v.input + "[a]")) +
              ";\n";
    gs += "gl_Position=generated?gl_in[b].gl_Position+gl_in[c].gl_Position-gl_in[a].gl_Position:gl_in[a].gl_Position;"
          "EmitVertex();}\n";
    if (quad)
        gs += "void "
              "main(){emitCorner(1,0,0,false);emitCorner(2,0,0,false);emitCorner(0,0,0,false);emitCorner(3,0,0,false);"
              "EndPrimitive();}\n";
    else
        gs += "void main(){float d01=length(gl_in[0].gl_Position.xy-gl_in[1].gl_Position.xy);float "
              "d02=length(gl_in[0].gl_Position.xy-gl_in[2].gl_Position.xy);float "
              "d12=length(gl_in[1].gl_Position.xy-gl_in[2].gl_Position.xy);int "
              "a=0,b=1,c=2;if(d01>d02&&d01>d12){a=2;b=1;c=0;}else "
              "if(d02>d01&&d02>d12){a=1;b=2;c=0;}emitCorner(a,b,c,false);emitCorner(b,a,c,false);emitCorner(c,a,b,"
              "false);emitCorner(a,b,c,true);EndPrimitive();}\n";
    return gs;
}
LatteFetchShader::~LatteFetchShader() = default;

// GX2InitFetchShaderEx converts GX2 format IDs to Latte fetch formats;
// these IDs are distinct from GX2 texture/surface formats.
static constexpr uint32_t fetch_formats[20] = {1,    2,    5,    6,    7,    0xD,  0xE,  0xF,  0x10, 0x16,
                                               0x1A, 0x19, 0x1D, 0x1E, 0x1F, 0x20, 0x2F, 0x30, 0x22, 0x23};
R700NativeAttribute r700_native_attribute(uint32_t fmt) {
    unsigned raw = fmt & 0x3F;
    if (raw >= 20) return {};
    // Raw integer inputs preserve bits for Cemu's GLSL attribute decoder.
    const uint32_t types[20] = {0x1401, 0x1401, 0x1403, 0x1403, 0x1401, 0x1405, 0x1405, 0x1403, 0x1403, 0x1405,
                                0x1401, 0x1405, 0x1405, 0x1405, 0x1403, 0x1403, 0x1405, 0x1405, 0x1405, 0x1405};
    // Exactly the attribute's bytes (4_4: 1, 8_8: 2, 2_10_10_10: 4): a wider
    // native fetch of a mesh's last vertex reads past its exactly sized
    // vertex buffer, and robust access then zeroes the whole attribute.
    const uint32_t counts[20] = {1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 4, 1, 2, 2, 4, 4, 3, 3, 4, 4};
    const uint32_t endian[20] = {0, 0, 1, 1, 0, 2, 2, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 2, 2, 2};
    return {types[raw], counts[raw], endian[raw]};
}

void r700_shader_registers(uint32_t* r, const uint8_t* p, bool pixel) {
    auto be = [&](unsigned offset) {
        return uint32_t(p[offset]) << 24 | uint32_t(p[offset + 1]) << 16 | uint32_t(p[offset + 2]) << 8 | p[offset + 3];
    };
    // gx2.rpl GX2SetVertexShader (0202fef8), GX2SetPixelShader (02030628).
    if (pixel) {
        r[mmSQ_PGM_RESOURCES_PS] = be(0);
        r[mmSQ_PGM_EXPORTS_PS] = be(4);
        r[mmSPI_PS_IN_CONTROL_0] = be(8);
        r[mmSPI_PS_IN_CONTROL_1] = be(12);
        const unsigned ps_in = std::min(be(16), 32u);
        for (unsigned i = 0; i < ps_in; ++i) r[mmSPI_PS_INPUT_CNTL_0 + i] = be(20 + i * 4);
        for (unsigned i = ps_in; i < 32u; ++i) r[mmSPI_PS_INPUT_CNTL_0 + i] = 0u;
        r[mmCB_SHADER_MASK] = be(0x94);
        r[mmCB_SHADER_CONTROL] = be(0x98);
        r[mmDB_SHADER_CONTROL] = be(0x9C);
        r[mmSPI_INPUT_Z] = be(0xA0);
    } else {
        r[mmSQ_PGM_RESOURCES_VS] = be(0);
        r[mmVGT_PRIMITIVEID_EN] = be(4);
        r[mmSPI_VS_OUT_CONFIG] = be(8);
        const unsigned vs_out = std::min(be(12), 10u);
        for (unsigned i = 0; i < vs_out; ++i) r[mmSPI_VS_OUT_ID_0 + i] = be(16 + i * 4);
        for (unsigned i = vs_out; i < 10u; ++i) r[mmSPI_VS_OUT_ID_0 + i] = 0xFFFFFFFFu;
        r[mmPA_CL_VS_OUT_CNTL] = be(0x38);
        r[mmSQ_VTX_SEMANTIC_CLEAR] = be(0x3C);
        for (unsigned i = 0; i < 32; ++i)
            r[mmSQ_VTX_SEMANTIC_0 + i] = i < std::min(be(0x40), 32u) ? be(0x44 + i * 4) : 0xFF;
        r[mmVGT_STRMOUT_BUFFER_EN] = be(0xC4);
        // GX2SetVertexShader calls 0202F114 for a stream-out VS, publishing
        // the descriptor's four byte strides as hardware dword strides.
        // Geometry-mode ES binding and non-stream-out VS leave them intact.
        if (be(0xD8) != 2 && be(0x110))
            for (unsigned i = 0; i < 4; ++i) r[mmVGT_STRMOUT_VTX_STRIDE_0 + i * 4] = be(0x114 + i * 4) >> 2;
    }
}

void r700_texture_registers(uint32_t* r, const uint8_t* p, uint32_t native_slot) {
    unsigned stage = native_slot / 32, slot = native_slot % 32;
    if (stage > 2 || slot >= 18) return;
    unsigned base = (stage == 0 ? 0xE000 : stage == 1 ? 0xE460 : 0xE930) + slot * 7;
    auto be = [&](unsigned a) {
        return uint32_t(p[a]) << 24 | uint32_t(p[a + 1]) << 16 | uint32_t(p[a + 2]) << 8 | p[a + 3];
    };
    // gx2.rpl GX2SetPixelTexture 020362d8 / FUN_020360e0.
    r[base] = be(0x88);
    r[base + 1] = be(0x8C);
    r[base + 2] = be(0x24) >> 8;
    r[base + 3] = (be(0x2C) ? be(0x2C) : be(0x24)) >> 8;
    r[base + 4] = be(0x90);
    r[base + 5] = be(0x94);
    r[base + 6] = be(0x98);
}

static thread_local R700TranslationReads s_last_reads;
R700TranslationReads r700_last_translation_reads() {
    return s_last_reads;
}

char* r700_to_glsl(const R700TranslateInput& in) {
    s_last_reads = {};
    if (!in.data || !in.size || (in.size & 7)) return nullptr;
    LatteDecompilerOutput_t out{};
    try {
        // An aligned private copy avoids aliasing and unaligned host reads.
        auto words = std::make_unique_for_overwrite<uint32_t[]>(in.size / 4);
        std::memcpy(words.get(), in.data, in.size);
        std::array<uint32_t, 0x10000> regs;
        if (in.context_registers)
            std::copy_n(in.context_registers, regs.size(), regs.data());
        else
            regs.fill(0);
        LatteShader_CreatePSInputTable(LatteSHRC_GetPSInputTable(), regs.data());
        LatteFetchShader fetch;
        std::vector<LatteParsedFetchShaderAttribute> attrs;
        attrs.reserve(in.attributes.size());
        for (const auto& a : in.attributes) {
            if (a.semantic >= 256 || a.buffer >= 16) throw std::runtime_error("Invalid GX2 attribute");
            LatteParsedFetchShaderAttribute attr{};
            attr.semanticId = uint8_t(a.semantic);
            attr.attributeBufferIndex = uint8_t(a.buffer);
            if ((a.format & 0x3F) >= 20) throw std::runtime_error("Unsupported GX2 attribute format");
            attr.offset = a.offset;
            attr.format = Latte::E_HWFMT(fetch_formats[a.format & 0x3F]);
            attr.nfa = (a.format & 0x800) ? 2 : (a.format & 0x100) ? 1 : 0;
            attr.isSigned = (a.format & 0x200) != 0;
            unsigned endian = a.endian == 3 ? r700_native_attribute(a.format).default_endian : a.endian;
            attr.endianSwap = LatteConst::VertexFetchEndianMode(endian);
            attr.fetchType = a.per_instance ? LatteConst::INSTANCE_DATA : LatteConst::VERTEX_DATA;
            attr.aluDivisor = a.divisor;
            for (unsigned i = 0; i < 4; ++i) attr.ds[i] = uint8_t(a.destination_select >> (24 - i * 8));
            attrs.push_back(attr);
        }
        for (auto& attr : attrs) {
            LatteParsedFetchShaderBufferGroup group{};
            group.attributeBufferIndex = attr.attributeBufferIndex;
            group.attribCount = 1;
            group.attrib = &attr;
            fetch.bufferGroups.push_back(group);
        }
        LatteDecompilerOptions options{};
        options.strictMul = true;
        options.nativeVulkan = true;
        options.nativeVulkanNegativeOneToOne = in.vulkan_negative_one_to_one;
        if (in.is_pixel)
            LatteDecompiler_DecompilePixelShader(0, regs.data(), reinterpret_cast<uint8_t*>(words.get()), in.size,
                                                 options, &out);
        else
            LatteDecompiler_DecompileVertexShader(0, regs.data(), reinterpret_cast<uint8_t*>(words.get()), in.size,
                                                  &fetch, options, &out);
        if (!out.shader || out.shader->hasError || !out.shader->strBuf_shaderSource)
            throw std::runtime_error("Latte shader translation failed");
        if (in.info) {
            auto& info = *in.info;
            info.remapped.clear();
            for (auto& e : out.shader->list_remappedUniformEntries)
                info.remapped.push_back({e.isRegister, e.kcacheBankId, e.index, e.mappedIndex});
            std::copy_n(out.sourceMapping.attributeMapping, 256, info.attribute_locations.begin());
            std::copy_n(out.sourceMapping.uniformBuffersBindingPoint, 16, info.buffer_bindings.begin());
            info.uniform_mode = out.shader->uniformMode;
            info.register_count = out.uniformOffsetsVK.count_uniformRegister;
            std::copy_n(out.shader->textureUsesDepthCompare, 18, info.texture_compare.begin());
            info.streamout_mask = uint32_t(out.streamoutBufferWriteMask.to_ulong());
            std::copy_n(out.streamoutBufferStride, 4, info.streamout_strides.begin());
            info.streamout_written_words = out.streamoutWrittenWords;
            auto& vk = info.vk;
            const auto& map = out.resourceMappingVK;
            const auto& offsets = out.uniformOffsetsVK;
            vk = R700VulkanInterface{};
            vk.set = map.setIndex;
            vk.uniform_vars_binding = map.uniformVarsBufferBindingPoint;
            std::copy_n(map.uniformBuffersBindingPoint, 16, vk.uniform_buffer_bindings.begin());
            for (const auto& entry : out.shader->list_quickBufferList)
                if (entry.index < 16) vk.uniform_buffer_vec4s[entry.index] = entry.size / 16;
            for (unsigned t = 0; t < 18; ++t) {
                // -1 whenever the unit does not fetch texels (upstream default),
                // so a zero offset can never be mistaken for the first uniform.
                vk.offset_tex_scale[t] = offsets.offset_texScale[t];
                vk.texture_bindings[t] = out.textureUnitMask[t] ? map.textureUnitToBindingPoint[t] : -1;
                if (!out.textureUnitMask[t]) continue;
                const auto dim = out.shader->textureUnitDim[t];
                vk.texture_dim[t] = dim == Latte::E_DIM::DIM_1D                                       ? 1
                                    : dim == Latte::E_DIM::DIM_2D || dim == Latte::E_DIM::DIM_2D_MSAA ? 2
                                    : dim == Latte::E_DIM::DIM_2D_ARRAY                               ? 3
                                    : dim == Latte::E_DIM::DIM_CUBEMAP                                ? 4
                                    : dim == Latte::E_DIM::DIM_3D                                     ? 5
                                                                                                      : 0;
                vk.texture_integer[t] = out.shader->textureIsIntegerFormat[t];
                vk.texture_shadow[t] = out.shader->textureUsesDepthCompare[t];
            }
            vk.offset_remapped = offsets.offset_remapped;
            vk.offset_registers = offsets.offset_uniformRegister;
            vk.register_vec4s = offsets.count_uniformRegister;
            vk.offset_alpha_ref = offsets.offset_alphaTestRef;
            vk.offset_point_size = offsets.offset_pointSize;
            vk.offset_frag_coord_scale = offsets.offset_fragCoordScale;
            vk.offset_window_to_clip = offsets.offset_windowSpaceToClipSpaceTransform;
            vk.uniform_block_bytes = uint32_t(std::max(0, offsets.offset_endOfBlock));
            vk.pixel_outputs = out.shader->pixelColorOutputMask;
        }
        s_last_reads.valid = true;
        s_last_reads.vs_parameter_mask = in.is_pixel ? 0 : out.shader->outputParameterMask;
        for (unsigned t = 0; t < LATTE_NUM_MAX_TEX_UNITS && t < 32; ++t)
            if (out.textureUnitMask[t]) s_last_reads.texture_unit_mask |= 1u << t;
        const auto* src = out.shader->strBuf_shaderSource->c_str();
        const size_t length = out.shader->strBuf_shaderSource->size();
        const std::string_view src_view(src, length);
        // On R700 hardware, any PS input declared in SPI_PS_INPUT_CNTL_i whose
        // semantic is not exported by the VS receives DEFAULT_VAL (bits 8..9).
        // In Vulkan, leaving that location undeclared in the VS violates
        // VUID-RuntimeSpirv-OpEntryPoint-08743 and feeds undefined values to
        // the fragment shader. Emit matching default-initialized VS outputs.
        std::string default_decls;
        std::string default_inits;
        if (!in.is_pixel && in.context_registers && !out.streamoutBufferWriteMask.any()) {
            uint32_t exported_locations = 0;
            for (const auto& output : vertex_outputs(src_view)) {
                uint32_t loc = 0;
                bool valid_loc = !output.location.empty();
                for (char ch : output.location) {
                    loc = loc * 10u + uint32_t(ch - '0');
                    if (loc >= 32u) {
                        valid_loc = false;
                        break;
                    }
                }
                if (valid_loc) exported_locations |= 1u << loc;
            }
            static constexpr const char* kDefaultVec4[4] = {
                "vec4(0.0, 0.0, 0.0, 0.0)",
                "vec4(0.0, 0.0, 0.0, 1.0)",
                "vec4(1.0, 1.0, 1.0, 0.0)",
                "vec4(1.0, 1.0, 1.0, 1.0)",
            };
            const LatteShaderPSInputTable* ps_table = LatteSHRC_GetPSInputTable();
            const int ps_count = std::min<int>(ps_table->count, 32);
            for (int i = 0; i < ps_count; ++i) {
                if (ps_table->import[i].semanticId > LATTE_ANALYZER_IMPORT_INDEX_PARAM_MAX) continue;
                if ((exported_locations >> i) & 1u) continue;
                const uint32_t def = (regs[mmSPI_PS_INPUT_CNTL_0 + i] >> 8) & 3u;
                default_decls += "layout(location = " + std::to_string(i) + ") ";
                if (ps_table->import[i].isFlat) default_decls += "flat ";
                if (ps_table->import[i].isNoPerspective) default_decls += "noperspective ";
                default_decls += "out vec4 rbrew_default_out_" + std::to_string(i) + ";\n";
                if (default_inits.empty()) default_inits = "\n";
                default_inits += "rbrew_default_out_" + std::to_string(i) + " = " + kDefaultVec4[def] + ";\n";
            }
        }
        constexpr std::string_view push =
            "layout(push_constant) uniform RbrewPushConstants { int rbrew_base_instance; };\n";
        const size_t extra = in.is_pixel ? 0 : push.size();
        const char* line = extra ? static_cast<const char*>(std::memchr(src, '\n', length)) : nullptr;
        const size_t prefix = line ? size_t(line - src) + 1 : 0;
        size_t main_pos = std::string_view::npos;
        size_t body_pos = std::string_view::npos;
        if (!default_decls.empty()) {
            main_pos = src_view.find("void main()");
            if (main_pos != std::string_view::npos && main_pos >= prefix) {
                const size_t brace = src_view.find('{', main_pos);
                if (brace != std::string_view::npos) body_pos = brace + 1;
            }
            if (body_pos == std::string_view::npos) {
                default_decls.clear();
                default_inits.clear();
            }
        }
        const size_t total = length + extra + default_decls.size() + default_inits.size();
        char* result = static_cast<char*>(std::malloc(total + 1));
        if (result) {
            // Assemble directly into the API's final buffer, avoiding an
            // intermediate string, insertion/reallocation and strlen pass.
            char* dst = result;
            auto emit = [&](const char* p, size_t n) {
                if (n) {
                    std::memcpy(dst, p, n);
                    dst += n;
                }
            };
            emit(src, prefix);
            if (extra) emit(push.data(), extra);
            if (!default_decls.empty()) {
                emit(src + prefix, main_pos - prefix);
                emit(default_decls.data(), default_decls.size());
                emit(src + main_pos, body_pos - main_pos);
                emit(default_inits.data(), default_inits.size());
                emit(src + body_pos, length - body_pos + 1);
            } else {
                emit(src + prefix, length - prefix + 1);
            }
        }
        delete out.shader->strBuf_shaderSource;
        delete out.shader;
        return result;
    } catch (const std::exception& error) {
        RBREW_LOG_INFO("shader", "%s\n", error.what());
        if (out.shader) {
            delete out.shader->strBuf_shaderSource;
            delete out.shader;
        }
        return nullptr;
    }
}

std::string r700_vulkan_integer_outputs(const std::string& pixel_source, uint32_t uint_mask, uint32_t sint_mask) {
    std::string source = pixel_source;
    std::string epilogue;
    for (unsigned i = 0; i < 8; ++i) {
        const bool is_uint = (uint_mask >> i) & 1, is_sint = (sint_mask >> i) & 1;
        if (!is_uint && !is_sint) continue;
        const std::string declaration =
            "layout(location = " + std::to_string(i) + ") out vec4 passPixelColor" + std::to_string(i) + ";";
        const auto at = source.find(declaration);
        if (at == std::string::npos) continue; // output not written: the attachment write mask is zero
        const std::string out = "rbrew_pixel_color" + std::to_string(i);
        source.replace(at, declaration.size(),
                       "vec4 passPixelColor" + std::to_string(i) + " = vec4(0.0);\nlayout(location = " +
                           std::to_string(i) + ") out " + (is_uint ? "uvec4 " : "ivec4 ") + out + ";");
        epilogue += out + " = " + (is_uint ? "floatBitsToUint(" : "floatBitsToInt(") + "passPixelColor" +
                    std::to_string(i) + ");\n";
    }
    if (epilogue.empty()) return source;
    const auto main = source.find("void main()");
    if (main == std::string::npos) return source;
    source.replace(main, 11, "void rbrew_main()");
    source += "\nvoid main()\n{\nrbrew_main();\n" + epilogue + "}\n";
    return source;
}

std::string r700_vulkan_capture_position(const std::string& input) {
    std::string source = input;
    const auto block = source.find("out gl_PerVertex");
    if (block == std::string::npos) return source;
    const auto position = source.find("vec4 gl_Position;", block);
    if (position == std::string::npos) return source;
    source.insert(position, "layout(xfb_offset = 0) ");
    source.insert(block, "layout(xfb_buffer = 0, xfb_stride = 16) ");
    // xfb qualifiers are core in GLSL 4.40; the translator emits 4.30.
    const auto version = source.find("#version");
    if (version != std::string::npos) {
        const auto eol = source.find('\n', version);
        if (eol != std::string::npos) source.insert(eol + 1, "#extension GL_ARB_enhanced_layouts : require\n");
    }
    return source;
}
