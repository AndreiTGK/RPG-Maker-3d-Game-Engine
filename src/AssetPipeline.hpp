#pragma once
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// .rpak binary asset pack format
//
// File layout:
//   [RpakHeader 16B][RpakEntry × count][blob data...]
//
// Mesh blob (type=Mesh):
//   lodCount u32 (always 3: full, 50%, 25%)
//   For each LOD:
//     vertCount u32 | idxCount u32 | Vertex[vertCount] | uint32_t[idxCount]
//
// Texture blob (type=Texture):
//   width u32 | height u32 | uint8_t rgba[width*height*4]
// ---------------------------------------------------------------------------

enum class RpakAssetType : uint8_t { Mesh = 0, Texture = 1 };

#pragma pack(push, 1)
struct RpakHeader {
    char     magic[4] = {'R','P','A','K'};
    uint32_t version  = 1;
    uint32_t count    = 0;
    uint32_t _pad     = 0;
};
static_assert(sizeof(RpakHeader) == 16);

struct RpakEntry {
    RpakAssetType type      = RpakAssetType::Mesh;
    char          name[255] = {};   // null-terminated asset name (max 254 chars)
    uint64_t      offset    = 0;    // byte offset from start of file
    uint64_t      size      = 0;    // blob size in bytes
};
static_assert(sizeof(RpakEntry) == 272);
#pragma pack(pop)
