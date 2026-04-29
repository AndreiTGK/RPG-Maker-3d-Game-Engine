#pragma once
#include <vector>
#include <glm/glm.hpp>

struct NavCell {
    bool walkable = true;
};

struct Navmesh {
    std::vector<NavCell> cells;
    int   width    = 0;
    int   depth    = 0;
    float cellSize = 0.5f;
    float originX  = 0.0f;
    float originY  = 0.0f;

    bool valid() const { return width > 0 && depth > 0; }

    int  idx(int cx, int cy) const { return cy * width + cx; }
    bool inBounds(int cx, int cy) const { return cx >= 0 && cx < width && cy >= 0 && cy < depth; }

    int cellAt(float wx, float wy) const {
        int cx = (int)((wx - originX) / cellSize);
        int cy = (int)((wy - originY) / cellSize);
        if (!inBounds(cx, cy)) return -1;
        return idx(cx, cy);
    }
    glm::vec2 worldPos(int cx, int cy) const {
        return { originX + (cx + 0.5f) * cellSize, originY + (cy + 0.5f) * cellSize };
    }
};
