#pragma once
#include <vulkan/vulkan.h>

class Renderable {
public:
    virtual ~Renderable() {}
    virtual void draw(VkCommandBuffer cmd) = 0;
};
