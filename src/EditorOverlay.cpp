#include "EditorOverlay.hpp"
#include <glm/glm.hpp>

void EditorOverlay::createGridGeometry(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
    std::vector<Vertex> vertices;
    int size = 20; // Limitam dimensiunea pentru a nu "inunda" ecranul

    for (int i = -size; i <= size; i++) {
        // Intensitatea culorii scade spre margini pentru un look curat
        float alpha = 1.0f - (std::abs(i) / (float)size);
        glm::vec3 color = glm::vec3(0.4f * alpha);

        // Axa X (Rosu) si Axa Y (Verde) la centru
        glm::vec3 colorX = (i == 0) ? glm::vec3(1.0f, 0.2f, 0.2f) : color;
        glm::vec3 colorY = (i == 0) ? glm::vec3(0.2f, 1.0f, 0.2f) : color;

        // Liniile sunt definite pe XY (Z = 0) pentru a fi orizontale
        vertices.push_back({{ (float)i, (float)-size, 0.0f }, colorX, {0,0}});
        vertices.push_back({{ (float)i, (float)size, 0.0f }, colorX, {0,0}});
        vertices.push_back({{ (float)-size, (float)i, 0.0f }, colorY, {0,0}});
        vertices.push_back({{ (float)size, (float)i, 0.0f }, colorY, {0,0}});
    }

    gridMesh.indexCount = static_cast<uint32_t>(vertices.size());
    // Aici apelezi functia de creare buffer (trebuie sa fie accesibila sau mutata in Utils)
}

void EditorOverlay::drawGrid(VkCommandBuffer commandBuffer, VkPipeline linePipeline, VkPipelineLayout layout) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);

    // Matricea model este identitate (grila e statica la 0,0,0)
    glm::mat4 model = glm::mat4(1.0f);
    vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &model);

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &gridMesh.vertexBuffer, offsets);
    vkCmdDraw(commandBuffer, gridMesh.indexCount, 1, 0, 0);
}
