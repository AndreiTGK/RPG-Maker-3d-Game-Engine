#pragma once
#include "Scene.hpp"

class PhysicsEngine {
public:
    // Verifica daca volumul obiectului A se intersecteaza cu volumul obiectului B
    static bool checkCollision(const GameObject& a, const GameObject& b) {
        // Extragem marginile pentru A
        glm::vec3 minA = a.transform.translation - (a.transform.scale * 0.5f);
        glm::vec3 maxA = a.transform.translation + (a.transform.scale * 0.5f);

        // Extragem marginile pentru B
        glm::vec3 minB = b.transform.translation - (b.transform.scale * 0.5f);
        glm::vec3 maxB = b.transform.translation + (b.transform.scale * 0.5f);

        // Verificam suprapunerea pe axele X, Y si Z
        return (minA.x <= maxB.x && maxA.x >= minB.x) &&
        (minA.y <= maxB.y && maxA.y >= minB.y) &&
        (minA.z <= maxB.z && maxA.z >= minB.z);
    }
};
