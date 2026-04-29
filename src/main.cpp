#include "VulkanEngine.hpp"
#include "EngineLog.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <memory>

int main() {
    auto app = std::make_unique<VulkanEngine>();

    try {
        std::cout << ">>> Initializare RPG Maker 3D...\n";
        app->run();
        std::cout << ">>> Inchidere normala.\n";
    } catch (const std::exception& e) {
        LOG_ERROR("CRASH: %s", e.what());
        std::cerr << "\n====================================\n";
        std::cerr << "[CRASH INTERCEPTAT IN MAIN]:\n" << e.what() << '\n';
        std::cerr << "====================================\n\n";
        return EXIT_FAILURE;
    } catch (...) {
        LOG_ERROR("CRASH: eroare fatala necunoscuta");
        std::cerr << "\n[CRASH NECUNOSCUT]: O eroare fatala fara detalii a oprit programul.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
