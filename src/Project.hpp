#pragma once
#include <string>
#include <vector>

struct ProjectSettings {
    std::string name = "Nume_Proiect";
    std::string rootPath = ""; // Calea catre folderul proiectului

    // Functii ajutatoare pentru a genera rutele corecte instantaneu
    std::string getModelsPath() const {
        return rootPath + "/assets/models";
    }

    std::string getTexturesPath() const {
        return rootPath + "/assets/textures";
    }

    std::string getScenesPath() const {
        return rootPath + "/scenes";
    }

    std::string getAudioPath() const {
        return rootPath + "/assets/audio";
    }

    std::string getScriptsPath() const {
        return rootPath + "/scripts";
    }

    std::string getPrefabsPath() const {
        return rootPath + "/assets/prefabs";
    }
};

struct Workspace {
    std::vector<std::string> availableProjects;
    ProjectSettings activeProject;
};
