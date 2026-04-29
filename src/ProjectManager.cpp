#include "ProjectManager.hpp"
#include <filesystem>

void ProjectManager::scanProjects(Workspace& workspace) {
    workspace.availableProjects.clear();
    const std::string path = "projects";
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_directory())
                workspace.availableProjects.push_back(entry.path().filename().string());
        }
    }
}

void ProjectManager::initWorkspace(Workspace& workspace) {
    scanProjects(workspace);
    if (workspace.availableProjects.empty()) return;

    std::string target = workspace.availableProjects[0];
    for (const auto& p : workspace.availableProjects) {
        if (p == "DefaultProject") { target = p; break; }
    }
    applyProject(target, workspace);
}

void ProjectManager::applyProject(const std::string& name, Workspace& workspace) {
    workspace.activeProject.name     = name;
    workspace.activeProject.rootPath = "projects/" + name;
}

void ProjectManager::createProjectFolders(const std::string& name) {
    const std::string root = "projects/" + name;
    std::filesystem::create_directories(root + "/assets/models");
    std::filesystem::create_directories(root + "/assets/textures");
    std::filesystem::create_directories(root + "/assets/audio");
    std::filesystem::create_directories(root + "/scenes");
    std::filesystem::create_directories(root + "/scripts");
}
