#pragma once
#include "Project.hpp"
#include <string>

class ProjectManager {
public:
    // Scan projects/ directory and populate workspace.availableProjects
    static void scanProjects(Workspace& workspace);

    // scanProjects + select DefaultProject (or first available) as active
    static void initWorkspace(Workspace& workspace);

    // Set workspace.activeProject to point at the named project folder
    static void applyProject(const std::string& name, Workspace& workspace);

    // Create the standard folder structure for a new project
    static void createProjectFolders(const std::string& name);
};
