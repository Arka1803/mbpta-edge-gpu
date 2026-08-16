#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define SCENES_DIR "../scenes"
#define ONNX_DIR   "../DNN_models"
#define ENGINE_DIR "trt_engines"
#define OUTPUT_DIR "results/csv_files"

/**
 * Runs trtexec for a given model against all video scenes found in SCENES_DIR.
 * The number of iterations is determined automatically from the scene length
 * by trtexec; we do not hard-code a fixed iteration count.
 *
 * Each scene produces one CSV output file:
 *   results/csv_files/<model>_<scene_stem>_raw.csv
 */
void run_profiling_for_scene(const char* model, const char* scene_path, const char* scene_stem) {
    char cmd[2048];

    printf("  [%s] Profiling scene: %s\n", model, scene_path);

    snprintf(cmd, sizeof(cmd),
             "trtexec --onnx=%s/%s.onnx"
             " --saveEngine=%s/%s.engine"
             " --exportTimes=%s/%s_%s_raw.json"
             " 2>&1 | awk -F',' 'NR>1{print $2}' > %s/%s_%s_raw.csv",
             ONNX_DIR, model,
             ENGINE_DIR, model,
             OUTPUT_DIR, model, scene_stem,
             OUTPUT_DIR, model, scene_stem);

    printf("  CMD: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "  ERROR: trtexec failed for model=%s scene=%s\n", model, scene_stem);
    } else {
        printf("  OK: results saved to %s/%s_%s_raw.csv\n", OUTPUT_DIR, model, scene_stem);
    }
}

/**
 * Strips the file extension from a filename (modifies in place).
 * e.g. "day_foggy.mp4" -> "day_foggy"
 */
void strip_extension(char* name) {
    char* dot = strrchr(name, '.');
    if (dot) *dot = '\0';
}

void run_all_scenes_for_model(const char* model) {
    printf("\n========================================\n");
    printf("Model: %s\n", model);
    printf("========================================\n");

    DIR* dir = opendir(SCENES_DIR);
    if (!dir) {
        fprintf(stderr, "ERROR: Cannot open scenes directory: %s\n", SCENES_DIR);
        fprintf(stderr, "       Make sure your scenes folder exists at %s\n", SCENES_DIR);
        return;
    }

    struct dirent* entry;
    int found = 0;
    while ((entry = readdir(dir)) != NULL) {
        const char* name = entry->d_name;
        // Skip hidden / current / parent entries
        if (name[0] == '.') continue;

        // Only process recognised video extensions
        const char* ext = strrchr(name, '.');
        if (!ext) continue;
        if (strcmp(ext, ".mp4") != 0 && strcmp(ext, ".avi") != 0) continue;

        found = 1;
        char scene_path[1024];
        char scene_stem[512];
        snprintf(scene_path, sizeof(scene_path), "%s/%s", SCENES_DIR, name);
        strncpy(scene_stem, name, sizeof(scene_stem) - 1);
        scene_stem[sizeof(scene_stem) - 1] = '\0';
        strip_extension(scene_stem);

        run_profiling_for_scene(model, scene_path, scene_stem);
    }
    closedir(dir);

    if (!found) {
        printf("  WARNING: No .mp4 or .avi scenes found in %s\n", SCENES_DIR);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_name|all>\n", argv[0]);
        printf("  model_name : one of lenet5, alexnet, vgg16, googlenet, resnet18, resnet50\n");
        printf("  all        : run every supported model against every scene\n");
        return 1;
    }

    const char* agent = argv[1];

    // Create necessary output directories
    system("mkdir -p " ENGINE_DIR);
    system("mkdir -p " OUTPUT_DIR);

    if (strcmp(agent, "all") == 0) {
        const char* models[] = {"lenet5", "alexnet", "vgg16", "googlenet", "resnet18", "resnet50"};
        int n = sizeof(models) / sizeof(models[0]);
        for (int i = 0; i < n; i++) {
            run_all_scenes_for_model(models[i]);
        }
    } else {
        run_all_scenes_for_model(agent);
    }

    printf("\nAll profiling complete.\n");
    return 0;
}
