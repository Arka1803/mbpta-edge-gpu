#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void run_profiling(const char* agent, int iterations) {
    char cmd[1024];
    printf("--- Profiling %s with trtexec ---\n", agent);
    
    // trtexec command to build the engine (if from ONNX) and profile it.
    // It will save the engine so subsequent runs are faster if --loadEngine is used, 
    // but here we just show the one-liner that loads onnx, saves engine, and profiles.
    snprintf(cmd, sizeof(cmd), 
             "trtexec --onnx=onnx_models/%s.onnx --saveEngine=trt_engines/%s.engine "
             "--iterations=%d --exportTimes=results/csv_files/%s_times.json", 
             agent, agent, iterations, agent);
    
    printf("Executing: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        printf("Error running trtexec for %s\n", agent);
    } else {
        printf("Profiling complete for %s. Times exported to results/csv_files/%s_times.json\n", agent, agent);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <agent_name|all> [iterations]\n", argv[0]);
        return 1;
    }
    const char* agent = argv[1];
    int iterations = 1000; // Default iterations
    if (argc >= 3) {
        iterations = atoi(argv[2]);
    }

    // Create necessary directories
    system("mkdir -p trt_engines");
    system("mkdir -p results/csv_files");

    if (strcmp(agent, "all") == 0) {
        const char* models[] = {"lenet5", "alexnet", "vgg16", "googlenet", "resnet18", "resnet50"};
        for (int i = 0; i < 6; i++) {
            run_profiling(models[i], iterations);
        }
    } else {
        run_profiling(agent, iterations);
    }

    return 0;
}
