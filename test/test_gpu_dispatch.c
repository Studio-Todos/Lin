// GPU end-to-end synthetic test
// Directly calls pic_gpu_dispatch with a crafted interaction net
// to verify the GPU kernel fires and processes real active pairs.
//
// Net layout: two nodes, nodeA(type=1/CON) and nodeB(type=1/CON), p0-connected.
// nodeA is at index 1, nodeB at index 2.
// net[4] = 1 (typeA), net[5] = p1A, net[6] = p2A
// net[8] = 1 (typeB), net[9] = p1B, net[10] = p2B
//
// Active pair: [1, 2] → annihilation rule should wire p1A↔p1B and p2A↔p2B.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Declared in gpu_runtime.c
void pic_gpu_dispatch(int64_t* netPtr, int32_t* activePairsPtr, int32_t numPairs, int32_t* alPtr, const char* spirvPath);

// Port encoding: nodeIdx << 2 | portNum
static int32_t makePort(int32_t nodeIdx, int portNum) {
    return (nodeIdx << 2) | portNum;
}

int main(void) {
    // Allocate the interaction net buffer
    // (matches gpu_runtime.c: (1000000*4 + 1000000*2 + 10) * 4 bytes)
    int32_t netSize = (1000000 * 4 + 1000000 * 2 + 10);
    int64_t* net = (int64_t*)calloc(netSize, sizeof(int64_t));
    if (!net) { fprintf(stderr, "OOM\n"); return 1; }

    // Node 1: CON type=1, p1→node3.p1(dummy), p2→node3.p2(dummy)
    int32_t idxA = 1, idxB = 2;
    net[idxA * 4 + 0] = 1; // type = CON
    net[idxA * 4 + 1] = makePort(3, 1); // p1A → dummy
    net[idxA * 4 + 2] = makePort(3, 2); // p2A → dummy

    // Node 2: CON type=1
    net[idxB * 4 + 0] = 1; // type = CON
    net[idxB * 4 + 1] = makePort(4, 1); // p1B → dummy
    net[idxB * 4 + 2] = makePort(4, 2); // p2B → dummy

    // Active pairs: [nodeA=1, nodeB=2]
    int32_t pairs[2] = { idxA, idxB };

    printf("[TEST] Dispatching GPU with 1 active pair: (%d, %d)\n", idxA, idxB);
    int32_t al = 5;
    pic_gpu_dispatch(net, pairs, 1, &al, "linc_out.spv");

    // After annihilation: p1A should point to p1B's target and vice versa
    printf("[TEST] After reduction:\n");
    printf("  net[%d].p1 = %d (expected %d)\n", idxA, net[idxA*4+1], makePort(4, 1));
    printf("  net[%d].p1 = %d (expected %d)\n", idxB, net[idxB*4+1], makePort(3, 1));

    free(net);
    return 0;
}
