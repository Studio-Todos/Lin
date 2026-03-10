#ifndef LINALANG_RUNTIME_H
#define LINALANG_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>

typedef uint32_t Port;

#define PORT_INDEX_MASK 0x3
#define NODE_INDEX_SHIFT 2

static inline Port make_port(uint32_t node_index, uint8_t port_index) {
    return (node_index << NODE_INDEX_SHIFT) | (port_index & PORT_INDEX_MASK);
}

static inline uint32_t port_node(Port p) {
    return p >> NODE_INDEX_SHIFT;
}

static inline uint8_t port_index(Port p) {
    return p & PORT_INDEX_MASK;
}

typedef enum {
    NODE_ERA, // Eraser
    NODE_CON, // Constructor
    NODE_DUP, // Duplicator
    NODE_NUM, // Number
    NODE_OP,  // Operation
    NODE_FREE // Free slot
} NodeType;

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_LT,
    OP_GT,
    OP_EQ,
} OpType;

typedef struct {
    uint8_t type;
    uint8_t op_type;
    uint32_t value;
    Port p0;
    Port p1;
    Port p2;
} Node;

void inet_init(uint32_t capacity);
void inet_free_memory();
uint32_t inet_alloc_node(uint8_t type);
void inet_free_node(uint32_t index);
void inet_link(Port a, Port b);
void inet_set_value(uint32_t node_index, uint32_t value);
void inet_set_op_type(uint32_t node_index, uint8_t op_type);
uint32_t inet_reduce();
void inet_print_net();

#endif // LINALANG_RUNTIME_H