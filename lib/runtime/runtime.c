#include "runtime.h"
#include <stdlib.h>
#include <stdio.h>

#define MAX_REDEXES 1024 * 1024

static Node* net;
static uint32_t net_capacity;
static uint32_t next_free;

typedef struct {
    uint32_t a;
    uint32_t b;
} Redex;

static Redex* redexes;
static uint32_t redex_count;
static uint32_t reductions;

void inet_init(uint32_t capacity) {
    net_capacity = capacity;
    net = (Node*)calloc(capacity, sizeof(Node));
    for (uint32_t i = 1; i < capacity; i++) {
        net[i].type = NODE_FREE;
        net[i].p0 = i + 1;
    }
    net[capacity - 1].p0 = 0;
    next_free = 1;

    redexes = (Redex*)malloc(MAX_REDEXES * sizeof(Redex));
    redex_count = 0;
    reductions = 0;
}

void inet_free_memory() {
    free(net);
    free(redexes);
}

uint32_t inet_alloc_node(uint8_t type) {
    if (next_free == 0) {
        fprintf(stderr, "Out of interaction net memory\n");
        exit(1);
    }
    uint32_t index = next_free;
    next_free = net[index].p0;

    net[index].type = type;
    net[index].op_type = 0;
    net[index].value = 0;
    net[index].p0 = 0;
    net[index].p1 = 0;
    net[index].p2 = 0;

    return index;
}

void inet_free_node(uint32_t index) {
    net[index].type = NODE_FREE;
    net[index].p0 = next_free;
    next_free = index;
}

void inet_set_value(uint32_t node_index, uint32_t value) {
    net[node_index].value = value;
}

void inet_set_op_type(uint32_t node_index, uint8_t op_type) {
    net[node_index].op_type = op_type;
}

static void add_redex(uint32_t a, uint32_t b) {
    if (redex_count >= MAX_REDEXES) {
        fprintf(stderr, "Redex queue overflow\n");
        exit(1);
    }
    redexes[redex_count++] = (Redex){a, b};
}

void inet_link(Port a, Port b) {
    uint32_t na = port_node(a);
    uint32_t nb = port_node(b);
    uint8_t pa = port_index(a);
    uint8_t pb = port_index(b);

    if (pa == 0) net[na].p0 = b;
    else if (pa == 1) net[na].p1 = b;
    else if (pa == 2) net[na].p2 = b;

    if (pb == 0) net[nb].p0 = a;
    else if (pb == 1) net[nb].p1 = a;
    else if (pb == 2) net[nb].p2 = a;

    if (pa == 0 && pb == 0) {
        add_redex(na, nb);
    }
}

static void rule_annihilation(uint32_t a, uint32_t b) {
    Port a_p1_target = net[a].p1;
    Port a_p2_target = net[a].p2;
    Port b_p1_target = net[b].p1;
    Port b_p2_target = net[b].p2;

    inet_link(a_p1_target, b_p1_target);
    inet_link(a_p2_target, b_p2_target);

    inet_free_node(a);
    inet_free_node(b);
}

static void rule_commutation(uint32_t a, uint32_t b) {
    uint32_t n1 = inet_alloc_node(net[b].type);
    uint32_t n2 = inet_alloc_node(net[b].type);
    uint32_t n3 = inet_alloc_node(net[a].type);
    uint32_t n4 = inet_alloc_node(net[a].type);

    Port a_p1_target = net[a].p1;
    Port a_p2_target = net[a].p2;
    Port b_p1_target = net[b].p1;
    Port b_p2_target = net[b].p2;

    inet_link(make_port(n1, 1), make_port(n3, 1));
    inet_link(make_port(n1, 2), make_port(n4, 1));
    inet_link(make_port(n2, 1), make_port(n3, 2));
    inet_link(make_port(n2, 2), make_port(n4, 2));

    inet_link(make_port(n1, 0), a_p1_target);
    inet_link(make_port(n2, 0), a_p2_target);
    inet_link(make_port(n3, 0), b_p1_target);
    inet_link(make_port(n4, 0), b_p2_target);

    inet_free_node(a);
    inet_free_node(b);
}

static void rule_era_con(uint32_t era, uint32_t con) {
    uint32_t e1 = inet_alloc_node(NODE_ERA);
    uint32_t e2 = inet_alloc_node(NODE_ERA);

    inet_link(make_port(e1, 0), net[con].p1);
    inet_link(make_port(e2, 0), net[con].p2);

    inet_free_node(era);
    inet_free_node(con);
}

static void rule_era_era(uint32_t a, uint32_t b) {
    inet_free_node(a);
    inet_free_node(b);
}

static void rule_num_era(uint32_t num, uint32_t era) {
    inet_free_node(num);
    inet_free_node(era);
}

static void rule_num_dup(uint32_t num, uint32_t dup) {
    uint32_t n1 = inet_alloc_node(NODE_NUM);
    uint32_t n2 = inet_alloc_node(NODE_NUM);
    net[n1].value = net[num].value;
    net[n2].value = net[num].value;

    inet_link(make_port(n1, 0), net[dup].p1);
    inet_link(make_port(n2, 0), net[dup].p2);

    inet_free_node(num);
    inet_free_node(dup);
}

static void rule_op_num(uint32_t op, uint32_t num) {
    uint32_t op2 = inet_alloc_node(NODE_OP);
    net[op2].op_type = net[op].op_type + 100;
    net[op2].value = net[num].value;

    Port rhs = net[op].p1;
    Port result = net[op].p2;

    inet_link(make_port(op2, 0), rhs);
    inet_link(make_port(op2, 1), result);

    inet_free_node(op);
    inet_free_node(num);
}

static void rule_op_partial_num(uint32_t op, uint32_t num) {
    uint32_t lhs = net[op].value;
    uint32_t rhs = net[num].value;
    uint32_t res = 0;

    uint8_t base_op = net[op].op_type - 100;
    switch (base_op) {
        case OP_ADD: res = lhs + rhs; break;
        case OP_SUB: res = lhs - rhs; break;
        case OP_MUL: res = lhs * rhs; break;
        case OP_DIV: res = rhs == 0 ? 0 : lhs / rhs; break;
        case OP_LT:  res = lhs < rhs ? 1 : 0; break;
        case OP_GT:  res = lhs > rhs ? 1 : 0; break;
        case OP_EQ:  res = lhs == rhs ? 1 : 0; break;
    }

    uint32_t res_num = inet_alloc_node(NODE_NUM);
    net[res_num].value = res;

    Port result_port = net[op].p1;
    inet_link(make_port(res_num, 0), result_port);

    inet_free_node(op);
    inet_free_node(num);
}

static void interact(uint32_t a, uint32_t b) {
    uint8_t ta = net[a].type;
    uint8_t tb = net[b].type;

    if (ta > tb) {
        uint32_t temp = a; a = b; b = temp;
        uint8_t ttemp = ta; ta = tb; tb = ttemp;
    }

    // PIC Rules
    if (ta == NODE_ERA && tb == NODE_ERA) {
        rule_era_era(a, b); // Rule 3 (Era-Era propagates and vanishes)
    } else if (ta == NODE_ERA && (tb == NODE_CON || tb == NODE_DUP || tb == NODE_OP)) {
        rule_era_con(a, b); // Rule 3 (Erasure Propagation on all auxiliary ports)
    } else if (ta == NODE_ERA && tb == NODE_NUM) {
        rule_num_era(b, a); // Rule 3 (Erasure Propagation on number)
    } else if (ta == NODE_CON && tb == NODE_CON) {
        rule_annihilation(a, b); // Rule 1 (Annihilation γ⁺ ⋈ γ⁻ or γ⁺ ⋈ γ⁺ depending on what we mapped it to)
    } else if (ta == NODE_CON && tb == NODE_DUP) {
        rule_commutation(a, b); // Rule 2 (Duplication δ ⋈ γ⁺)
    } else if (ta == NODE_DUP && tb == NODE_DUP) {
        rule_annihilation(a, b); // Rule 1 (Annihilation on matching dup ports)
    } else if (ta == NODE_DUP && tb == NODE_NUM) {
        rule_num_dup(b, a);
    } else if (ta == NODE_DUP && tb == NODE_OP) {
        rule_commutation(b, a); // Ops commute with Dups like Con
    } else if (ta == NODE_NUM && tb == NODE_OP) {
        if (net[b].op_type >= 100) {
            rule_op_partial_num(b, a);
        } else {
            rule_op_num(b, a);
        }
    } else if (ta == NODE_OP && tb == NODE_OP) {
        // Op annihilation (not implemented here fully for side effects, just clear)
        rule_annihilation(a, b);
    }
}

uint32_t inet_reduce() {
    while (redex_count > 0) {
        Redex r = redexes[--redex_count];
        if (net[r.a].type != NODE_FREE && net[r.b].type != NODE_FREE &&
            port_node(net[r.a].p0) == r.b && port_node(net[r.b].p0) == r.a) {
            interact(r.a, r.b);
            reductions++;
        }
    }
    return reductions;
}

void inet_print_net() {
    printf("--- Interaction Net ---\n");
    for (uint32_t i = 1; i < net_capacity; i++) {
        if (net[i].type != NODE_FREE) {
            printf("Node %d: Type %d", i, net[i].type);
            if (net[i].type == NODE_NUM) printf(" (Val %d)", net[i].value);
            if (net[i].type == NODE_OP) printf(" (Op %d)", net[i].op_type);
            printf(" | p0: %d:%d | p1: %d:%d | p2: %d:%d\n",
                port_node(net[i].p0), port_index(net[i].p0),
                port_node(net[i].p1), port_index(net[i].p1),
                port_node(net[i].p2), port_index(net[i].p2));
        }
    }
    printf("-----------------------\n");
}