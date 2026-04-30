#ifndef LINALANG_PARSER_H
#define LINALANG_PARSER_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    TOKEN_EOF,
    TOKEN_ERROR,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_FLOAT,
    TOKEN_COLON,        // :
    TOKEN_LBRACKET,     // [
    TOKEN_RBRACKET,     // ]
    TOKEN_LPAREN,       // (
    TOKEN_RPAREN,       // )
    TOKEN_BANG,         // !
    TOKEN_STRING,       // "..."
    TOKEN_STAR,         // *
    TOKEN_SLASH,        // /
    TOKEN_LESS,         // <
    TOKEN_GREATER,      // >
    TOKEN_EQUAL_EQUAL,  // ==
    TOKEN_NOT_EQUAL,   // !=
    TOKEN_GREATER_EQUAL, // >=
    TOKEN_LESS_EQUAL,   // <=
    TOKEN_PLUS,         // +
    TOKEN_MINUS,        // -
    TOKEN_AND,         // &&
    TOKEN_OR,          // ||
    // Keywords
    TOKEN_FUNC,
    TOKEN_RETURN,
    TOKEN_WHILE,
    TOKEN_EITHER,
    TOKEN_I1,
    TOKEN_I8,
    TOKEN_I16,
    TOKEN_I32,
    TOKEN_I64,
    TOKEN_F32,
    TOKEN_F64,
    TOKEN_BOOL,
    TOKEN_STR,
    TOKEN_IMPORT,
    TOKEN_MLIR_OP,
    TOKEN_MODULE,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_DOT,
} TokenType;

typedef struct {
    TokenType type;
    const char *start;
    int length;
    int line;
} Token;

typedef struct {
    const char *start;
    const char *current;
    int line;
} Lexer;

void initLexer(Lexer *lexer, const char *source);
Token scanToken(Lexer *lexer);

// AST Definitions
typedef enum {
    AST_NUMBER,
    AST_FLOAT,
    AST_IDENTIFIER,
    AST_BINARY,
    AST_CALL,
    AST_BLOCK,
    AST_BLOCK_DATA,
    AST_FUNC_DECL,
    AST_STRING,
    AST_ASSIGNMENT,
    AST_ASSIGNMENT_MULTI,
    AST_MLIR_OP,
    AST_IMPORT,
    AST_BOOL,
    AST_WHILE,
    AST_PAIR,
    AST_FIELD_ACCESS,
    AST_MODULE
} AstNodeType;

typedef struct AstNode {
    AstNodeType type;
    const char *resolved_type;
    int resolved_type_len;
    union {
        struct { int32_t value; } number;
        struct { bool value; } boolean;
        struct { float value; } f_number;
        struct { const char *name; int length; } identifier;
        struct { struct AstNode *left; TokenType op; struct AstNode *right; } binary;
        struct { const char *callee; int callee_len; struct AstNode **args; int arg_count; int capacity; } call;
        struct { struct AstNode **statements; int count; int capacity; } block;
        struct {
            const char *name;
            int name_len;
            struct AstFuncArg {
                const char *name;
                int name_len;
                const char *type_name;
                int type_name_len;
            } *args;
            int arg_count;
            int arg_capacity;
            struct AstNode *body;
            const char *return_type_name;
            int return_type_len;
        } func_decl;
        struct { const char *value; int length; } string;
        struct { const char *name; int name_len; struct AstNode *value; } assignment;
        struct { const char *name; int name_len; struct AstNode *value; struct AstNode *next; } assignment_multi;
        struct { const char *name; int name_len; const char *inputs; int inputs_len; const char *outputs; int outputs_len; const char *mlir_payload; int payload_len; } mlir_op;
        struct { const char *path; int length; struct AstNode *module_block; } import_stmt;
        struct { const char *name; int name_len; struct AstNode *module_block; } module;
        struct { struct AstNode *condition; struct AstNode *body; } while_loop;
        struct { struct AstNode *left; struct AstNode *right; } pair;
        struct { struct AstNode *base; int field_index; const char *field_name; int field_name_len; } field_access;
    } as;
} AstNode;

AstNode* parse(const char *source);
void freeAst(AstNode *node);
void printAst(AstNode *node, int depth);

#endif // LINALANG_PARSER_H
