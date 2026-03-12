#ifndef LINALANG_PARSER_H
#define LINALANG_PARSER_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    TOKEN_EOF,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_COLON,        // :
    TOKEN_LBRACKET,     // [
    TOKEN_RBRACKET,     // ]
    TOKEN_LPAREN,       // (
    TOKEN_RPAREN,       // )
    TOKEN_PLUS,         // +
    TOKEN_MINUS,        // -
    TOKEN_STAR,         // *
    TOKEN_SLASH,        // /
    TOKEN_LESS,         // <
    TOKEN_GREATER,      // >
    TOKEN_BANG,         // !
    TOKEN_STRING,       // "..."
    // Keywords
    TOKEN_FUNC,
    TOKEN_RETURN,
    TOKEN_EITHER,
    TOKEN_I32,
    TOKEN_IMPORT,
    TOKEN_MLIR_OP,
    TOKEN_LBRACE,
    TOKEN_RBRACE
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
    AST_IDENTIFIER,
    AST_BINARY,
    AST_CALL,
    AST_EITHER,
    AST_BLOCK,
    AST_FUNC_DECL,
    AST_STRING,
    AST_ASSIGNMENT,
    AST_MLIR_OP,
    AST_IMPORT
} AstNodeType;

typedef struct AstNode {
    AstNodeType type;
    union {
        struct { int32_t value; } number;
        struct { const char *name; int length; } identifier;
        struct { struct AstNode *left; TokenType op; struct AstNode *right; } binary;
        struct { const char *callee; int callee_len; struct AstNode **args; int arg_count; } call;
        struct { struct AstNode *condition; struct AstNode *true_branch; struct AstNode *false_branch; } either;
        struct { struct AstNode **statements; int count; } block;
        struct { const char *name; int name_len; const char *arg_name; int arg_name_len; struct AstNode *body; } func_decl;
        struct { const char *value; int length; } string;
        struct { const char *name; int name_len; struct AstNode *value; } assignment;
        struct { const char *name; int name_len; const char *inputs; int inputs_len; const char *outputs; int outputs_len; const char *mlir_payload; int payload_len; } mlir_op;
        struct { const char *path; int length; struct AstNode *module_block; } import_stmt;
    } as;
} AstNode;

AstNode* parse(const char *source);
void freeAst(AstNode *node);
void printAst(AstNode *node, int depth);

#endif // LINALANG_PARSER_H
