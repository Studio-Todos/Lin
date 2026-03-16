#include "lin/Parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static bool isAtEnd(Lexer *lexer) { return *lexer->current == '\0'; }
static char advance(Lexer *lexer) { return *lexer->current++; }
static char peek(Lexer *lexer) { return *lexer->current; }
static char peekNext(Lexer *lexer) {
    if (isAtEnd(lexer)) return '\0';
    return lexer->current[1];
}
static bool match(Lexer *lexer, char expected) {
    if (isAtEnd(lexer) || *lexer->current != expected) return false;
    lexer->current++;
    return true;
}

static void skipWhitespace(Lexer *lexer) {
    for (;;) {
        char c = peek(lexer);
        switch (c) {
            case ' ': case '\r': case '\t': advance(lexer); break;
            case '\n': lexer->line++; advance(lexer); break;
            case ';':
                while (peek(lexer) != '\n' && !isAtEnd(lexer)) advance(lexer);
                break;
            case '/':
                if (peekNext(lexer) == '/') {
                    while (peek(lexer) != '\n' && !isAtEnd(lexer)) advance(lexer);
                    break;
                }
                return;
            default: return;
        }
    }
}

static Token makeToken(Lexer *lexer, TokenType type) {
    Token token;
    token.type = type;
    token.start = lexer->start;
    token.length = (int)(lexer->current - lexer->start);
    token.line = lexer->line;
    return token;
}

static Token number(Lexer *lexer) {
    while (isdigit(peek(lexer))) advance(lexer);
    return makeToken(lexer, TOKEN_NUMBER);
}

static TokenType checkKeyword(Lexer *lexer, int start, int length, const char *rest, TokenType type) {
    if (lexer->current - lexer->start == start + length &&
        memcmp(lexer->start + start, rest, length) == 0) {
        return type;
    }
    return TOKEN_IDENTIFIER;
}

static TokenType identifierType(Lexer *lexer) {
    switch (lexer->start[0]) {
        case 'e': return checkKeyword(lexer, 1, 5, "ither", TOKEN_EITHER);
        case 'f':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'u': return checkKeyword(lexer, 2, 2, "nc", TOKEN_FUNC);
                    case '3': return checkKeyword(lexer, 2, 1, "2", TOKEN_F32);
                    case '6': return checkKeyword(lexer, 2, 1, "4", TOKEN_F64);
                }
            }
            break;
        case 'i':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case '1':
                        if (lexer->current - lexer->start == 2) return TOKEN_I1;
                        return checkKeyword(lexer, 2, 1, "6", TOKEN_I16);
                    case '3': return checkKeyword(lexer, 2, 1, "2", TOKEN_I32);
                    case '6': return checkKeyword(lexer, 2, 1, "4", TOKEN_I64);
                    case '8': return checkKeyword(lexer, 2, 0, "", TOKEN_I8);
                    case 'm': return checkKeyword(lexer, 2, 4, "port", TOKEN_IMPORT);
                }
            }
            break;
        case 'm': return checkKeyword(lexer, 1, 6, "lir-op", TOKEN_MLIR_OP);
        case 'r': return checkKeyword(lexer, 1, 5, "eturn", TOKEN_RETURN);
    }
    return TOKEN_IDENTIFIER;
}

static Token string(Lexer *lexer) {
    while (peek(lexer) != '"' && !isAtEnd(lexer)) {
        if (peek(lexer) == '\n') lexer->line++;
        advance(lexer);
    }
    if (isAtEnd(lexer)) return makeToken(lexer, TOKEN_EOF);
    advance(lexer);
    return makeToken(lexer, TOKEN_STRING);
}

static Token identifier(Lexer *lexer) {
    while (isalpha(peek(lexer)) || isdigit(peek(lexer)) || peek(lexer) == '_' || peek(lexer) == '-') advance(lexer);
    return makeToken(lexer, identifierType(lexer));
}

void initLexer(Lexer *lexer, const char *source) {
    lexer->start = source;
    lexer->current = source;
    lexer->line = 1;
}

Token scanToken(Lexer *lexer) {
    skipWhitespace(lexer);
    lexer->start = lexer->current;

    if (isAtEnd(lexer)) return makeToken(lexer, TOKEN_EOF);

    char c = advance(lexer);
    if (isalpha(c)) return identifier(lexer);
    if (isdigit(c)) return number(lexer);

    switch (c) {
        case '"': return string(lexer);
        case ':': return makeToken(lexer, TOKEN_COLON);
        case '[': return makeToken(lexer, TOKEN_LBRACKET);
        case ']': return makeToken(lexer, TOKEN_RBRACKET);
        case '(': return makeToken(lexer, TOKEN_LPAREN);
        case ')': return makeToken(lexer, TOKEN_RPAREN);
        case '!': return makeToken(lexer, TOKEN_BANG);
        case '{': return makeToken(lexer, TOKEN_LBRACE);
        case '}': return makeToken(lexer, TOKEN_RBRACE);
        case '%': return makeToken(lexer, TOKEN_IDENTIFIER);
        case '=': return makeToken(lexer, TOKEN_IDENTIFIER);
        case '.': return makeToken(lexer, TOKEN_IDENTIFIER);
        case ',': return makeToken(lexer, TOKEN_IDENTIFIER);
    }

    return makeToken(lexer, TOKEN_EOF);
}

typedef struct {
    Lexer lexer;
    Token current;
    Token previous;
    bool hadError;
} Parser;

static void errorAt(Parser *parser, Token *token, const char *message) {
    if (parser->hadError) return;
    fprintf(stderr, "[line %d] Error", token->line);
    if (token->type == TOKEN_EOF) {
        fprintf(stderr, " at end");
    } else {
        fprintf(stderr, " at '%.*s'", token->length, token->start);
    }
    fprintf(stderr, ": %s\n", message);
    parser->hadError = true;
}

static void error(Parser *parser, const char *message) {
    errorAt(parser, &parser->previous, message);
}

static void parserAdvance(Parser *parser) {
    parser->previous = parser->current;
    for (;;) {
        parser->current = scanToken(&parser->lexer);
        if (parser->current.type != TOKEN_EOF || parser->lexer.current[0] == '\0') break;
    }
}

static void consume(Parser *parser, TokenType type, const char *message) {
    if (parser->current.type == type) {
        parserAdvance(parser);
        return;
    }
    errorAt(parser, &parser->current, message);
}

static AstNode* parseExpression(Parser *parser);
static AstNode* parseStatement(Parser *parser);

static AstNode* createNode(AstNodeType type) {
    AstNode *node = (AstNode*)malloc(sizeof(AstNode));
    node->type = type;
    return node;
}

static AstNode* parsePrimary(Parser *parser) {
    if (parser->current.type == TOKEN_NUMBER) {
        AstNode *node = createNode(AST_NUMBER);
        node->as.number.value = atoi(parser->current.start);
        parserAdvance(parser);
        return node;
    }

    if (parser->current.type == TOKEN_STRING) {
        AstNode *node = createNode(AST_STRING);
        node->as.string.value = parser->current.start + 1;
        node->as.string.length = parser->current.length - 2;
        parserAdvance(parser);
        return node;
    }

    if (parser->current.type == TOKEN_IMPORT) {
        parserAdvance(parser);
        consume(parser, TOKEN_STRING, "Expect string path after 'import'.");
        AstNode *node = createNode(AST_IMPORT);
        node->as.import_stmt.path = parser->previous.start + 1;
        node->as.import_stmt.length = parser->previous.length - 2;
        node->as.import_stmt.module_block = NULL;
        return node;
    }

    if (parser->current.type == TOKEN_IDENTIFIER) {
        Token ident = parser->current;
        parserAdvance(parser);

        // Check for variable assignment: `word: value`
        if (parser->current.type == TOKEN_COLON) {
            const char* next_word = parser->lexer.current;
            while (*next_word == ' ' || *next_word == '\n' || *next_word == '\t' || *next_word == '\r') next_word++;

            bool is_func = (strncmp(next_word, "func", 4) == 0);
            bool is_mlir_op = (strncmp(next_word, "mlir-op", 7) == 0);

            if (is_mlir_op) {
                parserAdvance(parser); // consume colon
                consume(parser, TOKEN_MLIR_OP, "Expect 'mlir-op'.");

                Token inputs_outputs_start = parser->current;
                consume(parser, TOKEN_LBRACKET, "Expect '[' for mlir-op interface.");
                int bracket_count = 1;
                while (bracket_count > 0 && parser->current.type != TOKEN_EOF) {
                    if (parser->current.type == TOKEN_LBRACKET) bracket_count++;
                    else if (parser->current.type == TOKEN_RBRACKET) bracket_count--;
                    parserAdvance(parser);
                }
                Token inputs_outputs_end = parser->previous;

                Token payload_start = parser->current;
                consume(parser, TOKEN_LBRACE, "Expect '{' for mlir-op payload.");
                int brace_count = 1;
                while (brace_count > 0 && parser->current.type != TOKEN_EOF) {
                    if (parser->current.type == TOKEN_LBRACE) brace_count++;
                    else if (parser->current.type == TOKEN_RBRACE) brace_count--;
                    parserAdvance(parser);
                }
                Token payload_end = parser->previous;

                AstNode *node = createNode(AST_MLIR_OP);
                node->as.mlir_op.name = ident.start;
                node->as.mlir_op.name_len = ident.length;
                node->as.mlir_op.inputs = inputs_outputs_start.start;
                node->as.mlir_op.inputs_len = (int)(inputs_outputs_end.start + inputs_outputs_end.length - inputs_outputs_start.start);
                node->as.mlir_op.mlir_payload = payload_start.start + 1; // Skip `{`
                node->as.mlir_op.payload_len = (int)(payload_end.start - payload_start.start - 1); // Skip `{` and `}`

                while (node->as.mlir_op.payload_len > 0 && (*node->as.mlir_op.mlir_payload == '\n' || *node->as.mlir_op.mlir_payload == '\r')) {
                    node->as.mlir_op.mlir_payload++;
                    node->as.mlir_op.payload_len--;
                }

                return node;
            } else if (!is_func) {
                parserAdvance(parser); // consume colon
                AstNode *node = createNode(AST_ASSIGNMENT);
                node->as.assignment.name = ident.start;
                node->as.assignment.name_len = ident.length;
                node->as.assignment.value = parseExpression(parser);
                return node;
            }
        }

        AstNode *node = createNode(AST_IDENTIFIER);
        node->as.identifier.name = ident.start;
        node->as.identifier.length = ident.length;
        return node;
    }

    if (parser->current.type == TOKEN_LPAREN) {
        parserAdvance(parser);
        if (parser->current.type == TOKEN_IDENTIFIER) {
            AstNode *call = createNode(AST_CALL);
            call->as.call.callee = parser->current.start;
            call->as.call.callee_len = parser->current.length;
            parserAdvance(parser);

            call->as.call.args = NULL;
            call->as.call.arg_count = 0;
            while (parser->current.type != TOKEN_RPAREN && parser->current.type != TOKEN_EOF) {
                AstNode *arg = parseExpression(parser);
                call->as.call.arg_count++;
                call->as.call.args = (AstNode**)realloc(call->as.call.args, sizeof(AstNode*) * call->as.call.arg_count);
                call->as.call.args[call->as.call.arg_count - 1] = arg;
            }
            consume(parser, TOKEN_RPAREN, "Expect ')' after arguments.");
            return call;
        } else {
            AstNode *expr = parseExpression(parser);
            consume(parser, TOKEN_RPAREN, "Expect ')' after expression.");
            return expr;
        }
    }

    error(parser, "Expect expression.");
    return NULL;
}

static AstNode* parseExpression(Parser *parser) {
    return parsePrimary(parser);
}

static AstNode* parseBlock(Parser *parser) {
    consume(parser, TOKEN_LBRACKET, "Expect '[' before block.");
    AstNode *block = createNode(AST_BLOCK);
    block->as.block.statements = NULL;
    block->as.block.count = 0;

    while (parser->current.type != TOKEN_RBRACKET && parser->current.type != TOKEN_EOF) {
        AstNode *stmt = parseStatement(parser);
        block->as.block.count++;
        block->as.block.statements = (AstNode**)realloc(block->as.block.statements, sizeof(AstNode*) * block->as.block.count);
        block->as.block.statements[block->as.block.count - 1] = stmt;
    }

    consume(parser, TOKEN_RBRACKET, "Expect ']' after block.");
    return block;
}

static AstNode* parseStatement(Parser *parser) {
    if (parser->current.type == TOKEN_EITHER) {
        parserAdvance(parser);
        AstNode *either = createNode(AST_EITHER);
        either->as.either.condition = parseExpression(parser);
        either->as.either.true_branch = parseBlock(parser);
        either->as.either.false_branch = parseBlock(parser);
        return either;
    }

    return parseExpression(parser);
}

static AstNode* parseFuncDecl(Parser *parser) {
    Token name = parser->previous;
    consume(parser, TOKEN_COLON, "Expect ':' after function name.");
    consume(parser, TOKEN_FUNC, "Expect 'func'.");
    consume(parser, TOKEN_LBRACKET, "Expect '[' for args.");

    AstNode *func = createNode(AST_FUNC_DECL);
    func->as.func_decl.name = name.start;
    func->as.func_decl.name_len = name.length;
    func->as.func_decl.args = NULL;
    func->as.func_decl.arg_count = 0;

    while (parser->current.type != TOKEN_RETURN && parser->current.type != TOKEN_EOF) {
        Token argName = parser->current;
        consume(parser, TOKEN_IDENTIFIER, "Expect argument name.");
        consume(parser, TOKEN_LBRACKET, "Expect '[' for arg type.");
        if (parser->current.type == TOKEN_I1 || parser->current.type == TOKEN_I8 ||
            parser->current.type == TOKEN_I16 || parser->current.type == TOKEN_I32 ||
            parser->current.type == TOKEN_I64 || parser->current.type == TOKEN_F32 ||
            parser->current.type == TOKEN_F64) {
            parserAdvance(parser);
        } else {
            errorAt(parser, &parser->current, "Expect type (i1, i8, i16, i32, i64, f32, f64).");
        }
        consume(parser, TOKEN_BANG, "Expect '!'.");
        consume(parser, TOKEN_RBRACKET, "Expect ']' after arg type.");

        func->as.func_decl.arg_count++;
        func->as.func_decl.args = realloc(func->as.func_decl.args, sizeof(*func->as.func_decl.args) * func->as.func_decl.arg_count);
        func->as.func_decl.args[func->as.func_decl.arg_count - 1].name = argName.start;
        func->as.func_decl.args[func->as.func_decl.arg_count - 1].name_len = argName.length;
    }

    consume(parser, TOKEN_RETURN, "Expect 'return'.");
    consume(parser, TOKEN_COLON, "Expect ':' after return.");
    consume(parser, TOKEN_LBRACKET, "Expect '[' for return type.");
    if (parser->current.type == TOKEN_I1 || parser->current.type == TOKEN_I8 ||
        parser->current.type == TOKEN_I16 || parser->current.type == TOKEN_I32 ||
        parser->current.type == TOKEN_I64 || parser->current.type == TOKEN_F32 ||
        parser->current.type == TOKEN_F64) {
        parserAdvance(parser);
    } else {
        errorAt(parser, &parser->current, "Expect type (i1, i8, i16, i32, i64, f32, f64).");
    }
    consume(parser, TOKEN_BANG, "Expect '!'.");
    consume(parser, TOKEN_RBRACKET, "Expect ']' after return type.");

    consume(parser, TOKEN_RBRACKET, "Expect ']' to end args.");

    AstNode *body = parseBlock(parser);
    func->as.func_decl.body = body;
    return func;
}

void freeAst(AstNode *node) {
    if (!node) return;
    if (node->type == AST_BINARY) {
        freeAst(node->as.binary.left);
        freeAst(node->as.binary.right);
    } else if (node->type == AST_ASSIGNMENT) {
        freeAst(node->as.assignment.value);
    } else if (node->type == AST_CALL) {
        for (int i=0; i<node->as.call.arg_count; i++) freeAst(node->as.call.args[i]);
        free(node->as.call.args);
    } else if (node->type == AST_EITHER) {
        freeAst(node->as.either.condition);
        freeAst(node->as.either.true_branch);
        freeAst(node->as.either.false_branch);
    } else if (node->type == AST_BLOCK) {
        for (int i=0; i<node->as.block.count; i++) freeAst(node->as.block.statements[i]);
        free(node->as.block.statements);
    } else if (node->type == AST_FUNC_DECL) {
        if (node->as.func_decl.args) {
            free(node->as.func_decl.args);
        }
        freeAst(node->as.func_decl.body);
    } else if (node->type == AST_IMPORT) {
        // do nothing for imported AST root nodes to avoid double free
    }
    free(node);
}

AstNode* parse(const char *source) {
    Parser parser;
    initLexer(&parser.lexer, source);
    parser.hadError = false;
    parserAdvance(&parser);

    AstNode *block = createNode(AST_BLOCK);
    block->as.block.statements = NULL;
    block->as.block.count = 0;
    while (parser.current.type != TOKEN_EOF) {
        if (parser.current.type == TOKEN_IDENTIFIER && peek(&parser.lexer) == ':') {
             // Handle func decl or assignment at root level
             Token ident = parser.current;
             parserAdvance(&parser);

             const char* next_word = parser.lexer.current;
             while (*next_word == ' ' || *next_word == '\n' || *next_word == '\t' || *next_word == '\r') next_word++;

             if (strncmp(next_word, "func", 4) == 0) {
                  parser.previous = ident;
                  AstNode *func = parseFuncDecl(&parser);
                  block->as.block.count++;
                  block->as.block.statements = (AstNode**)realloc(block->as.block.statements, sizeof(AstNode*) * block->as.block.count);
                  block->as.block.statements[block->as.block.count - 1] = func;
                  continue;
             } else {
                  // Revert ident advance so primary can handle assignment or mlir-op
                  // Hack for MVP, we rely on primary parser doing it properly
                  parser.current = ident;
                  parser.lexer.current = ident.start;
                  parserAdvance(&parser);
             }
        }

        AstNode *stmt = parseStatement(&parser);
        if (stmt) {
             block->as.block.count++;
             block->as.block.statements = (AstNode**)realloc(block->as.block.statements, sizeof(AstNode*) * block->as.block.count);
             block->as.block.statements[block->as.block.count - 1] = stmt;
        }
    }
    return block;
}

void printAst(AstNode *node, int depth) {
    if (!node) return;
    for (int i=0; i<depth; i++) printf("  ");

    switch (node->type) {
        case AST_NUMBER: printf("Number(%d)\n", node->as.number.value); break;
        case AST_STRING: printf("String(%.*s)\n", node->as.string.length, node->as.string.value); break;
        case AST_IDENTIFIER: printf("Ident(%.*s)\n", node->as.identifier.length, node->as.identifier.name); break;
        case AST_BINARY:
            printf("Binary(%d)\n", node->as.binary.op);
            printAst(node->as.binary.left, depth + 1);
            printAst(node->as.binary.right, depth + 1);
            break;
        case AST_CALL:
            printf("Call(%.*s)\n", node->as.call.callee_len, node->as.call.callee);
            for (int i=0; i<node->as.call.arg_count; i++) printAst(node->as.call.args[i], depth + 1);
            break;
        case AST_EITHER:
            printf("Either\n");
            printAst(node->as.either.condition, depth + 1);
            printAst(node->as.either.true_branch, depth + 1);
            printAst(node->as.either.false_branch, depth + 1);
            break;
        case AST_BLOCK:
            printf("Block\n");
            for (int i=0; i<node->as.block.count; i++) printAst(node->as.block.statements[i], depth + 1);
            break;
        case AST_FUNC_DECL:
            printf("FuncDecl(%.*s, args: ", node->as.func_decl.name_len, node->as.func_decl.name);
            for (int i = 0; i < node->as.func_decl.arg_count; ++i) {
                printf("%.*s ", node->as.func_decl.args[i].name_len, node->as.func_decl.args[i].name);
            }
            printf(")\n");
            printAst(node->as.func_decl.body, depth + 1);
            break;
        case AST_MLIR_OP:
            printf("MlirOp(%.*s)\n", node->as.mlir_op.name_len, node->as.mlir_op.name);
            break;
        case AST_IMPORT:
            printf("Import(%.*s)\n", node->as.import_stmt.length, node->as.import_stmt.path);
            if (node->as.import_stmt.module_block) {
                printAst(node->as.import_stmt.module_block, depth + 1);
            }
            break;
        case AST_ASSIGNMENT:
            printf("Assignment(%.*s)\n", node->as.assignment.name_len, node->as.assignment.name);
            printAst(node->as.assignment.value, depth + 1);
            break;
    }
}
