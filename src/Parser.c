#include "lin/Parser.h"
#include "lin/Ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ──────────────── Lexer ────────────────

static bool isAtEnd(Lexer *lexer) { return *lexer->current == '\0'; }
static char advance(Lexer *lexer) {
    char c = *lexer->current++;
    lexer->col++;
    return c;
}
static char peek(Lexer *lexer) { return *lexer->current; }
static char peekNext(Lexer *lexer) {
    if (isAtEnd(lexer)) return '\0';
    return lexer->current[1];
}
static bool match(Lexer *lexer, char expected) {
    if (isAtEnd(lexer)) return false;
    if (*lexer->current != expected) return false;
    lexer->current++;
    lexer->col++;
    return true;
}
static void skipWhitespace(Lexer *lexer) {
    for (;;) {
        char c = peek(lexer);
        switch (c) {
            case ' ': case '\r': case '\t': advance(lexer); break;
            case '\n': lexer->line++; lexer->col = 0; advance(lexer); break;
            case ';':
                while (peek(lexer) != '\n' && !isAtEnd(lexer)) advance(lexer);
                break;
            case '/':
                if (peekNext(lexer) == '/') {
                    while (peek(lexer) != '\n' && !isAtEnd(lexer)) advance(lexer);
                    break;
                } else if (peekNext(lexer) == '*') {
                    advance(lexer); // consume '/'
                    advance(lexer); // consume '*'
                    while (!isAtEnd(lexer)) {
                        if (peek(lexer) == '\n') { lexer->line++; lexer->col = 0; }
                        if (peek(lexer) == '*' && peekNext(lexer) == '/') {
                            advance(lexer); // consume '*'
                            advance(lexer); // consume '/'
                            break;
                        }
                        advance(lexer);
                    }
                    break;
                }
                return;
            default: return;
        }
    }
}

static Token makeToken(const Lexer *lexer, TokenType type) {
    Token token;
    token.type = type;
    token.start = lexer->start;
    token.length = (int)(lexer->current - lexer->start);
    token.line = lexer->line;
    token.col  = lexer->col - token.length; // start column of token
    if (token.col < 1) token.col = 1;
    return token;
}

static Token number(Lexer *lexer) {
    while (isdigit(peek(lexer))) advance(lexer);

    if (peek(lexer) == '.' && isdigit(peekNext(lexer))) {
        advance(lexer); // Consume '.'
        while (isdigit(peek(lexer))) advance(lexer);
        return makeToken(lexer, TOKEN_FLOAT);
    }

    return makeToken(lexer, TOKEN_NUMBER);
}

static TokenType checkKeyword(const Lexer *lexer, int start, int length, const char *rest, TokenType type) {
    if (lexer->current - lexer->start == start + length &&
        memcmp(lexer->start + start, rest, length) == 0) {
        return type;
    }
    return TOKEN_IDENTIFIER;
}

static TokenType identifierType(const Lexer *lexer) {
    switch (lexer->start[0]) {
        case 'f':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'u': return checkKeyword(lexer, 2, 2, "nc", TOKEN_FUNC);
                }
            }
            break;
        case 'i':
            if (lexer->current - lexer->start > 1) {
                switch (lexer->start[1]) {
                    case 'm': return checkKeyword(lexer, 2, 4, "port", TOKEN_IMPORT);
                }
            }
            break;
        case 'e': return checkKeyword(lexer, 1, 5, "ither", TOKEN_EITHER);
        case 'm': return checkKeyword(lexer, 1, 6, "lir-op", TOKEN_MLIR_OP);
        case 'r': return checkKeyword(lexer, 1, 5, "eturn", TOKEN_RETURN);
        case 'w': return checkKeyword(lexer, 1, 4, "hile", TOKEN_WHILE);
    }
    return TOKEN_IDENTIFIER;
}

static Token string(Lexer *lexer) {
    while (peek(lexer) != '"' && !isAtEnd(lexer)) {
        if (peek(lexer) == '\n') lexer->line++;
        advance(lexer);
    }
    if (isAtEnd(lexer)) return makeToken(lexer, TOKEN_ERROR);
    advance(lexer);
    return makeToken(lexer, TOKEN_STRING);
}

static Token identifier(Lexer *lexer) {
    while (isalpha(peek(lexer)) || isdigit(peek(lexer)) || peek(lexer) == '_' || peek(lexer) == '?' ||
           (peek(lexer) == '-' && isalpha(peekNext(lexer)))) {
        advance(lexer);
    }
    return makeToken(lexer, identifierType(lexer));
}

void initLexer(Lexer *lexer, const char *source) {
    lexer->start = source;
    lexer->current = source;
    lexer->line = 1;
    lexer->col  = 1;
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
        case '!': if (match(lexer, '=')) return makeToken(lexer, TOKEN_NOT_EQUAL);
                  return makeToken(lexer, TOKEN_BANG);
        case '{': return makeToken(lexer, TOKEN_LBRACE);
        case '}': return makeToken(lexer, TOKEN_RBRACE);
        case '*': return makeToken(lexer, TOKEN_STAR);
        case '/': return makeToken(lexer, TOKEN_SLASH);
        case '&': if (match(lexer, '&')) return makeToken(lexer, TOKEN_AND); break;
        case '|': if (match(lexer, '|')) return makeToken(lexer, TOKEN_OR); break;
        case '=': if (match(lexer, '=')) return makeToken(lexer, TOKEN_EQUAL_EQUAL); break;
        case '<': if (match(lexer, '=')) return makeToken(lexer, TOKEN_LESS_EQUAL);
                  else if (match(lexer, '<')) return makeToken(lexer, TOKEN_LESS); // << shift
                  return makeToken(lexer, TOKEN_LESS);
        case '>': if (match(lexer, '=')) return makeToken(lexer, TOKEN_GREATER_EQUAL);
                  else if (match(lexer, '>')) return makeToken(lexer, TOKEN_GREATER); // >> shift
                  return makeToken(lexer, TOKEN_GREATER);
        case '+': return makeToken(lexer, TOKEN_PLUS);
        case '-': return makeToken(lexer, TOKEN_MINUS);
        case '%': return makeToken(lexer, TOKEN_IDENTIFIER);
        case '.': return makeToken(lexer, TOKEN_DOT);
        case '@': {
            if (isalpha(peek(lexer)) || peek(lexer) == '_') {
                advance(lexer);
                while (isalpha(peek(lexer)) || isdigit(peek(lexer)) || peek(lexer) == '_' || peek(lexer) == '?' ||
                       (peek(lexer) == '-' && isalpha(peekNext(lexer)))) {
                    advance(lexer);
                }
                return makeToken(lexer, TOKEN_ANNOTATION);
            }
            return makeToken(lexer, TOKEN_ERROR);
        }
        case ',': return makeToken(lexer, TOKEN_IDENTIFIER);
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9': return number(lexer);
    }

    return makeToken(lexer, TOKEN_EOF);
}

typedef struct {
    Lexer lexer;
    Token current;
    Token previous;
    bool hadError;
} Parser;

static void errorAt(Parser *parser, const Token *token, const char *message) {
    if (parser->hadError) return;
    fprintf(stderr, "[line %d:%d] Error", token->line, token->col);
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
        if (parser->current.type != TOKEN_ERROR) break;
        errorAt(parser, &parser->current, "Lexical error: possibly unterminated string.");
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

static AstNode* parseWhile(Parser *parser);

static AstNode* parseEither(Parser *parser);

static AstNode* parseFuncDecl(Parser *parser, bool anonymous);

static AstNode* parseBlock(Parser *parser);

static AstNode* parseFieldAccess(Parser *parser, AstNode *base);

static AstNode* createNode(Parser *parser, AstNodeType type) {
    AstNode *node = (AstNode*)calloc(1, sizeof(AstNode));
    if (!node) {
        error(parser, "Out of memory");
        return NULL;
    }
    node->type = type;
    node->line = parser->current.line;
    node->col  = parser->current.col;
    return node;
}

static AstNode* parseNumberExpr(Parser *parser) {
    AstNode *node = createNode(parser, AST_NUMBER);
    if (!node) return NULL;
    node->as.number.value = atoll(parser->current.start);
    parserAdvance(parser);
    return node;
}

static AstNode* parseFloatExpr(Parser *parser) {
    AstNode *node = createNode(parser, AST_FLOAT);
    if (!node) return NULL;
    node->as.f_number.value = strtod(parser->current.start, NULL);
    parserAdvance(parser);
    return node;
}

static AstNode* parseStringExpr(Parser *parser) {
    AstNode *node = createNode(parser, AST_STRING);
    if (!node) return NULL;
    node->as.string.value = parser->current.start + 1;
    node->as.string.length = parser->current.length - 2;
    parserAdvance(parser);
    return node;
}

static AstNode* parseImportExpr(Parser *parser) {
    parserAdvance(parser);
    consume(parser, TOKEN_STRING, "Expect string path after 'import'.");
    AstNode *node = createNode(parser, AST_IMPORT);
    if (!node) return NULL;
    node->as.import_stmt.path = parser->previous.start + 1;
    node->as.import_stmt.length = parser->previous.length - 2;
    node->as.import_stmt.module_block = NULL;
    return node;
}

static AstNode* parseIdentifierExpr(Parser *parser) {
    Token ident = parser->current;
    parserAdvance(parser);

    // Check for variable assignment: `word: value`
    if (parser->current.type == TOKEN_COLON) {
        const char* next_word = parser->lexer.current;
        while (*next_word == ' ' || *next_word == '\n' || *next_word == '\t' || *next_word == '\r') next_word++;

        bool is_func = (strncmp(next_word, "func", 4) == 0);
        bool is_mlir_op = (strncmp(next_word, "mlir-op", 7) == 0);
        bool is_module = (strncmp(next_word, "module", 6) == 0);

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

            const char *payload_begin = payload_start.start + 1;
            const char *p = payload_begin;
            int braces = 1;
            while (braces > 0 && *p != '\0') {
                if (*p == '{') braces++;
                else if (*p == '}') braces--;
                p++;
            }
            if (braces > 0) {
                errorAt(parser, &payload_start, "Unterminated mlir-op payload.");
                return NULL;
            }
            const char *payload_end_ptr = p - 1; // point to '}'

            // Sync the parser's lexer state to after the '}'
            parser->lexer.current = p;
            parserAdvance(parser); // load the token after '}'

            AstNode *node = createNode(parser, AST_MLIR_OP);
            if (!node) return NULL;
            node->as.mlir_op.name = ident.start;
            node->as.mlir_op.name_len = ident.length;
            node->as.mlir_op.inputs = inputs_outputs_start.start;
            node->as.mlir_op.inputs_len = (int)(inputs_outputs_end.start + inputs_outputs_end.length - inputs_outputs_start.start);
            node->as.mlir_op.mlir_payload = payload_begin;
            node->as.mlir_op.payload_len = (int)(payload_end_ptr - payload_begin);

            while (node->as.mlir_op.payload_len > 0 && (*node->as.mlir_op.mlir_payload == '\n' || *node->as.mlir_op.mlir_payload == '\r')) {
                node->as.mlir_op.mlir_payload++;
                node->as.mlir_op.payload_len--;
            }
            // Also strip trailing whitespace
            while (node->as.mlir_op.payload_len > 0 && (node->as.mlir_op.mlir_payload[node->as.mlir_op.payload_len - 1] == ' ' || node->as.mlir_op.mlir_payload[node->as.mlir_op.payload_len - 1] == '\t' || node->as.mlir_op.mlir_payload[node->as.mlir_op.payload_len - 1] == '\n' || node->as.mlir_op.mlir_payload[node->as.mlir_op.payload_len - 1] == '\r')) {
                node->as.mlir_op.payload_len--;
            }

            return node;
        } else if (is_module) {
            parserAdvance(parser); // consume colon
            // Expect "module" keyword
            if (parser->current.type != TOKEN_IDENTIFIER || 
                strncmp(parser->current.start, "module", 6) != 0) {
                errorAt(parser, &parser->current, "Expect 'module' keyword.");
                return NULL;
            }
            parserAdvance(parser); // consume "module"
            
            // Parse metadata block [ ... ]
            AstNode *module_block = parseBlock(parser);
            
            AstNode *node = createNode(parser, AST_MODULE);
            if (!node) return NULL;
            node->as.module.name = ident.start;
            node->as.module.name_len = ident.length;
            node->as.module.module_block = module_block;
            return node;
        } else if (is_func) {
            return parseFuncDecl(parser, false);
        } else {
            parserAdvance(parser); // consume colon
            AstNode *node = createNode(parser, AST_ASSIGNMENT);
            if (!node) return NULL;
            node->as.assignment.name = ident.start;
            node->as.assignment.name_len = ident.length;
            node->as.assignment.value = parseExpression(parser);
            return node;
        }
    }

    if (ident.length == 4 && strncmp(ident.start, "true", 4) == 0) {
        AstNode *node = createNode(parser, AST_BOOL);
        if (!node) return NULL;
        node->as.boolean.value = true;
        return node;
    }
    if (ident.length == 5 && strncmp(ident.start, "false", 5) == 0) {
        AstNode *node = createNode(parser, AST_BOOL);
        if (!node) return NULL;
        node->as.boolean.value = false;
        return node;
    }

    AstNode *node = createNode(parser, AST_IDENTIFIER);
    if (!node) return NULL;
    node->as.identifier.name = ident.start;
    node->as.identifier.length = ident.length;
    
    AstNode *result = node;
    while (parser->current.type == TOKEN_DOT) {
        result = parseFieldAccess(parser, result);
    }
    
    // Check for path access: identifier/path or identifier/field or identifier/(expr)
    while (parser->current.type == TOKEN_SLASH) {
        parserAdvance(parser); // consume '/'
        Token pathSegment = parser->current;
        
        if (pathSegment.type == TOKEN_NUMBER) {
            // Numeric index: /1, /2, etc.
            parserAdvance(parser);
            AstNode *access = createNode(parser, AST_FIELD_ACCESS);
            if (!access) return result;
            access->as.field_access.base = result;
            access->as.field_access.field_index = atoi(pathSegment.start);
            result = access;
        } else if (pathSegment.type == TOKEN_IDENTIFIER) {
            // Named path segment: /field
            parserAdvance(parser);
            AstNode *access = createNode(parser, AST_FIELD_ACCESS);
            if (!access) return result;
            access->as.field_access.base = result;
            access->as.field_access.field_index = 0; // 0 reserved for named fields
            access->as.field_access.field_name = pathSegment.start;
            access->as.field_access.field_name_len = pathSegment.length;
            result = access;
        } else if (pathSegment.type == TOKEN_LPAREN) {
            // Computed index: /(expression)
            AstNode *idxExpr = parseExpression(parser);
            consume(parser, TOKEN_RPAREN, "Expect ')' after computed index.");
            AstNode *access = createNode(parser, AST_FIELD_ACCESS);
            if (!access) return result;
            access->as.field_access.base = result;
            access->as.field_access.computed_index = idxExpr;
            access->as.field_access.field_index = -1; // sentinel for computed
            result = access;
        } else {
            // Not a valid path segment, just return what we have
            break;
        }
    }
    
    return result;
}

static AstNode* parseGroupingExpr(Parser *parser) {
    parserAdvance(parser);
    if (parser->current.type == TOKEN_IDENTIFIER) {
        Token savedIdent = parser->current;
        
        // Peek at what follows to decide: call or grouped expression with postfix ops.
        // If followed by `/`, `.`, `)`, then it's a grouping (path access, field access, or bare identifier).
        // If followed by anything else (expressions), it's a function call.
        const char *p = parser->lexer.current;
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        bool isGroupingExpr = (*p == '/' || *p == '.' || *p == ']' || *p == ',');

        if (isGroupingExpr) {
            AstNode *expr = parseExpression(parser);
            consume(parser, TOKEN_RPAREN, "Expect ')' after expression.");
            
            while (parser->current.type == TOKEN_DOT) {
                expr = parseFieldAccess(parser, expr);
            }
            while (parser->current.type == TOKEN_SLASH) {
                parserAdvance(parser);
                Token pathSegment = parser->current;
                if (pathSegment.type == TOKEN_NUMBER || pathSegment.type == TOKEN_IDENTIFIER) {
                    parserAdvance(parser);
                    AstNode *access = createNode(parser, AST_FIELD_ACCESS);
                    if (!access) return expr;
                    access->as.field_access.base = expr;
                    if (pathSegment.type == TOKEN_NUMBER) {
                        access->as.field_access.field_index = atoi(pathSegment.start) - 1;
                    } else {
                        access->as.field_access.field_index = 0;
                        access->as.field_access.field_name = pathSegment.start;
                        access->as.field_access.field_name_len = pathSegment.length;
                    }
                    expr = access;
                } else {
                    break;
                }
            }
            return expr;
        } else {
            AstNode *call = createNode(parser, AST_CALL);
            if (!call) return NULL;
            call->as.call.callee = parser->current.start;
            call->as.call.callee_len = parser->current.length;
            parserAdvance(parser);

            call->as.call.args = NULL;
            call->as.call.arg_count = 0;
            call->as.call.capacity = 0;
            call->as.call.resolved_callee = NULL;
            while (parser->current.type != TOKEN_RPAREN && parser->current.type != TOKEN_EOF) {
                AstNode *arg = parseExpression(parser);
                if (call->as.call.arg_count >= call->as.call.capacity) {
                    call->as.call.capacity = call->as.call.capacity < 8 ? 8 : call->as.call.capacity * 2;
                    void *tmp = realloc(call->as.call.args, sizeof(AstNode*) * call->as.call.capacity);
                    if (!tmp) {
                        error(parser, "Out of memory");
                        freeAst(call);
                        return NULL;
                    }
                    call->as.call.args = (AstNode**)tmp;
                }
                call->as.call.args[call->as.call.arg_count++] = arg;
            }
            consume(parser, TOKEN_RPAREN, "Expect ')' after arguments.");

            AstNode *result = call;
            while (parser->current.type == TOKEN_DOT) {
                result = parseFieldAccess(parser, result);
            }
            while (parser->current.type == TOKEN_SLASH) {
                parserAdvance(parser);
                Token pathSegment = parser->current;
                if (pathSegment.type == TOKEN_NUMBER || pathSegment.type == TOKEN_IDENTIFIER) {
                    parserAdvance(parser);
                    AstNode *access = createNode(parser, AST_FIELD_ACCESS);
                    if (!access) return result;
                    access->as.field_access.base = result;
                    if (pathSegment.type == TOKEN_NUMBER) {
                        access->as.field_access.field_index = atoi(pathSegment.start) - 1;
                    } else {
                        access->as.field_access.field_index = 0;
                        access->as.field_access.field_name = pathSegment.start;
                        access->as.field_access.field_name_len = pathSegment.length;
                    }
                    result = access;
                } else {
                    break;
                }
            }
            return result;
        }
    } else {
        AstNode *expr = parseExpression(parser);
        consume(parser, TOKEN_RPAREN, "Expect ')' after expression.");
        
        while (parser->current.type == TOKEN_DOT) {
            expr = parseFieldAccess(parser, expr);
        }
        while (parser->current.type == TOKEN_SLASH) {
            parserAdvance(parser);
            Token pathSegment = parser->current;
            if (pathSegment.type == TOKEN_NUMBER || pathSegment.type == TOKEN_IDENTIFIER) {
                parserAdvance(parser);
                AstNode *access = createNode(parser, AST_FIELD_ACCESS);
                if (!access) return expr;
                access->as.field_access.base = expr;
                if (pathSegment.type == TOKEN_NUMBER) {
                    access->as.field_access.field_index = atoi(pathSegment.start) - 1;
                } else {
                    access->as.field_access.field_index = 0;
                    access->as.field_access.field_name = pathSegment.start;
                    access->as.field_access.field_name_len = pathSegment.length;
                }
                expr = access;
            } else {
                break;
            }
        }
        return expr;
    }
}

// Parse field access: identifier . 0 | identifier . 1
static AstNode* parseFieldAccess(Parser *parser, AstNode *base) {
    consume(parser, TOKEN_DOT, "Expect '.' after identifier for field access.");
    
    Token field = parser->current;
    if (field.type != TOKEN_NUMBER) {
        error(parser, "Expect field index (0 or 1) after '.'.");
        return base;
    }
    parserAdvance(parser);
    
    AstNode *node = createNode(parser, AST_FIELD_ACCESS);
    node->as.field_access.base = base;
    node->as.field_access.field_index = atoi(field.start);
    return node;
}

static AstNode* parsePrimary(Parser *parser) {
    switch (parser->current.type) {
        case TOKEN_NUMBER:     return parseNumberExpr(parser);
        case TOKEN_FLOAT:      return parseFloatExpr(parser);
        case TOKEN_STRING:     return parseStringExpr(parser);
        case TOKEN_IMPORT:     return parseImportExpr(parser);
        case TOKEN_IDENTIFIER:
            return parseIdentifierExpr(parser);
        case TOKEN_LPAREN:     return parseGroupingExpr(parser);
        case TOKEN_WHILE:      return parseWhile(parser);
        case TOKEN_EITHER:     return parseEither(parser);
        case TOKEN_FUNC:       return parseFuncDecl(parser, true);
        case TOKEN_LBRACKET:   return parseBlock(parser);
        case TOKEN_MINUS: {
            parserAdvance(parser); // consume '-'
            if (parser->current.type == TOKEN_NUMBER) {
                AstNode *node = createNode(parser, AST_NUMBER);
                if (!node) return NULL;
                node->as.number.value = -atoll(parser->current.start);
                parserAdvance(parser);
                return node;
            } else if (parser->current.type == TOKEN_FLOAT) {
                AstNode *node = createNode(parser, AST_FLOAT);
                if (!node) return NULL;
                node->as.f_number.value = -strtod(parser->current.start, NULL);
                parserAdvance(parser);
                return node;
            } else {
                error(parser, "Expect number or float after unary '-'.");
                return NULL;
            }
        }
        default:
            error(parser, "Expect expression.");
            parserAdvance(parser);
            return NULL;
    }
}

static AstNode* parseExpression(Parser *parser) {
    AstNode *expr = parsePrimary(parser);

    while (parser->current.type == TOKEN_LESS || parser->current.type == TOKEN_PLUS || parser->current.type == TOKEN_MINUS ||
           parser->current.type == TOKEN_STAR || parser->current.type == TOKEN_GREATER ||
           parser->current.type == TOKEN_EQUAL_EQUAL || parser->current.type == TOKEN_NOT_EQUAL ||
           parser->current.type == TOKEN_GREATER_EQUAL || parser->current.type == TOKEN_LESS_EQUAL ||
           parser->current.type == TOKEN_AND || parser->current.type == TOKEN_OR) {
        TokenType op = parser->current.type;
        parserAdvance(parser);
        AstNode *right = parsePrimary(parser);

        const char* funcName = "add";
        if (op == TOKEN_MINUS) funcName = "sub";
        else if (op == TOKEN_STAR) funcName = "mul";
        else if (op == TOKEN_SLASH) funcName = "div";
        else if (op == TOKEN_LESS) funcName = "lt";
        else if (op == TOKEN_GREATER) funcName = "gt";
        else if (op == TOKEN_LESS_EQUAL) funcName = "le";
        else if (op == TOKEN_GREATER_EQUAL) funcName = "ge";
        else if (op == TOKEN_EQUAL_EQUAL) funcName = "eq";
        else if (op == TOKEN_NOT_EQUAL) funcName = "ne";
        else if (op == TOKEN_AND) funcName = "and";
        else if (op == TOKEN_OR) funcName = "std.or";

        AstNode *call = createNode(parser, AST_CALL);
        call->as.call.callee = strdup(funcName);
        call->as.call.callee_len = strlen(funcName);
        call->as.call.arg_count = 2;
        call->as.call.resolved_callee = NULL;
        call->as.call.args = malloc(sizeof(AstNode*) * 2);
        call->as.call.args[0] = expr;
        call->as.call.args[1] = right;
        expr = call;
    }

    // Post-expression path access: expr/path
    while (parser->current.type == TOKEN_SLASH) {
        parserAdvance(parser); // consume '/'
        Token pathSegment = parser->current;
        
        if (pathSegment.type == TOKEN_NUMBER || pathSegment.type == TOKEN_IDENTIFIER) {
            parserAdvance(parser);
            AstNode *access = createNode(parser, AST_FIELD_ACCESS);
            if (!access) return expr;
            access->as.field_access.base = expr;
            if (pathSegment.type == TOKEN_NUMBER) {
                access->as.field_access.field_index = atoi(pathSegment.start) - 1; // convert to 0-indexed
            } else {
                access->as.field_access.field_index = 0;
                access->as.field_access.field_name = pathSegment.start;
                access->as.field_access.field_name_len = pathSegment.length;
            }
            expr = access;
        } else {
            break;
        }
    }

    return expr;
}

static AstNode* parseBlock(Parser *parser) {
    consume(parser, TOKEN_LBRACKET, "Expect '[' before block.");
    
    AstNode *block = createNode(parser, AST_BLOCK);
    if (!block) return NULL;
    block->as.block.statements = NULL;
    block->as.block.count = 0;
    block->as.block.capacity = 0;

    while (parser->current.type != TOKEN_RBRACKET && parser->current.type != TOKEN_EOF) {
        AstNode *stmt = parseStatement(parser);
        if (block->as.block.count >= block->as.block.capacity) {
            block->as.block.capacity = block->as.block.capacity < 8 ? 8 : block->as.block.capacity * 2;
            void *tmp = realloc(block->as.block.statements, sizeof(AstNode*) * block->as.block.capacity);
            if (!tmp) {
                error(parser, "Out of memory");
                freeAst(block);
                return NULL;
            }
            block->as.block.statements = (AstNode**)tmp;
        }
        block->as.block.statements[block->as.block.count++] = stmt;
    }

    consume(parser, TOKEN_RBRACKET, "Expect ']' after block.");
    return block;
}

static AstNode* parseWhile(Parser *parser) {
    parserAdvance(parser);
    AstNode *while_node = createNode(parser, AST_WHILE);
    if (!while_node) return NULL;
    if (parser->current.type == TOKEN_LBRACKET) {
        while_node->as.while_loop.condition = parseBlock(parser);
    } else {
        while_node->as.while_loop.condition = parseExpression(parser);
    }
    while_node->as.while_loop.body = parseBlock(parser);
    return while_node;
}

static AstNode* parseEither(Parser *parser) {
    parserAdvance(parser);
    AstNode *call = createNode(parser, AST_CALL);
    if (!call) return NULL;
    call->as.call.callee = "either";
    call->as.call.callee_len = 6;
    call->as.call.arg_count = 2;
    call->as.call.capacity = 2;
    call->as.call.resolved_callee = NULL;
    call->as.call.args = malloc(sizeof(AstNode*) * 2);
    if (!call->as.call.args) {
        error(parser, "Out of memory");
        freeAst(call);
        return NULL;
    }

    if (parser->current.type == TOKEN_LBRACKET) {
        call->as.call.args[0] = parseBlock(parser);
    } else {
        call->as.call.args[0] = parseExpression(parser);
    }

    AstNode *pair = createNode(parser, AST_PAIR);
    if (!pair) {
        freeAst(call);
        return NULL;
    }
    pair->as.pair.left = parseBlock(parser);
    if (parser->current.type == TOKEN_LBRACKET) {
        pair->as.pair.right = parseBlock(parser);
    } else {
        AstNode *empty = createNode(parser, AST_BLOCK);
        if (!empty) {
            freeAst(call);
            freeAst(pair);
            return NULL;
        }
        empty->as.block.statements = NULL;
        empty->as.block.count = 0;
        empty->as.block.capacity = 0;
        pair->as.pair.right = empty;
    }
    call->as.call.args[1] = pair;

    return call;
}

static AstNode* parseStatement(Parser *parser) {
    if (parser->current.type == TOKEN_ANNOTATION) {
        Token ann = parser->current;
        parserAdvance(parser); // consume annotation
        AstNode *node = parseExpression(parser);
        if (node) {
            if (node->type == AST_FUNC_DECL) {
                node->as.func_decl.dispatch = ann.start + 1; // skip '@'
                node->as.func_decl.dispatch_len = ann.length - 1;
            } else if (node->type == AST_MLIR_OP) {
                node->as.mlir_op.dispatch = ann.start + 1; // skip '@'
                node->as.mlir_op.dispatch_len = ann.length - 1;
            }
        }
        return node;
    }
    return parseExpression(parser);
}

static AstNode* parseFuncDecl(Parser *parser, bool anonymous) {
    Token name;
    if (!anonymous) {
        name = parser->previous;
        consume(parser, TOKEN_COLON, "Expect ':' after function name.");
    } else {
        name.start = "";
        name.length = 0;
    }
    consume(parser, TOKEN_FUNC, "Expect 'func'.");
    consume(parser, TOKEN_LBRACKET, "Expect '[' for args.");

    int arg_capacity = 4;
    int arg_count = 0;
    struct AstFuncArg *args = malloc(sizeof(*args) * arg_capacity);
    if (!args) {
        error(parser, "Out of memory");
        return NULL;
    }

    while (parser->current.type != TOKEN_RETURN && parser->current.type != TOKEN_EOF && parser->current.type != TOKEN_RBRACKET) {
        Token argName = parser->current;
        consume(parser, TOKEN_IDENTIFIER, "Expect argument name.");
        consume(parser, TOKEN_LBRACKET, "Expect '[' for arg type.");
        Token typeName = parser->current;
        if (parser->current.type == TOKEN_IDENTIFIER) {
            parserAdvance(parser);
        } else {
            errorAt(parser, &parser->current, "Expect type identifier.");
        }
        consume(parser, TOKEN_BANG, "Expect '!'.");
        consume(parser, TOKEN_RBRACKET, "Expect ']' after arg type.");

        if (arg_count >= arg_capacity) {
            arg_capacity *= 2;
            void *tmp = realloc(args, sizeof(*args) * arg_capacity);
            if (!tmp) {
                error(parser, "Out of memory");
                free(args);
                return NULL;
            }
            args = tmp;
        }
        args[arg_count].name = argName.start;
        args[arg_count].name_len = argName.length;
        args[arg_count].type_name = typeName.start;
        args[arg_count].type_name_len = typeName.length;
        arg_count++;
    }

    bool hasReturn = false;
    if (parser->current.type == TOKEN_RETURN) {
        hasReturn = true;
        parserAdvance(parser); // consume 'return'
        consume(parser, TOKEN_COLON, "Expect ':' after return.");
    }

    Token returnTypeName;
    returnTypeName.start = "i32"; // Default
    returnTypeName.length = 3;

    if (hasReturn) {
        consume(parser, TOKEN_LBRACKET, "Expect '[' for return type.");
        returnTypeName = parser->current;
        if (parser->current.type == TOKEN_IDENTIFIER) {
            parserAdvance(parser);
            consume(parser, TOKEN_BANG, "Expect '!'.");
        } else {
            errorAt(parser, &parser->current, "Expect return type identifier.");
        }
        consume(parser, TOKEN_RBRACKET, "Expect ']' after return type.");
    }
    
    consume(parser, TOKEN_RBRACKET, "Expect ']' to close argument list.");
    
    AstNode *body = parseBlock(parser);

    AstNode *func = createNode(parser, AST_FUNC_DECL);
    if (!func) {
        free(args);
        return NULL;
    }
    func->as.func_decl.name = name.start;
    func->as.func_decl.name_len = name.length;
    func->as.func_decl.args = args;
    func->as.func_decl.arg_count = arg_count;
    func->as.func_decl.arg_capacity = arg_capacity;
    func->as.func_decl.body = body;
    func->as.func_decl.return_type_name = returnTypeName.start;
    func->as.func_decl.return_type_len = returnTypeName.length;
    return func;
}


AstNode* parse(const char *source) {
    Parser parser;
    initLexer(&parser.lexer, source);
    parser.hadError = false;
    parser.current = makeToken(&parser.lexer, TOKEN_EOF);



    parserAdvance(&parser);

    AstNode *block = createNode(&parser, AST_BLOCK);
    if (!block) return NULL;
    block->as.block.statements = NULL;
    block->as.block.count = 0;
    block->as.block.capacity = 0;

    if (parser.hadError) {
        freeAst(block);
        return NULL;
    }

    while (parser.current.type != TOKEN_EOF) {
        AstNode *stmt = parseStatement(&parser);
        if (stmt) {
             if (block->as.block.count >= block->as.block.capacity) {
                 block->as.block.capacity = block->as.block.capacity < 8 ? 8 : block->as.block.capacity * 2;
                 void *tmp = realloc(block->as.block.statements, sizeof(AstNode*) * block->as.block.capacity);
                 if (!tmp) {
                     error(&parser, "Out of memory");
                     freeAst(block);
                     return NULL;
                 }
                 block->as.block.statements = (AstNode**)tmp;
             }
             block->as.block.statements[block->as.block.count++] = stmt;
        }
    }
    if (parser.hadError) {
        freeAst(block);
        return NULL;
    }
    return block;
}


