#include "Internal.h"

#include "lin/Parser.h"
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

Token makeToken(const Lexer *lexer, TokenType type) {
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
    while (isalpha(peek(lexer)) || isdigit(peek(lexer)) || peek(lexer) == '_' ||
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
                while (isalpha(peek(lexer)) || isdigit(peek(lexer)) || peek(lexer) == '_' ||
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
