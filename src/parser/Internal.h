#pragma once

#include "lin/Parser.h"

#ifdef __cplusplus
extern "C" {
#endif


Token makeToken(const Lexer *lexer, TokenType type);
void initLexer(Lexer *lexer, const char *source);
Token scanToken(Lexer *lexer);

#ifdef __cplusplus
}
#endif
