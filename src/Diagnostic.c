#include "lin/Diagnostic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static bool colorChecked = false;
static bool colorEnabled = false;

bool diagUseColor(void) {
    if (!colorChecked) {
        colorChecked = true;
        const char *noColor = getenv("NO_COLOR");
        if (noColor && noColor[0] != '\0') {
            colorEnabled = false;
        } else {
            colorEnabled = isatty(STDERR_FILENO) || isatty(STDOUT_FILENO);
        }
    }
    return colorEnabled;
}

const char *diagColor(const char *role) {
    if (!diagUseColor()) return "";
    if (strcmp(role, "red") == 0)    return "\033[31m";
    if (strcmp(role, "green") == 0)  return "\033[32m";
    if (strcmp(role, "yellow") == 0) return "\033[33m";
    if (strcmp(role, "cyan") == 0)   return "\033[36m";
    if (strcmp(role, "bold") == 0)   return "\033[1m";
    if (strcmp(role, "reset") == 0)  return "\033[0m";
    return "";
}

char *diagGetLine(const char *source, int lineZero) {
    if (!source) return NULL;
    int currentLine = 0;
    const char *start = source;
    const char *p = source;

    while (*p) {
        if (currentLine == lineZero) {
            start = p;
            while (*p && *p != '\n') p++;
            size_t len = (size_t)(p - start);
            char *result = (char *)malloc(len + 1);
            if (!result) return NULL;
            memcpy(result, start, len);
            result[len] = '\0';
            return result;
        }
        if (*p == '\n') {
            currentLine++;
            if (currentLine > lineZero) break;
        }
        p++;
    }
    return NULL;
}

static void printCaretLine(int lineNum, int col, const char *lineText, bool useColor) {
    const char *reset = useColor ? "\033[0m" : "";

    // Print line number
    if (useColor) fprintf(stderr, "\033[36m");
    fprintf(stderr, "%4d ", lineNum);
    if (useColor) fprintf(stderr, "\033[0m");

    fprintf(stderr, "| ");

    // Print the source line
    if (lineText) {
        fprintf(stderr, "%s", lineText);
    }
    fprintf(stderr, "\n");

    // Print caret line
    fprintf(stderr, "     | ");
    for (int i = 1; i < col; i++) {
        fprintf(stderr, " ");
    }
    if (useColor) fprintf(stderr, "\033[31m\033[1m");
    fprintf(stderr, "^");
    // Print the rest of the token as carets
    if (lineText && col - 1 < (int)strlen(lineText)) {
        int tokenLen = (int)strlen(lineText + (col - 1));
        for (int i = 1; i < tokenLen; i++) {
            char c = lineText[col - 1 + i];
            if (c == ' ' || c == '\t' || c == ')' || c == ']' || c == '}' || c == ',' || c == ';') break;
            fprintf(stderr, "~");
        }
    }
    if (useColor) fprintf(stderr, "\033[0m");
    fprintf(stderr, "\n");
}

void diagError(const char *source, const char *file, int line, int col, const char *msg) {
    bool uc = diagUseColor();
    const char *red = uc ? "\033[31m" : "";
    const char *bold = uc ? "\033[1m" : "";
    const char *reset = uc ? "\033[0m" : "";
    const char *cyan = uc ? "\033[36m" : "";

    // error[E000]: message
    fprintf(stderr, "%s%serror%s%s: %s\n", bold, red, reset, bold, msg);
    //  --> file:line:col
    if (file) {
        fprintf(stderr, "%s%s -->%s %s:%d:%d\n", cyan, bold, reset, file, line, col);
    } else {
        fprintf(stderr, "%s%s -->%s %d:%d\n", cyan, bold, reset, line, col);
    }
    //  |
    fprintf(stderr, " %s |%s\n", cyan, reset);

    char *lineText = NULL;
    if (source) {
        lineText = diagGetLine(source, line - 1);
    }
    printCaretLine(line, col, lineText, uc);
    free(lineText);
}

void diagErrorWithNote(const char *source, const char *file, int line, int col,
                       const char *msg, const char *noteText) {
    diagError(source, file, line, col, msg);
    if (noteText) {
        bool uc = diagUseColor();
        const char *cyan = uc ? "\033[36m" : "";
        const char *reset = uc ? "\033[0m" : "";
        fprintf(stderr, "%s = %s%s%s: %s\n", cyan, reset, cyan, "note", noteText);
    }
}

void diagNote(const char *source, const char *file, int line, int col,
              const char *msg) {
    bool uc = diagUseColor();
    const char *cyan = uc ? "\033[36m" : "";
    const char *bold = uc ? "\033[1m" : "";
    const char *reset = uc ? "\033[0m" : "";

    fprintf(stderr, "%s%s%s%s: %s\n", cyan, bold, "note", reset, msg);
    if (file) {
        fprintf(stderr, "%s%s -->%s %s:%d:%d\n", cyan, bold, reset, file, line, col);
    }
    fprintf(stderr, " %s |%s\n", cyan, reset);

    char *lineText = NULL;
    if (source) {
        lineText = diagGetLine(source, line - 1);
    }
    printCaretLine(line, col, lineText, uc);
    free(lineText);
}