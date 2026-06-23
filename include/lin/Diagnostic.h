#ifndef LINALANG_DIAGNOSTIC_H
#define LINALANG_DIAGNOSTIC_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Print a Rust-style error diagnostic with source line and caret pointer.
//   error[E001]: undeclared type 'foat'
//    --> file.lin:5:12
//     |
//   5 |     x: foat
//     |        ^^^^
//     = note: did you mean 'float'?
//
// source: the full source text (used to extract the offending line)
// file:   the file path for display (can be empty/NULL)
// line:   1-based line number
// col:    1-based column number
// msg:    the error message
void diagError(const char *source, const char *file, int line, int col, const char *msg);

// Like diagError but with a suggested fix / help note appended.
void diagErrorWithNote(const char *source, const char *file, int line, int col,
                       const char *msg, const char *noteText);

// Print a note attached to a source location (no "error" prefix).
void diagNote(const char *source, const char *file, int line, int col,
              const char *msg);

// Check if ANSI color output should be used (TTY + NO_COLOR).
bool diagUseColor(void);

// Return ANSI color escape code for the given role.
const char *diagColor(const char *role);

// Extract a single line from source text (0-indexed line number).
// Returns a null-terminated string that must be freed by the caller, or NULL.
char *diagGetLine(const char *source, int lineZero);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_DIAGNOSTIC_H