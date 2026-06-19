#ifndef PIC_REDUCE_UTILS_H
#define PIC_REDUCE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/ADT/StringRef.h"
#include "PicRuntimeDialect.h"
#include <string>
#include <vector>
#include <array>

using namespace mlir;

inline constexpr const char* kAllLiteralTypes[] = {"num", "i1", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "str"};
inline constexpr size_t kNumLiteralTypes = 10;

inline bool isKnownLiteralType(const std::string& label) {
    for (auto lit : kAllLiteralTypes) {
        if (label == lit) return true;
    }
    return false;
}

inline bool is64BitTypeLabel(const std::string& label) {
    return label == "f64" || label == "i64" || label == "num";
}

inline bool isValidMLIRType(const std::string& type) {
    return type == "i1" || type == "i8" || type == "i16" || type == "i32" || type == "i64" || type == "f32" || type == "f64" || type == "ptr";
}

static bool suffixToTypeName(const std::string& label, std::string& opName, std::string& typeName, std::string& originalBase) {
    std::string suffix = "";
    if (label.size() > 2) {
        suffix = label.substr(label.size() - 2);
    }
    if (suffix == "64" || suffix == "32") {
        std::string base = label.substr(0, label.size() - 2);
        originalBase = base;
        bool isFloat = (base[0] == 'f');
        if (isFloat) {
            base = base.substr(1);
            typeName = (suffix == "64") ? "f64" : "f32";
        } else {
            typeName = (suffix == "64") ? "i64" : "i32";
        }
        if (base == "divs" || base == "divu") opName = "div";
        else if (base == "rems" || base == "remu") opName = "rem";
        else if (base == "slt") opName = "lt";
        else if (base == "sgt") opName = "gt";
        else if (base == "sle") opName = "le";
        else if (base == "sge") opName = "ge";
        else if (base == "neq") opName = "ne";
        else opName = base;
        return true;
    }
    return false;
}

static Value safeZExt(OpBuilder &b, Location loc, Type targetType, Value val) {
    if (!val || !val.getType().isa<IntegerType>()) return val;
    unsigned srcW = val.getType().cast<IntegerType>().getWidth();
    unsigned tgtW = targetType.cast<IntegerType>().getWidth();
    if (srcW < tgtW)
        return b.create<LLVM::ZExtOp>(loc, targetType, val);
    if (srcW > tgtW)
        return b.create<LLVM::TruncOp>(loc, targetType, val);
    return val;
}

static void replaceAll(std::string &str, const std::string &from, const std::string &to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

static std::string extractType(const std::string& payload, const std::string& varName) {
    // ... (same content as original)
    size_t pos = 0;
    std::string targetLine = "";
    bool foundVar = false;
    while (true) {
        pos = payload.find(varName, pos);
        if (pos == std::string::npos) break;
        size_t next = pos + varName.size();
        while (next < payload.size() && std::isspace(static_cast<unsigned char>(payload[next]))) next++;
        if (next < payload.size() && payload[next] == '=') {
            size_t endOfLine = payload.find('\n', pos);
            if (endOfLine == std::string::npos) endOfLine = payload.size();
            targetLine = payload.substr(pos, endOfLine - pos);
            foundVar = true;
            break;
        }
        pos += varName.size();
    }
    if (!foundVar) {
        size_t lastLinePos = payload.find_last_not_of(" \t\r\n");
        if (lastLinePos != std::string::npos) {
            size_t startOfLastLine = payload.find_last_of("\n", lastLinePos);
            if (startOfLastLine == std::string::npos) startOfLastLine = 0;
            else startOfLastLine++;
            targetLine = payload.substr(startOfLastLine, lastLinePos - startOfLastLine + 1);
        } else {
            targetLine = payload;
        }
    }
    size_t colon = targetLine.rfind(':');
    if (colon != std::string::npos) {
        std::string typePart = targetLine.substr(colon + 1);
        typePart.erase(0, typePart.find_first_not_of(" \t\r\n"));
        typePart.erase(typePart.find_last_not_of(" \t\r\n") + 1);
        size_t toPos = typePart.rfind(" to ");
        if (toPos != std::string::npos) {
            std::string t = typePart.substr(toPos + 4);
            t.erase(0, t.find_first_not_of(" \t\r\n"));
            t.erase(t.find_last_not_of(" \t\r\n") + 1);
            return t;
        }
        size_t arrowPos = typePart.rfind("->");
        if (arrowPos != std::string::npos) {
            std::string t = typePart.substr(arrowPos + 2);
            t.erase(0, t.find_first_not_of(" \t\r\n"));
            t.erase(t.find_last_not_of(" \t\r\n") + 1);
            return t;
        }
        if (targetLine.find("cmpi") != std::string::npos ||
            targetLine.find("cmpf") != std::string::npos ||
            targetLine.find("icmp") != std::string::npos ||
            targetLine.find("fcmp") != std::string::npos) {
            return "i1";
        }
        return typePart;
    }
    return "i64";
}

static uint32_t opcodeForLabel(StringRef label) {
  uint32_t hash = 2166136261u;
  for (unsigned char c : label) {
    hash ^= c;
    hash *= 16777619u;
  }
  return (hash & 0xFFFFFF) ? (hash & 0xFFFFFF) : 1u;
}

enum class TargetBackend {
    CPU,
    GPU
};

struct UserOp {
    uint32_t hash;
    std::string label;
    std::string funcName;
    int numArgs;
    SmallVector<std::string> argTypes;
};

#endif // PIC_REDUCE_UTILS_H
