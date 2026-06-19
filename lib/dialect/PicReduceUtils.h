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
inline constexpr const char* kDefaultType = "i64";
inline constexpr const char* kBoolType = "bool";
inline constexpr const char* kF32Type = "f32";
inline constexpr const char* kF64Type = "f64";
inline constexpr const char* kStrType = "str";

// Load known type labels from the module's lin.type_names attribute, falling back
// to kAllLiteralTypes if not present.
inline SmallVector<std::string> getTypeLabels(ModuleOp module) {
    if (auto attr = module->getAttrOfType<StringAttr>("lin.type_names")) {
        SmallVector<std::string> result;
        StringRef val = attr.getValue();
        while (!val.empty()) {
            size_t comma = val.find(',');
            if (comma == StringRef::npos) {
                result.push_back(val.str());
                break;
            }
            result.push_back(val.substr(0, comma).str());
            val = val.substr(comma + 1);
        }
        // Also include the fallback types that aren't typically in type declarations
        // but are needed by the runtime (e.g., "num")
        for (auto lit : kAllLiteralTypes) {
            bool found = false;
            for (auto &r : result) {
                if (r == lit) { found = true; break; }
            }
            if (!found) result.push_back(lit);
        }
        return result;
    }
    SmallVector<std::string> fallback;
    for (auto lit : kAllLiteralTypes) fallback.push_back(lit);
    return fallback;
}

// Check if a type label uses 2 ports (64-bit) vs 1 port (32-bit).
// Known types: "f64", "i64", "num" are 64-bit. Others are guessed by suffix convention.
inline bool is64BitLabel(const std::string& label) {
    if (label == "f64" || label == "i64" || label == "num") return true;
    if (label.size() > 2 && label.substr(label.size() - 2) == "64") return true;
    return false;
}

inline int typeIndex(const std::string& type) {
    if (type == "i32") return 0;
    if (type == "i64") return 1;
    if (type == "f32") return 2;
    if (type == "f64") return 3;
    return -1;
}

inline bool isKnownLiteralType(const std::string& label) {
    for (auto lit : kAllLiteralTypes) {
        if (label == lit) return true;
    }
    return false;
}

inline bool isKnownLiteralType(const std::string& label, ModuleOp module) {
    auto types = getTypeLabels(module);
    for (auto &t : types) {
        if (label == t) return true;
    }
    return isKnownLiteralType(label);
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
            typeName = (suffix == "64") ? kF64Type : kF32Type;
        } else {
            typeName = (suffix == "64") ? kDefaultType : "i32";
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
    return kDefaultType;
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

inline constexpr const char* kTargetGpu = "gpu";
inline constexpr const char* kTargetCpu = "cpu";

struct UserOp {
    uint32_t hash;
    std::string label;
    std::string funcName;
    int numArgs;
    SmallVector<std::string> argTypes;
    SmallVector<std::string> targets;
    std::string forcedTarget;

    UserOp() = default;
    UserOp(uint32_t hash, const std::string& label, const std::string& funcName,
           int numArgs, const SmallVector<std::string>& argTypes,
           const SmallVector<std::string>& targets, const std::string& forcedTarget)
        : hash(hash), label(label), funcName(funcName), numArgs(numArgs),
          argTypes(argTypes), targets(targets), forcedTarget(forcedTarget) {}
};

#endif // PIC_REDUCE_UTILS_H
