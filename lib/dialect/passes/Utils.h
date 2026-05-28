#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#endif
#include "mlir/Pass/Pass.h"
#include <set>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include <sstream>
#include <unordered_map>

using namespace mlir;

namespace {

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

enum PicrNodeType {
    NODE_CON = 1,
    NODE_DES = 2,
    NODE_DUP = 3,
    NODE_ERA = 4,
    NODE_OP  = 5,
    NODE_HIS = 6,
    NODE_LOG = 7,
    NODE_RVEC = 8
};
