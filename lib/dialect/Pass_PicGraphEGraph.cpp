#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

using namespace mlir;
using namespace mlir::pic::graph;

namespace {

struct AgentKey {
  StringRef agentType;
  StringRef label;
  std::optional<uint64_t> value;
  std::optional<StringRef> strVal;
  StringRef polarity;

  bool operator==(const AgentKey &o) const {
    return agentType == o.agentType && label == o.label &&
           value == o.value && polarity == o.polarity &&
           strVal == o.strVal;
  }
};

struct AgentKeyInfo : public llvm::DenseMapInfo<AgentKey> {
  static inline AgentKey getEmptyKey() {
    return {StringRef("~EMPTY~"), StringRef(), std::nullopt, std::nullopt, StringRef()};
  }
  static inline AgentKey getTombstoneKey() {
    return {StringRef("~TOMB~"), StringRef(), std::nullopt, std::nullopt, StringRef()};
  }
static unsigned getHashValue(const AgentKey &k) {
      unsigned h = 0;
      h = (h ^ llvm::hash_combine_range(k.agentType.begin(), k.agentType.end())) * 16777619u;
      h = (h ^ llvm::hash_combine_range(k.label.begin(), k.label.end())) * 16777619u;
      h = (h ^ llvm::hash_combine_range(k.polarity.begin(), k.polarity.end())) * 16777619u;
      if (k.value.has_value())
        h = (h ^ (unsigned)(k.value.value() * 0x9e3779b9)) * 16777619u;
      return h;
    }
  static bool isEqual(const AgentKey &a, const AgentKey &b) {
    if (&a == &b) return true;
    return a == b;
  }
};

using AgentMap = DenseMap<AgentKey, AgentOp, AgentKeyInfo>;

struct PicGraphEGraphPass : public PassWrapper<PicGraphEGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicGraphEGraphPass)

  StringRef getArgument() const override { return "pic-graph-egraph"; }
  StringRef getDescription() const override {
    return "E-graph congruence (Rule 4): merges identical subgraphs inside boundaries";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      funcOp.walk([&](BoundaryOp boundaryOp) {
        Region &bodyRegion = boundaryOp.getBody();
        if (!bodyRegion.empty())
          processBlock(&bodyRegion.front());
      });
    });
  }

private:
  void processBlock(Block *block) {
    if (!block) return;

    AgentMap seen;
    SmallVector<std::pair<AgentOp, AgentOp>> merges;

    for (auto &op : *block) {
      if (auto agentOp = dyn_cast<AgentOp>(&op)) {
        if (agentOp.getLabel() == "era" || agentOp.getLabel() == "dup")
          continue;
        if (agentOp.getAgentType() != "omega")
          continue;

        AgentKey key{agentOp.getAgentType(), agentOp.getLabel(),
                     agentOp.getValue(), agentOp.getStrVal(),
                     agentOp.getPolarity()};
        auto it = seen.find(key);
        if (it != seen.end()) {
          merges.push_back({it->second, agentOp});
        } else {
          seen[key] = agentOp;
        }
      }
    }

    if (merges.empty()) return;

    MLIRContext *ctx = block->getParentOp()->getContext();
    OpBuilder builder(ctx);
    Type portType = PortType::get(ctx);

    for (auto &[existing, duplicate] : merges) {
      Location loc = duplicate.getLoc();

      auto dupNode = builder.create<AgentOp>(loc, portType, portType, portType,
          builder.getStringAttr("delta"), builder.getStringAttr("*"),
          builder.getStringAttr("dup"),
          IntegerAttr{}, StringAttr{}, StringAttr{});

      Value existingP0 = existing.getP0();
      Value dupP0 = dupNode.getResult(0);
      Value dupP1 = dupNode.getResult(1);
      Value dupP2 = dupNode.getResult(2);

      Value consumerOfExisting;
      Value consumerOfDuplicate;

      for (auto &userOp : *block) {
        if (auto linkOp = dyn_cast<LinkOp>(&userOp)) {
          if (linkOp.getOperand(0) == existingP0)
            consumerOfExisting = linkOp.getOperand(1);
          else if (linkOp.getOperand(1) == existingP0)
            consumerOfExisting = linkOp.getOperand(0);

          if (linkOp.getOperand(0) == duplicate.getP0())
            consumerOfDuplicate = linkOp.getOperand(1);
          else if (linkOp.getOperand(1) == duplicate.getP0())
            consumerOfDuplicate = linkOp.getOperand(0);
        }
      }

      if (consumerOfExisting && consumerOfDuplicate) {
        SmallVector<Operation *> toErase;
        for (auto &userOp : *block) {
          if (auto linkOp = dyn_cast<LinkOp>(&userOp)) {
            bool hits = (linkOp.getOperand(0) == existingP0 ||
                         linkOp.getOperand(1) == existingP0) ||
                        (linkOp.getOperand(0) == duplicate.getP0() ||
                         linkOp.getOperand(1) == duplicate.getP0());
            if (hits) toErase.push_back(linkOp);
          }
        }
        for (auto *op : toErase) op->erase();

        builder.create<LinkOp>(loc, existingP0, dupP0);
        builder.create<LinkOp>(loc, dupP1, consumerOfExisting);
        builder.create<LinkOp>(loc, dupP2, consumerOfDuplicate);

        // Track merged origins for reversibility
        int originCount = 1;
        if (auto existingAttr = existing->getAttr("origins")) {
          if (auto intAttr = existingAttr.dyn_cast<IntegerAttr>())
            originCount += intAttr.getInt();
        }
        existing->setAttr("origins", builder.getI64IntegerAttr(originCount));

        duplicate.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPicGraphEGraphPass() {
  return std::make_unique<PicGraphEGraphPass>();
}