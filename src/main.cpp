#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include <sstream>
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#if __has_include("mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h")
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#if __has_include("mlir/Dialect/SPIRV/Transforms/Passes.h")
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#endif
#if __has_include("mlir/Conversion/ArithToSPIRV/ArithToSPIRVPass.h")
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/IndexToSPIRV/IndexToSPIRVPass.h")
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h")
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h")
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#endif
#endif
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#endif
#if __has_include("mlir/Target/SPIRV/Serialization.h")
#include "mlir/Target/SPIRV/Serialization.h"
#endif
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "lin/LoweringPasses.h"

#include "lin/Semantic.h"
#include "lin/Ast.h"

extern "C" {
#include "lin/Parser.h"
#include "lin/Lowering.h"
}

using namespace mlir;

#include <sstream>
#include <filesystem>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <cctype>
#include <linux/limits.h>
#include <fstream>

#define LINC_VERSION "0.1.0"
#define LINC_EDITION "2025"

// ── Color / Terminal support ────────────────────────────────────────────────

enum Color { C_RESET, C_RED, C_GREEN, C_YELLOW, C_CYAN, C_BOLD };

static bool useColor() {
    static bool checked = false;
    static bool color = false;
    if (!checked) {
        checked = true;
        const char *noColor = getenv("NO_COLOR");
        if (noColor && noColor[0] != '\0') {
            color = false;
        } else {
            color = isatty(STDERR_FILENO) || isatty(STDOUT_FILENO);
        }
    }
    return color;
}

static const char *ansiColor(Color c) {
    if (!useColor()) return "";
    switch (c) {
        case C_RED:    return "\033[31m";
        case C_GREEN:  return "\033[32m";
        case C_YELLOW: return "\033[33m";
        case C_CYAN:   return "\033[36m";
        case C_BOLD:   return "\033[1m";
        default:       return "\033[0m";
    }
}

// ── Diagnostic helpers ──────────────────────────────────────────────────────

static void err(const std::string &msg) {
    std::cerr << ansiColor(C_RED) << "error" << ansiColor(C_RESET) << ": " << msg << "\n";
}

static void warn(const std::string &msg) {
    std::cerr << ansiColor(C_YELLOW) << "warning" << ansiColor(C_RESET) << ": " << msg << "\n";
}

static void note(const std::string &msg) {
    std::cerr << ansiColor(C_CYAN) << "note" << ansiColor(C_RESET) << ": " << msg << "\n";
}

static void green(const std::string &msg) {
    std::cout << ansiColor(C_GREEN) << msg << ansiColor(C_RESET) << "\n";
}

// ── Lin.toml project config ─────────────────────────────────────────────────

struct ProjectConfig {
    std::string name;
    std::string version = "0.1.0";
    std::string edition = LINC_EDITION;
    std::vector<std::string> includes;

    bool valid = false;
};

static std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static ProjectConfig parseProjectConfig(const std::string &path) {
    ProjectConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) return cfg;

    std::string line;
    std::string section;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        if (t[0] == '[') {
            size_t close = t.find(']');
            if (close != std::string::npos)
                section = t.substr(1, close - 1);
            continue;
        }

        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));

        if (val.size() >= 2 && val[0] == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if (section == "project") {
            if (key == "name")    cfg.name = val;
            if (key == "version") cfg.version = val;
            if (key == "edition") cfg.edition = val;
        }
        if (section == "compiler") {
            if (key == "includes") {
                size_t s = val.find('['), e = val.find(']');
                if (s != std::string::npos && e != std::string::npos) {
                    std::string inner = val.substr(s + 1, e - s - 1);
                    std::stringstream ss(inner);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        std::string ii = trim(item);
                        if (!ii.empty()) {
                            if (ii.size() >= 2 && ii[0] == '"' && ii.back() == '"')
                                ii = ii.substr(1, ii.size() - 2);
                            cfg.includes.push_back(ii);
                        }
                    }
                }
            }
        }
    }

    if (!cfg.name.empty()) cfg.valid = true;
    return cfg;
}

// ── File I/O helpers ────────────────────────────────────────────────────────

static bool writeFile(const std::string &path, const std::string &content) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << content;
    return true;
}

static bool mkdirP(const std::string &path) {
    std::error_code ec;
    return std::filesystem::create_directories(path, ec);
}

// ── Template for `linc new` ─────────────────────────────────────────────────

static const char *TEMPLATE_MAIN = R"(main: func [
    return: [i32!]
][
    print "Hello from {{name}}!\n"
    0
]
)";

static const char *TEMPLATE_GITIGNORE = R"(/build/
*.o
*.line
*.spv
result/
)";

static const char *TEMPLATE_TOML = R"([project]
name = "{{name}}"
version = "0.1.0"
edition = "{{edition}}"

[compiler]
includes = ["std"]
)";

static std::string replacePlaceholder(const std::string &s, const std::string &key, const std::string &val) {
    std::string r = s;
    size_t pos = 0;
    while ((pos = r.find(key, pos)) != std::string::npos) {
        r.replace(pos, key.length(), val);
        pos += val.length();
    }
    return r;
}

static int cmdNewProject(const std::string &name) {
    if (!std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(c) || c == '-' || c == '_';
    })) {
        err("Project name must only contain alphanumeric characters, hyphens, and underscores.");
        return 1;
    }

    std::string dir = name;
    if (std::filesystem::exists(dir)) {
        err("Directory '" + dir + "' already exists.");
        return 1;
    }

    mkdirP(dir + "/src");

    std::string mainContent = replacePlaceholder(TEMPLATE_MAIN, "{{name}}", name);
    if (!writeFile(dir + "/src/main.lin", mainContent)) {
        err("Failed to write src/main.lin");
        return 1;
    }

    std::string tomlContent = replacePlaceholder(TEMPLATE_TOML, "{{name}}", name);
    tomlContent = replacePlaceholder(tomlContent, "{{edition}}", LINC_EDITION);
    if (!writeFile(dir + "/Lin.toml", tomlContent)) {
        err("Failed to write Lin.toml");
        return 1;
    }

    writeFile(dir + "/.gitignore", TEMPLATE_GITIGNORE);

    green("Created new Lin project '" + name + "'");
    note("cd " + dir + " && linc build src/main.lin -o " + name);
    return 0;
}

static int cmdInitProject(const std::string &dir) {
    std::string targetDir = dir.empty() ? "." : dir;
    if (!std::filesystem::exists(targetDir)) {
        mkdirP(targetDir);
    }

    std::string tomlPath = targetDir + "/Lin.toml";
    if (std::filesystem::exists(tomlPath)) {
        err("Lin.toml already exists in '" + targetDir + "'.");
        return 1;
    }

    std::string projectName = std::filesystem::path(targetDir).filename().string();
    if (projectName == "." || projectName.empty()) {
        projectName = std::filesystem::current_path().filename().string();
    }

    mkdirP(targetDir + "/src");

    std::string mainContent = replacePlaceholder(TEMPLATE_MAIN, "{{name}}", projectName);
    std::string mainPath = targetDir + "/src/main.lin";
    if (!std::filesystem::exists(mainPath)) {
        writeFile(mainPath, mainContent);
    }

    std::string tomlContent = replacePlaceholder(TEMPLATE_TOML, "{{name}}", projectName);
    tomlContent = replacePlaceholder(tomlContent, "{{edition}}", LINC_EDITION);
    if (!writeFile(tomlPath, tomlContent)) {
        err("Failed to write Lin.toml");
        return 1;
    }

    std::string gitignorePath = targetDir + "/.gitignore";
    if (!std::filesystem::exists(gitignorePath)) {
        writeFile(gitignorePath, TEMPLATE_GITIGNORE);
    }

    green("Initialized Lin project in '" + targetDir + "'");
    return 0;
}

// ── Help / Version ──────────────────────────────────────────────────────────

static void printVersion() {
    std::cout << "linc " << LINC_VERSION << " (edition " << LINC_EDITION << ")\n";
}

static void printHelp() {
    std::cout << ansiColor(C_BOLD) << "linc" << ansiColor(C_RESET) << " " << LINC_VERSION << "\n";
    std::cout << "The Lin compiler\n\n";
    std::cout << ansiColor(C_BOLD) << "USAGE:" << ansiColor(C_RESET) << "\n";
    std::cout << "    linc <SUBCOMMAND> [options]\n\n";
    std::cout << ansiColor(C_BOLD) << "SUBCOMMANDS:" << ansiColor(C_RESET) << "\n";
    std::cout << "    build   Compile a .lin source file into an executable\n";
    std::cout << "    check   Type-check and analyze source without codegen\n";
    std::cout << "    test    Run built-in checkstyle and static analysis\n";
    std::cout << "    run     Compile and execute in one step\n";
    std::cout << "    new     Create a new Lin project from a template\n";
    std::cout << "    init    Initialize a Lin.toml in an existing directory\n";
    std::cout << "    help    Print this help message\n";
    std::cout << "\n";
    std::cout << ansiColor(C_BOLD) << "BUILD OPTIONS:" << ansiColor(C_RESET) << "\n";
    std::cout << "    -o <file>         Output binary path (default: linc_out)\n";
    std::cout << "    -I <path>         Add include search path\n";
    std::cout << "    --gpu             Enable GPU backend (Vulkan/SPIR-V)\n";
    std::cout << "    --wasm            Enable WebAssembly target\n";
    std::cout << "    --link-c <file>   Link additional C source file\n";
    std::cout << "    --link-libs <libs> Link additional libraries\n";
    std::cout << "    --quiet, -q       Suppress compiler diagnostic output\n";
    std::cout << "\n";
    std::cout << ansiColor(C_BOLD) << "EXAMPLES:" << ansiColor(C_RESET) << "\n";
    std::cout << "    linc new my-project\n";
    std::cout << "    linc build src/main.lin -o my-app\n";
    std::cout << "    linc check src/main.lin\n";
    std::cout << "    linc test src/main.lin\n";
    std::cout << "    linc build src/main.lin --gpu\n";
    std::cout << "\n";
}

// ── Compilation pipeline (core logic extracted from main) ───────────────────

static bool gQuiet = false;

static void log(const std::string &msg) {
    if (!gQuiet) std::cout << msg << "\n";
}

static int runCompilationPipeline(
    const std::string &sourceFile,
    const std::string &outputBinary,
    bool enableGPU,
    bool enableWasm,
    bool doLink,
    const std::vector<std::string> &includePaths,
    std::vector<const char*> &importSources,
    std::vector<std::string> &linkCFiles,
    std::vector<std::string> &linkLibs
) {
    std::ifstream file(sourceFile);
    if (!file.is_open()) {
        err("Failed to open file: " + sourceFile);
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source_str = buffer.str();

    // Implicitly inject std/types.lin and std/io.lin at the beginning of the AST
    // std/io.lin is needed for print/read I/O operations to register their rules and user-op functions.
    std::string fullSource = "import \"std/types.lin\"\nimport \"std/io.lin\"\n" + source_str;
    const char *patchedSource = strdup(fullSource.c_str());
    importSources.push_back(patchedSource);

    log("Parsing Source:\n" + fullSource);
    AstNode *ast = parse(patchedSource);
    if (ast) {
        if (ast->type == AST_BLOCK) {
            std::vector<std::string> searchPaths;
            searchPaths.push_back(".");

            std::filesystem::path srcPath(sourceFile);
            if (srcPath.has_parent_path()) {
                searchPaths.push_back(srcPath.parent_path().string());
            }

            char exePath[PATH_MAX];
            ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
            if (len != -1) {
                exePath[len] = '\0';
                std::filesystem::path ePath(exePath);
                if (ePath.has_parent_path()) {
                    std::filesystem::path binDir = ePath.parent_path();
                    searchPaths.push_back(binDir.string());
                    if (binDir.has_parent_path()) {
                        searchPaths.push_back(binDir.parent_path().string());
                        if (binDir.parent_path().has_parent_path()) {
                            searchPaths.push_back(binDir.parent_path().parent_path().string());
                            if (binDir.parent_path().parent_path().has_parent_path()) {
                                searchPaths.push_back(binDir.parent_path().parent_path().parent_path().string());
                            }
                        }
                    }
                }
            }

            searchPaths.insert(searchPaths.end(), includePaths.begin(), includePaths.end());

            int total_count = 0;
            int capacity = 16;
            AstNode **new_stmts = static_cast<AstNode**>(malloc(sizeof(AstNode*) * capacity));
            if (!new_stmts) {
                std::cerr << "Out of memory\n";
                return 1;
            }

            for (int i = 0; i < ast->as.block.count; i++) {
                if (ast->as.block.statements[i]->type == AST_IMPORT) {
                    std::string importPath = std::string(ast->as.block.statements[i]->as.import_stmt.path, ast->as.block.statements[i]->as.import_stmt.length);

                    std::ifstream importFile;
                    for (const auto& base : searchPaths) {
                        std::filesystem::path fullPath = std::filesystem::path(base) / importPath;
                        importFile.open(fullPath);
                        if (importFile.is_open()) break;
                    }

                    if (importFile.is_open()) {
                        std::stringstream importBuffer;
                        importBuffer << importFile.rdbuf();
                        std::string importSourceStr = importBuffer.str();
                        const char *importSource = strdup(importSourceStr.c_str());
                        importSources.push_back(importSource);
                        AstNode *importAst = parse(importSource);

                        if (importAst && importAst->type == AST_BLOCK) {
                            int import_count = importAst->as.block.count;
                            if (total_count + import_count > capacity) {
                                while (total_count + import_count > capacity) capacity *= 2;
                                AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                                if (!temp) {
                                    free(new_stmts);
                                    return 1;
                                }
                                new_stmts = temp;
                            }
                            memcpy(new_stmts + total_count, importAst->as.block.statements, sizeof(AstNode*) * import_count);
                            total_count += import_count;
                            free(importAst->as.block.statements);
                            free(importAst);
                        } else if (importAst) {
                            freeAst(importAst);
                        }
                    } else {
                        std::cerr << "Failed to import file: " << importPath << "\n";
                    }

                    ast->as.block.statements[i]->as.import_stmt.module_block = nullptr;
                    freeAst(ast->as.block.statements[i]);
                } else {
                    if (total_count >= capacity) {
                        capacity *= 2;
                        AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                        if (!temp) {
                            free(new_stmts);
                            return 1;
                        }
                        new_stmts = temp;
                    }
                    new_stmts[total_count++] = ast->as.block.statements[i];
                }
            }
            free(ast->as.block.statements);
            ast->as.block.statements = new_stmts;
            ast->as.block.count = total_count;
        }
        printAst(ast, 0);
    }

    if (!ast) {
        err("Parse failed.");
        return 1;
    }

    log("Initializing Lin Compiler...");

    mlir::DialectRegistry registry;
    registry.insert<mlir::pic::graph::PicGraphDialect>();
    registry.insert<mlir::pic::reduce::PicReduceDialect>();
    registry.insert<mlir::pic::runtime::PicRuntimeDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
    registry.insert<mlir::spirv::SPIRVDialect>();
#endif
    mlir::registerAllToLLVMIRTranslations(registry);
    mlir::registerConvertMathToLLVMInterface(registry);

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();

    log("PIC dialects registered successfully.\n");

    if (hasGpuAnnotation(ast)) {
        enableGPU = true;
    }

    std::unordered_set<std::string> declaredTypes;
    int semanticErrors = semanticTypeCheckAst(ast, declaredTypes);
    if (semanticErrors > 0) {
        err("Semantic analysis failed with " + std::to_string(semanticErrors) + " error(s).");
        if (ast) freeAst(ast);
        for (const char *src : importSources) free((void*)src);
        return 1;
    }

    // Type-check ran above; proceed to codegen + linking

    performTypeDirectedDispatch(ast);

    MlirContext cCtx = wrap(&context);
    MlirModule cModule = lowerAstToMlir(cCtx, ast);

    ModuleOp module = unwrap(cModule);

#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
    if (enableGPU) {
        auto targetEnv = mlir::spirv::getDefaultTargetEnv(module.getContext());
        module->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv);
    }
#endif

    log("Generated MLIR (before lowering):");
    if (!gQuiet) {
        module.print(llvm::outs());
        llvm::outs().flush();
    }

    PassManager pm(&context);
    pm.addPass(createPicGraphVerifyPass());
    pm.addPass(createPicGraphEGraphPass());
    pm.addPass(createPicGraphToReducePass());
    pm.addPass(createPicReduceToRuntimePass(enableGPU ? TargetBackend::GPU : TargetBackend::CPU));
    pm.addPass(createPicReduceLoweringPass(enableGPU ? TargetBackend::GPU : TargetBackend::CPU));
    pm.addPass(mlir::createConvertSCFToCFPass());
    if (enableGPU) {
        pm.addPass(createPicRuntimeToSPIRVPass());
    }
    pm.addPass(createPicRuntimeToLLVMPass(enableGPU ? TargetBackend::GPU : TargetBackend::CPU, outputBinary + ".spv"));

    if (mlir::failed(pm.run(module))) {
        err("Lowering and outlining pass failed.");
        return 1;
    }

#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
    if (enableGPU) {
        module.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
            if (gpuFunc.isKernel()) {
                auto abi = mlir::spirv::getEntryPointABIAttr(gpuFunc.getContext(), {1, 1, 1});
                gpuFunc->setAttr(mlir::spirv::getEntryPointABIAttrName(), abi);
            }
        });
    }
#endif

    if (enableGPU) {}

#if __has_include("mlir/Target/SPIRV/Serialization.h")
    if (enableGPU) {
        module.walk([&](mlir::spirv::ModuleOp spirvModule) {
            if (spirvModule->hasAttr("vce_triple")) return;
            auto vce = mlir::spirv::VerCapExtAttr::get(
                mlir::spirv::Version::V_1_0,
                mlir::ArrayRef<mlir::spirv::Capability>{
                    mlir::spirv::Capability::Shader,
                    mlir::spirv::Capability::Linkage},
                mlir::ArrayRef<mlir::spirv::Extension>{},
                spirvModule.getContext());
            spirvModule->setAttr("vce_triple", vce);
        });
        mlir::SmallVector<uint32_t, 0> spirvBinary;
        bool hasSpirv = false;
        module.walk([&](mlir::Operation *op) {
            if (hasSpirv) return;
            auto spirvModule = llvm::dyn_cast<mlir::spirv::ModuleOp>(op);
            if (!spirvModule) return;
            if (mlir::succeeded(mlir::spirv::serialize(spirvModule, spirvBinary))) {
                hasSpirv = true;
            }
        });

        if (hasSpirv) {
            std::string spirvPath = outputBinary + ".spv";
            std::ofstream spirvOut(spirvPath, std::ios::binary);
            if (!spirvOut) {
                err("Failed to open SPIR-V output file: " + spirvPath);
            } else {
                spirvOut.write(reinterpret_cast<const char *>(spirvBinary.data()),
                               static_cast<std::streamsize>(spirvBinary.size() * sizeof(uint32_t)));
                spirvOut.flush();
                note("Emitted SPIR-V module to " + spirvPath);
            }
        } else {
            warn("No SPIR-V module was produced during GPU lowering.");
        }
    }
#endif

    if (enableGPU) {
        llvm::SmallVector<mlir::Operation *> opsToErase;
        module.walk([&](mlir::Operation *op) {
            if (auto symNameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
                if (symNameAttr.getValue() == "pic") {
                    opsToErase.push_back(op);
                }
            }
        });
        for (auto op : opsToErase) {
            op->erase();
        }
        llvm::SmallVector<mlir::spirv::ModuleOp> spirvModules;
        module.walk([&](mlir::spirv::ModuleOp op) {
            spirvModules.push_back(op);
        });
        for (auto op : spirvModules) op.erase();

        llvm::SmallVector<mlir::gpu::GPUModuleOp> gpuModules;
        module.walk([&](mlir::gpu::GPUModuleOp op) {
            gpuModules.push_back(op);
        });
        for (auto op : gpuModules) op.erase();
    }

    PassManager pm2(&context);
    pm2.addPass(mlir::createConvertSCFToCFPass());
    pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm2.addPass(mlir::createConvertMathToLibmPass());
    pm2.addPass(mlir::createConvertMathToLLVMPass());
    pm2.addPass(mlir::createArithToLLVMConversionPass());
    pm2.addPass(mlir::createConvertFuncToLLVMPass());
    pm2.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm2.run(module))) {
        err("LLVM conversion passes failed.");
        return 1;
    }

    log("\nLowering pass successful.");

    log("--- MLIR Module after lowering ---");
    if (!gQuiet) {
        module.print(llvm::outs());
        log("------------------------------------");
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        err("Failed to translate MLIR to LLVM IR.");
        return 1;
    }

    std::string targetError;
    std::string targetTriple = enableWasm ? "wasm32-unknown-unknown" : llvm::sys::getDefaultTargetTriple();
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, targetError);
    if (!target && !enableWasm) {
        targetTriple = "x86_64-unknown-linux-gnu";
        target = llvm::TargetRegistry::lookupTarget(targetTriple, targetError);
    }

    if (!target) {
        err("Failed to lookup target: " + targetError);
        return 1;
    }

    llvm::TargetOptions opt;
    auto rm = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
    auto targetMachine = target->createTargetMachine(targetTriple, "generic", "", opt, rm);

    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetTriple);

    std::string objFile = outputBinary + ".o";
    std::error_code ec;
    llvm::raw_fd_ostream dest(objFile, ec, llvm::sys::fs::OF_None);
    if (ec) {
        err("Could not open file: " + ec.message());
        return 1;
    }

    llvm::legacy::PassManager pass;
    auto fileType = llvm::CodeGenFileType::ObjectFile;
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
        err("TargetMachine can't emit a file of this type");
        return 1;
    }

    pass.run(*llvmModule);
    dest.flush();

    log("Successfully emitted object file to " + objFile);

    delete targetMachine;

    if (doLink) {
        pid_t pid = fork();
        if (pid == -1) {
            err("Failed to fork process for linking.");
            return 1;
        } else if (pid == 0) {
            std::vector<char*> args;
            if (enableWasm) {
                args.push_back(const_cast<char*>("wasm-ld"));
                args.push_back(const_cast<char*>(objFile.c_str()));
                for (auto &cf : linkCFiles) args.push_back(const_cast<char*>(cf.c_str()));
                args.push_back(const_cast<char*>("-o"));
                args.push_back(const_cast<char*>(outputBinary.c_str()));
                args.push_back(const_cast<char*>("--no-entry"));
                args.push_back(const_cast<char*>("--export-all"));
                args.push_back(nullptr);
                execvp("wasm-ld", args.data());
            } else {
                args.push_back(const_cast<char*>("gcc"));
                args.push_back(const_cast<char*>(objFile.c_str()));
                for (auto &cf : linkCFiles) args.push_back(const_cast<char*>(cf.c_str()));
                args.push_back(const_cast<char*>("-o"));
                args.push_back(const_cast<char*>(outputBinary.c_str()));
                args.push_back(const_cast<char*>("-lpthread"));
                const char* ldPath = getenv("LD_LIBRARY_PATH");
                if (ldPath) {
                    std::string s(ldPath);
                    size_t pos = 0;
                    while ((pos = s.find(":")) != std::string::npos) {
                        std::string path = s.substr(0, pos);
                        if (!path.empty()) {
                            char* arg = new char[path.length() + 3];
                            sprintf(arg, "-L%s", path.c_str());
                            args.push_back(arg);
                        }
                        s.erase(0, pos + 1);
                    }
                    if (!s.empty()) {
                        char* arg = new char[s.length() + 3];
                        sprintf(arg, "-L%s", s.c_str());
                        args.push_back(arg);
                    }
                }
                std::string gpuRuntimePath = "src/gpu_runtime.c";
                if (enableGPU) {
                    char exePath[PATH_MAX];
                    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
                    if (len != -1) {
                        exePath[len] = '\0';
                        std::filesystem::path ePath(exePath);
                        std::filesystem::path binDir = ePath.parent_path();
                        std::filesystem::path pDir = binDir.parent_path();
                        if (pDir.has_parent_path()) {
                            std::filesystem::path candidate = pDir.parent_path() / "src" / "gpu_runtime.c";
                            if (std::filesystem::exists(candidate)) gpuRuntimePath = candidate.string();
                            else {
                                candidate = pDir / "src" / "gpu_runtime.c";
                                if (std::filesystem::exists(candidate)) gpuRuntimePath = candidate.string();
                            }
                        }
                    }
                    args.push_back(const_cast<char*>(gpuRuntimePath.c_str()));
                    args.push_back(const_cast<char*>("-lvulkan"));
                }
                args.push_back(const_cast<char*>("-lm"));
                for (auto &lib : linkLibs) {
                    std::istringstream libStream(lib);
                    std::string singleLib;
                    while (libStream >> singleLib) {
                        args.push_back(const_cast<char*>(strdup(singleLib.c_str())));
                    }
                }
                args.push_back(nullptr);
                execvp("gcc", args.data());
            }
            perror("execvp failed");
            exit(1);
        } else {
            int status;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                err("Linking failed (exit code " + std::to_string(WEXITSTATUS(status)) + ").");
                return 1;
            } else if (WIFSIGNALED(status)) {
                if (llvm::sys::fs::exists(outputBinary)) {
                    warn("Linker terminated by signal " + std::to_string(WTERMSIG(status)) + ", but output binary was created.");
                } else {
                    err("Linking failed (signal " + std::to_string(WTERMSIG(status)) + ").");
                    return 1;
                }
            }
        }
        log("Successfully compiled and linked to '" + outputBinary + "'.");
    } else {
        log("Compilation complete.");
    }

    mlirModuleDestroy(cModule);

    if (ast) freeAst(ast);
    for (const char *src : importSources) free((void*)src);

    return 0;
}

// ── Main entry point ────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        printHelp();
        return 1;
    }

    // Scan for global flags and subcommands at any position.
    // The LIT substitution passes `-I <path>` before the subcommand:
    //   build/src/linc -I /projectroot build file.lin -o tmp
    // So we cannot assume argv[1] is the subcommand.

    std::string command;
    std::string sourceFile;
    std::string outputBinary = "linc_out";
    bool enableGPU = false;
    bool enableWasm = false;
    std::vector<std::string> includePaths;
    std::vector<const char*> importSources;
    std::vector<std::string> linkCFiles;
    std::vector<std::string> linkLibs;

    // Recognized subcommands
    auto isSubcommand = [](const std::string &s) -> bool {
        return s == "build" || s == "test" || s == "run" || s == "check"
            || s == "new" || s == "init" || s == "help";
    };

    // First pass: find subcommand and collect positional args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Handle global flags
        if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        }
        if (arg == "--version" || arg == "-V") {
            printVersion();
            return 0;
        }
        if (arg == "--quiet" || arg == "-q") {
            gQuiet = true;
            continue;
        }

        if (isSubcommand(arg)) {
            command = arg;
            // Parse remaining args as subcommand-specific flags
            for (int j = i + 1; j < argc; ++j) {
                std::string opt = argv[j];
                if (opt == "-o" && j + 1 < argc) {
                    outputBinary = argv[++j];
                } else if (opt == "-I" && j + 1 < argc) {
                    includePaths.push_back(argv[++j]);
                } else if (opt == "--quiet" || opt == "-q") {
                    gQuiet = true;
                } else if (opt == "--gpu" || opt == "-gpu") {
                    enableGPU = true;
                } else if (opt == "--wasm" || opt == "-wasm") {
                    enableWasm = true;
                } else if ((opt == "--link-c" || opt == "-link-c") && j + 1 < argc) {
                    linkCFiles.push_back(argv[++j]);
                } else if ((opt == "--link-libs" || opt == "-link-libs") && j + 1 < argc) {
                    linkLibs.push_back(argv[++j]);
                } else if (sourceFile.empty()) {
                    sourceFile = opt;
                }
            }
            break;
        }
    }

    if (command.empty()) {
        err("No valid subcommand found. Use 'linc help'.");
        return 1;
    }

    // ── `help` subcommand ───────────────────────────────────────────────────
    if (command == "help") {
        printHelp();
        return 0;
    }

    // ── `new` subcommand ───────────────────────────────────────────────────
    if (command == "new") {
        if (sourceFile.empty()) {
            err("Usage: linc new <project-name>");
            return 1;
        }
        return cmdNewProject(sourceFile);
    }

    // ── `init` subcommand ─────────────────────────────────────────────────
    if (command == "init") {
        std::string dir = sourceFile.empty() ? "." : sourceFile;
        return cmdInitProject(dir);
    }

    // All remaining subcommands require a source file
    bool checkOnly = (command == "check");
    bool doLink = (command == "build");
    bool doRun = (command == "run");
    bool doTest = (command == "test");

    if (sourceFile.empty()) {
        err("No source file provided.");
        note("Usage: linc " + command + " <source_file.lin> [-o output]");
        return 1;
    }

    if (!outputBinary.empty() && outputBinary[0] == '-') {
        err("Output binary name cannot start with a hyphen.");
        return 1;
    }

    // Check for Lin.toml in the source directory for extended include resolution
    std::filesystem::path srcPath(sourceFile);
    std::string searchDir = srcPath.has_parent_path() ? srcPath.parent_path().string() : ".";
    std::string tomlPath = searchDir + "/Lin.toml";
    if (std::filesystem::exists(tomlPath)) {
        ProjectConfig cfg = parseProjectConfig(tomlPath);
        if (cfg.valid) {
            for (const auto &inc : cfg.includes) {
                includePaths.push_back(inc);
            }
        }
    }

    // ── `check` subcommand ─────────────────────────────────────────────────
    if (checkOnly) {
        // type-check only — skip codegen
        // We reuse the pipeline but skip the MLIR lowering and linking
        std::ifstream file(sourceFile);
        if (!file.is_open()) {
            err("Failed to open file: " + sourceFile);
            return 1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source_str = buffer.str();
        std::string fullSource = "import \"std/types.lin\"\nimport \"std/io.lin\"\n" + source_str;
        const char *patchedSource = strdup(fullSource.c_str());
        importSources.push_back(patchedSource);

        AstNode *ast = parse(patchedSource);
        if (ast && ast->type == AST_BLOCK) {
            std::vector<std::string> searchPaths;
            searchPaths.push_back(".");
            std::filesystem::path srcPathSF(sourceFile);
            if (srcPathSF.has_parent_path())
                searchPaths.push_back(srcPathSF.parent_path().string());
            char exePath[PATH_MAX];
            ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
            if (len != -1) {
                exePath[len] = '\0';
                std::filesystem::path ePath(exePath);
                if (ePath.has_parent_path()) {
                    std::filesystem::path binDir = ePath.parent_path();
                    searchPaths.push_back(binDir.string());
                    if (binDir.has_parent_path()) {
                        searchPaths.push_back(binDir.parent_path().string());
                        if (binDir.parent_path().has_parent_path()) {
                            searchPaths.push_back(binDir.parent_path().parent_path().string());
                            if (binDir.parent_path().parent_path().has_parent_path())
                                searchPaths.push_back(binDir.parent_path().parent_path().parent_path().string());
                        }
                    }
                }
            }
            searchPaths.insert(searchPaths.end(), includePaths.begin(), includePaths.end());

            int total_count = 0;
            int capacity = 16;
            AstNode **new_stmts = static_cast<AstNode**>(malloc(sizeof(AstNode*) * capacity));
            if (!new_stmts) { std::cerr << "Out of memory\n"; return 1; }

            for (int i = 0; i < ast->as.block.count; i++) {
                if (ast->as.block.statements[i]->type == AST_IMPORT) {
                    std::string importPath(ast->as.block.statements[i]->as.import_stmt.path, ast->as.block.statements[i]->as.import_stmt.length);
                    std::ifstream importFile;
                    for (const auto& base : searchPaths) {
                        std::filesystem::path fullPath = std::filesystem::path(base) / importPath;
                        importFile.open(fullPath);
                        if (importFile.is_open()) break;
                    }
                    if (importFile.is_open()) {
                        std::stringstream importBuffer;
                        importBuffer << importFile.rdbuf();
                        std::string importSourceStr = importBuffer.str();
                        const char *importSource = strdup(importSourceStr.c_str());
                        importSources.push_back(importSource);
                        AstNode *importAst = parse(importSource);
                        if (importAst && importAst->type == AST_BLOCK) {
                            int import_count = importAst->as.block.count;
                            if (total_count + import_count > capacity) {
                                while (total_count + import_count > capacity) capacity *= 2;
                                AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                                if (!temp) { free(new_stmts); return 1; }
                                new_stmts = temp;
                            }
                            memcpy(new_stmts + total_count, importAst->as.block.statements, sizeof(AstNode*) * import_count);
                            total_count += import_count;
                            free(importAst->as.block.statements);
                            free(importAst);
                        } else if (importAst) { freeAst(importAst); }
                    } else {
                        std::cerr << "Failed to import file: " << importPath << "\n";
                    }
                    ast->as.block.statements[i]->as.import_stmt.module_block = nullptr;
                    freeAst(ast->as.block.statements[i]);
                } else {
                    if (total_count >= capacity) {
                        capacity *= 2;
                        AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                        if (!temp) { free(new_stmts); return 1; }
                        new_stmts = temp;
                    }
                    new_stmts[total_count++] = ast->as.block.statements[i];
                }
            }
            free(ast->as.block.statements);
            ast->as.block.statements = new_stmts;
            ast->as.block.count = total_count;
        }

        if (!ast) {
            err("Parse failed.");
            for (const char *src : importSources) free((void*)src);
            return 1;
        }

        std::unordered_set<std::string> declaredTypes;
        int errors = semanticTypeCheckAst(ast, declaredTypes);
        if (ast) freeAst(ast);
        for (const char *src : importSources) free((void*)src);

        if (errors > 0) {
            err("Type-check failed with " + std::to_string(errors) + " error(s).");
            return 1;
        }
        green("Type-check passed (no errors).");
        return 0;
    }

    // ── `build` subcommand ─────────────────────────────────────────────────
    if (doLink) {
        int ret = runCompilationPipeline(sourceFile, outputBinary, enableGPU, enableWasm, true, includePaths, importSources, linkCFiles, linkLibs);
        if (ret == 0) {
            green("Compiled and linked to '" + outputBinary + "'.");
        }
        return ret;
    }

    // ── `run` subcommand ───────────────────────────────────────────────────
    if (doRun) {
        int ret = runCompilationPipeline(sourceFile, outputBinary, enableGPU, enableWasm, true, includePaths, importSources, linkCFiles, linkLibs);
        if (ret != 0) return ret;

        std::cout << "Running " << outputBinary << "...\n";
        int exitCode = system(outputBinary.c_str());
        exitCode = WEXITSTATUS(exitCode);
        return exitCode;
    }

    // ── `test` subcommand ─────────────────────────────────────────────────
    if (doTest) {
        int ret = runCompilationPipeline(sourceFile, outputBinary, enableGPU, enableWasm, false, includePaths, importSources, linkCFiles, linkLibs);
        if (ret != 0) return ret;

        // Re-parse the source for checkstyle (runs on original source, not import-resolved AST)
        std::ifstream file(sourceFile);
        if (!file.is_open()) { err("Failed to open file."); return 1; }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source_str = buffer.str();
        std::string fullSource = "import \"std/types.lin\"\nimport \"std/io.lin\"\n" + source_str;
        const char *patchedSource = strdup(fullSource.c_str());
        std::vector<const char*> testImportSources;
        testImportSources.push_back(patchedSource);

        // Resolve imports to match original behavior
        AstNode *ast = parse(patchedSource);
        if (!ast) { err("Parse failed."); return 1; }

        // Apply the same import resolution as the pipeline
        if (ast->type == AST_BLOCK) {
            std::vector<std::string> searchPaths;
            searchPaths.push_back(".");
            std::filesystem::path srcPath(sourceFile);
            if (srcPath.has_parent_path())
                searchPaths.push_back(srcPath.parent_path().string());
            char exePath[PATH_MAX];
            ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
            if (len != -1) {
                exePath[len] = '\0';
                std::filesystem::path ePath(exePath);
                if (ePath.has_parent_path()) {
                    std::filesystem::path binDir = ePath.parent_path();
                    searchPaths.push_back(binDir.string());
                    if (binDir.has_parent_path()) {
                        searchPaths.push_back(binDir.parent_path().string());
                        if (binDir.parent_path().has_parent_path())
                            searchPaths.push_back(binDir.parent_path().parent_path().string());
                    }
                }
            }
            searchPaths.insert(searchPaths.end(), includePaths.begin(), includePaths.end());

            int total_count = 0;
            int capacity = 16;
            AstNode **new_stmts = (AstNode**)malloc(sizeof(AstNode*) * capacity);
            if (!new_stmts) { std::cerr << "Out of memory\n"; return 1; }

            for (int i = 0; i < ast->as.block.count; i++) {
                if (ast->as.block.statements[i]->type == AST_IMPORT) {
                    std::string importPath(ast->as.block.statements[i]->as.import_stmt.path, ast->as.block.statements[i]->as.import_stmt.length);
                    std::ifstream importFile;
                    for (const auto& base : searchPaths) {
                        std::filesystem::path fullPath = std::filesystem::path(base) / importPath;
                        importFile.open(fullPath);
                        if (importFile.is_open()) break;
                    }
                    if (importFile.is_open()) {
                        std::stringstream importBuffer;
                        importBuffer << importFile.rdbuf();
                        std::string importSourceStr = importBuffer.str();
                        const char *importSource = strdup(importSourceStr.c_str());
                        testImportSources.push_back(importSource);
                        AstNode *importAst = parse(importSource);
                        if (importAst && importAst->type == AST_BLOCK) {
                            int import_count = importAst->as.block.count;
                            if (total_count + import_count > capacity) {
                                while (total_count + import_count > capacity) capacity *= 2;
                                AstNode **temp = (AstNode**)realloc(new_stmts, sizeof(AstNode*) * capacity);
                                if (!temp) { free(new_stmts); return 1; }
                                new_stmts = temp;
                            }
                            memcpy(new_stmts + total_count, importAst->as.block.statements, sizeof(AstNode*) * import_count);
                            total_count += import_count;
                            free(importAst->as.block.statements);
                            free(importAst);
                        } else if (importAst) { freeAst(importAst); }
                    }
                    ast->as.block.statements[i]->as.import_stmt.module_block = nullptr;
                    freeAst(ast->as.block.statements[i]);
                } else {
                    if (total_count >= capacity) {
                        capacity *= 2;
                        AstNode **temp = (AstNode**)realloc(new_stmts, sizeof(AstNode*) * capacity);
                        if (!temp) { free(new_stmts); return 1; }
                        new_stmts = temp;
                    }
                    new_stmts[total_count++] = ast->as.block.statements[i];
                }
            }
            free(ast->as.block.statements);
            ast->as.block.statements = new_stmts;
            ast->as.block.count = total_count;
        }

        std::cout << "\nRunning built-in checkstyle and static analysis...\n";
        int styleErrors = checkstyleAst(ast);
        if (ast) freeAst(ast);
        for (const char *src : testImportSources) free((void*)src);

        if (styleErrors > 0) {
            std::cerr << "Checkstyle and static analysis failed with " << styleErrors << " error(s).\n";
            return 1;
        }
        std::cout << "Built-in checkstyle and static analysis passed.\n";
        std::cout << "Compilation complete.\n";
        return 0;
    }

    return 0;
}