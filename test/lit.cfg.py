import os
import lit.formats
import lit.llvm

# Initialize lit.llvm (which will help find FileCheck)
lit.llvm.initialize(lit_config, config)

config.name = 'Lin'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.lin']
config.test_source_root = os.path.dirname(__file__)

# Use CMake-configured paths if available, fall back to relative defaults
lin_src_root = getattr(config, 'lin_src_root', None)
lin_obj_root = getattr(config, 'lin_obj_root', None)

if not lin_src_root:
    lin_src_root = os.path.abspath(os.path.join(config.test_source_root, '..'))
if not lin_obj_root:
    lin_obj_root = os.path.abspath(os.path.join(config.test_source_root, '..', 'build'))

config.test_exec_root = os.path.join(lin_obj_root, 'test')

# Let lit know about the linc compiler binary
project_root = lin_src_root
config.substitutions.append(('%linc', os.path.join(lin_obj_root, 'src', 'linc') + " -I " + project_root))
config.substitutions.append(('%parser_test', os.path.join(lin_obj_root, 'test', 'parser_test')))
config.substitutions.append(('%print_ast_test', os.path.join(lin_obj_root, 'test', 'print_ast_test')))
# Propagate all environment variables for Nix
for var in os.environ:
    config.environment[var] = os.environ[var]
