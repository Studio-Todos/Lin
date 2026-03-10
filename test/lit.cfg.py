import os
import lit.formats
import lit.llvm

# Initialize lit.llvm (which will help find FileCheck)
lit.llvm.initialize(lit_config, config)

config.name = 'Lin'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.lin']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, '..', 'build', 'test')

# Let lit know about the lin compiler binary
config.substitutions.append(('%lin', os.path.join(config.test_source_root, '..', 'build', 'src', 'lin')))
