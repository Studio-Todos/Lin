import os
import sys
import re
import platform
import subprocess
import lit.util
import lit.formats
# Configuration file for the 'lit' test runner.
import lit.formats

# name: The name of this test suite.
config.name = 'Lin'

config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lin_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

import lit.llvm
lit.llvm.initialize(lit_config, config)

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lin_obj_root, 'test')
config.lin_tools_dir = os.path.join(config.lin_obj_root, 'src')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.lin_tools_dir, append_path=True)

tool_dirs = [config.lin_tools_dir, config.llvm_tools_dir]
tools = [
    'lin',
    ToolSubst('%mlir_runner_utils', config.mlir_runner_utils_dir),
    ToolSubst('%mlir_c_runner_utils', config.mlir_c_runner_utils_dir)
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
