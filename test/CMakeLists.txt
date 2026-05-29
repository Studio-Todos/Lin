find_program(LIT_EXECUTABLE lit)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_executable(parser_test parser_test.c ../src/Parser.c ../src/Ast.c)
target_include_directories(parser_test PRIVATE ../include)

add_executable(print_ast_test test_print_ast_standalone.c ../src/Parser.c ../src/Ast.c)
target_include_directories(print_ast_test PRIVATE ../include)

add_custom_target(check-linc
  COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS linc parser_test print_ast_test
  COMMENT "Running Lin tests"
)
