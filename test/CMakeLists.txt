find_program(LIT_EXECUTABLE lit)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check-lin
  COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS lin
  COMMENT "Running Lin tests"
)
