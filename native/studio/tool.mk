.PHONY: all
CLANGXX ?= clang++
all: build/clang/$(TOOL)

build/clang/$(TOOL): $(SOURCE) Makefile ../tool.mk
	@mkdir -p "$(@D)"
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	"$(CLANGXX)" $(CPPFLAGS) -std=c++17 -O2 -Wall -Wextra -Wpedantic $(CXXFLAGS) $(SOURCE) $(LDFLAGS) -o "$$temp"; \
	chmod u+x "$$temp"; mv -f -- "$$temp" "$@"
