CLANG ?= /usr/bin/clang
LIB := build/clang/resample.so

.PHONY: all
all: $(LIB)

$(LIB): resample.c Makefile ../tool.mk
	@mkdir -p "$(@D)"
	@set -eu; tmp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$tmp"' EXIT; \
	"$(CLANG)" $(CPPFLAGS) -std=c17 -O2 -Wall -Wextra -Wpedantic $(CFLAGS) \
		-fPIC -shared "$<" $(LDFLAGS) -o "$$tmp"; \
	mv -f -- "$$tmp" "$@"
