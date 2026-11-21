CLANG ?= clang
OUT := build/clang/metrics.so
.PHONY: all
all: $(OUT)

$(OUT): metrics.c Makefile
	@mkdir -p "$(@D)"
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	"$(CLANG)" $(CPPFLAGS) -std=c17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -fPIC -shared $(CFLAGS) "$<" $(LDFLAGS) -o "$$temp"; \
	mv -f -- "$$temp" "$@"
