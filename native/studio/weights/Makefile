CLANG ?= clang
OUT := build/clang/measure.so
.PHONY: all
all: $(OUT)

$(OUT): measure.c Makefile
	@mkdir -p "$(@D)"
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	"$(CLANG)" $(CPPFLAGS) -std=c17 -O2 -Wall -Wextra -fPIC -shared $(CFLAGS) "$<" $(LDFLAGS) -o "$$temp"; \
	mv -f -- "$$temp" "$@"
