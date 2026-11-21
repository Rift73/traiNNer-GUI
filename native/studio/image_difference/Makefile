CLANG ?= /usr/bin/clang
BUILD := build/clang
STAMP := $(BUILD)/toolchain.txt

.PHONY: all FORCE
all: $(BUILD)/difference.so

$(STAMP): FORCE
	@mkdir -p "$(@D)"
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	driver=$$(command -v "$(CLANG)"); \
	printf 'driver=%s\nbinary=%s\n' "$$driver" "$$(readlink -f "$$driver")" > "$$temp"; \
	"$(CLANG)" $(CPPFLAGS) $(CFLAGS) --version >> "$$temp"; \
	pkg-config --modversion libpng >> "$$temp"; \
	printf '%s\n' "$(CPPFLAGS) $(CFLAGS) $(LDFLAGS)" >> "$$temp"; \
	grep -q 'clang version ' "$$temp"; \
	if ! cmp -s "$$temp" "$@"; then mv -f -- "$$temp" "$@"; fi

$(BUILD)/difference.so: difference.c Makefile $(STAMP)
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	flags=$$(pkg-config --cflags --libs libpng); \
	"$(CLANG)" $(CPPFLAGS) -std=c17 -O2 -Wall -Wextra -Wpedantic $(CFLAGS) \
		-fPIC -shared "$<" $(LDFLAGS) $$flags -lm -o "$$temp"; \
	mv -f -- "$$temp" "$@"
