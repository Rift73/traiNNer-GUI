CLANGXX ?= /usr/bin/clang++
BUILD := build/clang
STAMP := $(BUILD)/toolchain.txt

.PHONY: all FORCE
all: $(BUILD)/sha256.so

$(STAMP): FORCE
	@mkdir -p "$(@D)"
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	driver=$$(command -v "$(CLANGXX)"); \
	printf 'driver=%s\nbinary=%s\n' "$$driver" "$$(readlink -f "$$driver")" > "$$temp"; \
	"$(CLANGXX)" $(CPPFLAGS) $(CXXFLAGS) --version >> "$$temp"; \
	grep -q 'clang version ' "$$temp"; \
	if ! cmp -s "$$temp" "$@"; then mv -f -- "$$temp" "$@"; fi

$(BUILD)/sha256.so: sha256.cpp Makefile $(STAMP)
	@set -eu; temp=$$(mktemp "$@.XXXXXX"); trap 'rm -f -- "$$temp"' EXIT; \
	flags=$$(pkg-config --cflags --libs glib-2.0); \
	"$(CLANGXX)" $(CPPFLAGS) -std=c++17 -O2 -Wall -Wextra -Wpedantic $(CXXFLAGS) \
		-fPIC -shared "$<" $(LDFLAGS) $$flags -o "$$temp"; \
	mv -f -- "$$temp" "$@"
