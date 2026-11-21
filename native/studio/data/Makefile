CLANGXX ?= /usr/bin/clang++
BUILD_DIR := build/clang
TOOLCHAIN := $(BUILD_DIR)/toolchain.txt

.PHONY: all paired-folders paired-video paired-crop paired-identity task-sample-pack mask-histogram FORCE

all: $(BUILD_DIR)/inspect
paired-video: $(BUILD_DIR)/paired_video_pack.so
paired-crop: $(BUILD_DIR)/paired_crop.so
paired-identity: $(BUILD_DIR)/paired_identity.so
task-sample-pack: $(BUILD_DIR)/task_sample_pack.so
mask-histogram: $(BUILD_DIR)/mask_histogram.so
paired-folders: $(BUILD_DIR)/pair-folders

$(BUILD_DIR)/pair-folders: $(BUILD_DIR)/inspect
	ln -sfn -- inspect "$@"

$(TOOLCHAIN): FORCE
	@mkdir -p "$(@D)"
	@set -eu; \
	temp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f -- "$$temp"' EXIT; \
	driver=$$(command -v "$(CLANGXX)"); \
	printf 'driver=%s\nbinary=%s\n' "$$driver" "$$(readlink -f "$$driver")" > "$$temp"; \
	"$(CLANGXX)" $(CPPFLAGS) $(CXXFLAGS) --version >> "$$temp"; \
	grep -q 'clang version ' "$$temp" || { echo 'LLVM Clang C++ is required' >&2; exit 1; }; \
	printf 'linker=%s\n' "$$("$(CLANGXX)" $(CXXFLAGS) $(LDFLAGS) -print-prog-name=ld)" >> "$$temp"; \
	if ! cmp -s "$$temp" "$@"; then mv -f -- "$$temp" "$@"; fi

$(BUILD_DIR)/inspect: codec_packages := gdk-pixbuf-2.0 libtiff-4 libwebp
$(BUILD_DIR)/inspect: inspect.cpp
$(BUILD_DIR)/paired_video_pack.so: target_flags := -shared -fPIC
$(BUILD_DIR)/paired_video_pack.so: paired_video_pack.cpp rgb8_pack.hpp

$(BUILD_DIR)/paired_crop.so: target_flags := -shared -fPIC
$(BUILD_DIR)/paired_crop.so: paired_crop.cpp

$(BUILD_DIR)/paired_identity.so: target_flags := -shared -fPIC
$(BUILD_DIR)/paired_identity.so: paired_identity.cpp

$(BUILD_DIR)/task_sample_pack.so: target_flags := -shared -fPIC
$(BUILD_DIR)/task_sample_pack.so: task_sample_pack.cpp rgb8_pack.hpp

$(BUILD_DIR)/mask_histogram.so: codec_packages := libpng
$(BUILD_DIR)/mask_histogram.so: target_flags := -shared -fPIC
$(BUILD_DIR)/mask_histogram.so: mask_histogram.cpp

$(BUILD_DIR)/inspect $(BUILD_DIR)/paired_video_pack.so $(BUILD_DIR)/paired_crop.so $(BUILD_DIR)/paired_identity.so $(BUILD_DIR)/task_sample_pack.so $(BUILD_DIR)/mask_histogram.so: Makefile $(TOOLCHAIN)
	@set -eu; \
	temp=$$(mktemp "$@.XXXXXX"); \
	trap 'rm -f -- "$$temp"' EXIT; \
	flags=""; \
	if [ -n "$(codec_packages)" ]; then flags=$$(pkg-config --cflags --libs $(codec_packages)); fi; \
	"$(CLANGXX)" $(CPPFLAGS) -std=c++17 -O2 -Wall -Wextra -Wpedantic $(CXXFLAGS) \
		$(target_flags) $(filter %.cpp,$^) $(LDFLAGS) $$flags -o "$$temp"; \
	chmod u+x "$$temp"; \
	mv -f -- "$$temp" "$@"
