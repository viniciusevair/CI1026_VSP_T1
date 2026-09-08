SHELL := /bin/sh

LATEXMK ?= latexmk
REPORT_DIR ?= report
TEX_FILE ?= $(REPORT_DIR)/main.tex
BUILD_DIR ?= $(REPORT_DIR)/build
IMAGE_DIR ?= $(REPORT_DIR)/images

PDF_FILE := $(BUILD_DIR)/$(basename $(notdir $(TEX_FILE))).pdf
IMAGE_FILES := $(wildcard $(IMAGE_DIR)/*.png) \
               $(wildcard $(IMAGE_DIR)/*.jpg) \
               $(wildcard $(IMAGE_DIR)/*.jpeg) \
               $(wildcard $(IMAGE_DIR)/*.pdf)

LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf images clean

all: pdf

pdf: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) $(IMAGE_FILES) | $(BUILD_DIR) images
	cd $(REPORT_DIR) && $(LATEXMK) $(LATEXMK_FLAGS) -outdir=$(notdir $(BUILD_DIR)) $(notdir $(TEX_FILE))

$(BUILD_DIR):
	mkdir -p $@

images:
	mkdir -p $(IMAGE_DIR)

clean:
	@if [ -f "$(TEX_FILE)" ]; then \
		cd $(REPORT_DIR) && $(LATEXMK) -C -outdir=$(notdir $(BUILD_DIR)) $(notdir $(TEX_FILE)); \
	fi
	rm -rf $(BUILD_DIR)
	rm -f \
		$(REPORT_DIR)/$(basename $(notdir $(TEX_FILE))).pdf \
		$(REPORT_DIR)/*.aux \
		$(REPORT_DIR)/*.fdb_latexmk \
		$(REPORT_DIR)/*.fls \
		$(REPORT_DIR)/*.log \
		$(REPORT_DIR)/*.out \
		$(REPORT_DIR)/*.synctex.gz \
		$(REPORT_DIR)/*.toc
