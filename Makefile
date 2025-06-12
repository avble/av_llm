# vim: ts=2 sw=2

AV_LLM_VERSION = 0.0.1-preview
UNAME_S := $(shell uname -s)

# Define common paths
BUILD_DIR = build
STAGING_DIR = staging
OUTPUT_DIR = output
INSTALL_BIN_DIR = usr/local/bin
INSTALL_CONFIG_DIR = usr/local/etc/.av_llm
MODEL_FILE = qwen1_5-0_5b-chat-q8_0.gguf
MODEL_URL = https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/$(MODEL_FILE)

RM = rm -rf
MKDIR = mkdir -p
CP = cp -rf
WGET = wget -O

all: compile
	@echo "Create the package"

compile:
	$(MKDIR) $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release  && cmake --build $(BUILD_DIR) 

package-prepare: compile
	$(MKDIR) $(STAGING_DIR)/$(INSTALL_BIN_DIR)
	$(MKDIR) $(STAGING_DIR)/$(INSTALL_CONFIG_DIR)
	$(CP) $(BUILD_DIR)/av_llm $(STAGING_DIR)/$(INSTALL_BIN_DIR)/
ifeq ($(UNAME_S), Linux)
	$(MKDIR) $(STAGING_DIR)/DEBIAN
	$(CP) scripts/control $(STAGING_DIR)/DEBIAN/
endif

package: compile package-prepare
	$(MKDIR) $(OUTPUT_DIR)
ifeq ($(UNAME_S), Darwin)
	pkgbuild --root ./$(STAGING_DIR) --identifier avble.llm.app --version 1.0 av_llm.pkg
	productbuild --distribution ./scripts/Distribution.xml --package-path ./ $(OUTPUT_DIR)/av_llm-universal-installer-$(AV_LLM_VERSION).pkg && rm av_llm.pkg
else ifeq ($(UNAME_S), Linux)
		dpkg-deb --build $(STAGING_DIR) $(OUTPUT_DIR)/av_llm-linux-installer-$(AV_LLM_VERSION).deb
endif

package-clean:
	$(RM) $(OUTPUT_DIR)

clean:
	$(RM) $(STAGING_DIR)
	cmake --build $(BUILD_DIR) --target clean
	$(RM) $(OUTPUT_DIR)

install-test:
ifeq ($(UNAME_S), Darwin)
		sudo installer -pkg $(OUTPUT_DIR)/av_llm-universal-installer-$(AV_LLM_VERSION).pkg -target /
else ifeq ($(UNAME_S), Linux)
		sudo dpkg -i $(OUTPUT_DIR)/av_llm-linux-installer-$(AV_LLM_VERSION).deb
endif

