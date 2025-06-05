AV_LLM_VERSION = 0.0.1-beta
UNAME_S := $(shell uname -s)

all: package
	@echo "Create the package"	


compile:
	mkdir -p build && cmake . -B build -DCMAKE_BUILD_TYPE=Release &&  cmake --build build

package-prepare: compile
	@mkdir -p staging/usr/local/bin
	@mkdir -p staging/usr/local/etc/.av_llm
	@cp -rf build/av_llm staging/usr/local/bin/
	@if [ ! -f "staging/usr/local/etc/.av_llm/qwen1_5-0_5b-chat-q8_0.gguf" ]; then \
		wget -O staging/usr/local/etc/.av_llm/qwen1_5-0_5b-chat-q8_0.gguf https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q8_0.gguf; \
	fi
ifeq ($(UNAME_S), Linux)
	@mkdir -p staging/DEBIAN
	@cp scripts/control staging/DEBIAN/
endif

package: package-prepare
	mkdir -p output
ifeq ($(UNAME_S), Darwin)
	pkgbuild --root ./staging --identifier avble.llm.app --version 1.0 av_llm.pkg
	productbuild --distribution ./scripts/Distribution.xml --package-path ./ output/av_llm-universal-installer-${AV_LLM_VERSION}.pkg && rm av_llm.pkg
else ifeq ($(UNAME_S), Linux)
	dpkg-deb --build staging output/av_llm-linux-installer-${AV_LLM_VERSION}.deb
else 
	echo "windows build"
endif

package-clean:
	rm -f ouput/*.pkg

clean:
	rm -rf staging
	cmake --build build --target clean
	rm -f output/*.pkg

install-test:
ifeq ($(UNAME_S), Darwin)
	sudo installer -pkg output/av_llm-universal-installer-${AV_LLM_VERSION}.pkg -target /
else ifeq ($(UNAME_S), Linux)
	sudo dpkg -i output/av_llm-linux-installer-${AV_LLM_VERSION}.deb
else 
	echo "windows build"	
endif