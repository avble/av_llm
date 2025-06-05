AV_LLM_VERSION = 0.0.1-beta

UNAME_S := $(shell uname -s)

# ifeq ($(UNAME_S), Darwin)
# 	OS := macOS
# else ifeq ($(UNAME_S, Linux))
# 	OS := Linux
# else
# 	echo "windows build"
# endif

all: package
	@echo "Create the package"	


compile:
	mkdir -p build && cmake . -B build -DCMAKE_BUILD_TYPE=Release &&  cmake --build build

package-prepare: compile
	@mkdir -p staging/usr/local/bin
	@cp -rf build/av_llm staging/usr/local/bin/
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

install:
	sudo installer -pkg output/av_llm-universal-installer-${AV_LLM_VERSION}.pkg -target /