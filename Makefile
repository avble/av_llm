AV_LLM_VERSION = 0.0.1-beta

all: package
	@echo "Create the package"	


compile:
	mkdir -p build && cmake . -B build -DCMAKE_BUILD_TYPE=Release &&  cmake --build build

package-prepare: compile
	@mkdir -p staging/usr/local/bin
	@cp -rf build/av_llm staging/usr/local/bin/

package: package-prepare
	mkdir -p output
	pkgbuild --root ./staging --identifier avble.llm.app --version 1.0 av_llm.pkg
	productbuild --distribution ./scripts/Distribution.xml --package-path ./ output/av_llm-universal-installer-${AV_LLM_VERSION}.pkg && rm av_llm.pkg

package-clean:
	rm -f ouput/*.pkg

clean:
	rm -rf staging
	cmake --build build --target clean
	rm -f output/*.pkg

install:
	sudo installer -pkg output/av_llm-universal-installer-${AV_LLM_VERSION}.pkg -target /