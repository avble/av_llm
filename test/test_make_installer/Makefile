# first make macOS installer package

all: package
	@echo "Create the package"	


compile:
	mkdir -p build && cmake . -B build &&  cmake --build build

package-prepare: compile
	@mkdir -p staging/usr/local/bin
	@cp -rf build/av_llm_installer staging/usr/local/bin/

package: package-prepare
	pkgbuild --root ./staging --identifier avble.llm.app --version 1.0 av_llm_installer.pkg
	productbuild --distribution Distribution.xml --package-path ./ av_llm_installer-installer.pkg

clean:
	rm -rf staging
	cmake --build build --target clean
	rm -f *.pkg
