with import <nixpkgs> {};
mkShell {
	buildInputs = [
		(python310.withPackages(ps: with ps; [ safe-pysha3 setuptools requests ]))
		poetry
		graphviz
		gprof2dot
	];
	LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
}
