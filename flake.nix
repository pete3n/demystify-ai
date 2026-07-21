{
  description = "Bigram language model + Ollama logprobs inspector — educational tkinter demos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python with tkinter support.
        # nixpkgs splits tkinter out as python3.pkgs.tkinter which pulls in
        # tcl/tk and compiles the _tkinter C extension into the interpreter.
        python = pkgs.python3.withPackages (ps: [
          ps.tkinter
        ]);

        mkApp = script: {
          type = "app";
          program = "${pkgs.writeShellScript "run-${script}" ''
            exec ${python}/bin/python3 ${./${script}}
          ''}";
        };

      in
      {
        # `nix develop` — shell with python+tkinter on PATH
        devShells.default = pkgs.mkShell {
          packages = [ python ];
          shellHook = ''
            echo "LLM demo environment"
            echo "Python: $(python3 --version)"
            echo ""
            echo "  python3 bigram_lm.py        — bigram model demo"
            echo "  python3 ollama_logprobs.py  — Ollama logprobs inspector"
          '';
        };

        # `nix run`               — bigram demo (default)
        # `nix run .#bigram`      — bigram demo (explicit)
        # `nix run .#logprobs`    — Ollama logprobs inspector
        apps = {
          default = mkApp "bigram_lm.py";
          bigram = mkApp "bigram_lm.py";
          logprobs = mkApp "ollama_logprobs.py";
        };
      }
    );
}
