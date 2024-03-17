# shell.nix

with import <nixpkgs> { };
let
  pythonPackages = python3Packages;
in pkgs.mkShell rec {
  name = "tensorflowEnv";
  venvDir = "./.venv";
  buildInputs = [
    pythonPackages.python
    pythonPackages.matplotlib
    pythonPackages.numpy
    pythonPackages.scipy
    pythonPackages.tensorflow
    pythonPackages.tensorflow-probability
    pythonPackages.keras
    pythonPackages.jupyter
    pythonPackages.jupyterlab-git
    pythonPackages.tqdm
    pythonPackages.venvShellHook
  ];
}
