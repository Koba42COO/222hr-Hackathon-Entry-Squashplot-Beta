# Replit Configuration for SquashPlot
# ==================================

{ pkgs }: {
  deps = [
    # Python 3.9 or later
    pkgs.python39

    # System dependencies
    pkgs.python39Packages.pip
    pkgs.python39Packages.virtualenv

    # Compression libraries
    pkgs.zlib
    pkgs.bzip2
    pkgs.xz

    # Development tools
    pkgs.git
    pkgs.nodejs
    pkgs.yarn
  ];

  # Environment variables
  env = {
    PYTHONPATH = "/home/runner/${REPL_SLUG}";
    FLASK_ENV = "development";
  };
}
