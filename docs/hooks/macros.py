import importlib.metadata


def define_env(env):
    @env.macro
    def requirements():
        packages = dict()
        for package in sorted(set(importlib.metadata.packages_distributions().keys())):
            if package.startswith("numcodecs-wasm") or package.startswith("numcodecs_wasm"):
                packages[package] = importlib.metadata.version(package)
        return packages
