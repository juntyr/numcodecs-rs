# Try the `numcodecs-wasm-*` codecs using JupyterLite

/// details | **Warning:** JupyterLite may not work in every web browser
    type: warning
<img src="https://baseline.js.org/features/wasm-multi-memory/responsive-adaptive.svg" alt="Baseline Status: Multi-memory (WebAssembly)" style="width: 100%; height: auto;" />
///

<iframe id="try-jupyterlite" width="100%" height="750px"></iframe>

<script>
  window.addEventListener("load", () => {
    document.getElementById("try-jupyterlite").src = "https://lab.climet.eu/main/repl/index.html?kernel=python&toolbar=1&code=" + encodeURIComponent(`\
import numpy as np
from matplotlib import pyplot as plt
from numcodecs_wasm_sz3 import Sz3

x = np.linspace(-np.pi, np.pi)
y = np.sin(x)

sz3 = Sz3(eb_mode="abs", eb_abs=0.1)
enc = sz3.encode(y)
dec = sz3.decode(enc)

plt.plot(x, y, label="original")
plt.plot(x, dec, label="decompressed")
plt.legend()
plt.show()\
`) + "&pyodideKernelPackages=" + encodeURIComponent(JSON.stringify([
  // standard kernel packages
  "comm",
  "packaging",
  "ipython",
  "micropip",
  "pyodide-http",
  "widgetsnbextension",
  // example packages
  "matplotlib",
  "numcodecs-wasm",
  "numpy",
  // example package lazy dependencies
  "crc32c",
  "msgpack",
]));
  });
</script>
