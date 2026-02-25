# Try the `numcodecs-wasm-*` codecs using JupyterLite

<iframe
    src="https://lab.climet.eu/main/repl/index.html?kernel=python&toolbar=1&code=import%20numpy%20as%20np%0Afrom%20matplotlib%20import%20pyplot%20as%20plt%0Afrom%20numcodecs_wasm_sz3%20import%20Sz3%0A%0Ax%20%3D%20np.linspace(-np.pi%2C%20np.pi)%0Ay%20%3D%20np.sin(x)%0A%0Asz3%20%3D%20Sz3(eb_mode%3D%22abs%22%2C%20eb_abs%3D0.1)%0Aenc%20%3D%20sz3.encode(y)%0Adec%20%3D%20sz3.decode(enc)%0A%0Aplt.plot(x%2C%20y%2C%20label%3D%22original%22)%0Aplt.plot(x%2C%20dec%2C%20label%3D%22decompressed%22)%0Aplt.legend()%0Aplt.show()"
    width="100%"
    height="750px"
></iframe>
