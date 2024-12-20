## Installation

```bash
pip install Cython
```

After cloning, create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```
Install zipline in develop mode with test dependencies:

```bash
setopt no_nomatch
pip install -e .[test]
setopt nomatch  # 恢复默认的通配符解析
```

During development, you can rebuild the C extensions by running

```bash
./rebuid-cython.sh
```
This step will generate a zipline_reloaded.egg-info directory(do not delete or commit )

install ta-lib

```bash
cd tools
./install_talib.sh
```


install pyfolio, tqdm
```
pip install tqdm
pip install git+https://github.com/stefan-jansen/pyfolio-reloaded.git
```