## Build ##
To build the Python wheel, run the following commands:

`
pip install .
`

Alternatively, when debugging or developing the module, run:

`
pip install -e .
`

The flag `-e` will make sure any changes made are also reflected in the Python package.

## Running ##
### Using Python Package ###
Install the package with pip. To perform a detection on an image, run:

```python
from anpr_module import run

source = <PATH TO SOURCE IMAGE>
model = <PATH TO *.np MODEL>

registration = run(source, model)
```
