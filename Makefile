PYTHON        := $(shell which python3)
PY_INCLUDES   := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PB11_INCLUDES := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX    := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CXXFLAGS      := -O2 -fPIC -shared -std=c++14 -Wall -undefined dynamic_lookup
BACKEND_DIR   := minigrad/backend_ndarray

.PHONY: all cpu clean test

all: cpu

cpu:
	@echo "Using Python: $(PYTHON)"
	@echo "Building CPU backend..."
	$(CXX) $(CXXFLAGS) \
		-I$(PY_INCLUDES) \
		-I$(PB11_INCLUDES) \
		$(BACKEND_DIR)/ndarray_backend_cpu.cc \
		-o $(BACKEND_DIR)/ndarray_backend_cpu$(EXT_SUFFIX)
	@echo "Done."

clean:
	rm -f $(BACKEND_DIR)/*.so $(BACKEND_DIR)/*.dylib

test:
	$(PYTHON) -m pytest tests/ -v

install-deps:
	$(PYTHON) -m pip install numpy pytest pybind11